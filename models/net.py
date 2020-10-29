import numpy as np 
import random
import cv2
from PIL import Image
import os 
from .mbnetv2 import mobilenet_v2
import logging
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn 
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter


class ModelAgeGender:
    def __init__(self, device="cuda", log="./log", **kwargs):
        if os.path.isdir(log):
            now = datetime.now()
            now_str = now.strftime("%d%m%Y_%H%M%S")
            # self.log = os.path.join(log, now_str)
            self.log = "{}_{}".format(log, now_str)
            os.makedirs(self.log, exist_ok=True)
        else:
            self.log = log
            os.makedirs(self.log, exist_ok=True)

        self.epoch_count = 0
        self.writer = SummaryWriter()
        

    def __repr__(self):
        return self.model.__repr__()


    def _init_param(self, learning_rate=0.002):
        w_decay = 0.005
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        

    def init_model(self, model_name="mobilenet_v2", pretrained=True, **kwargs):
        ''' Init model before training
        '''
        if model_name == "mobilenet_v2":
            self.model = mobilenet_v2(pretrained=True, **kwargs)
        else:
            raise ValueError("Do not support model {}!".format(model_name))
        self.age_classifier = self.model.age_cls
        self.gender_classifier = self.model.gender_cls
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def load_dataset(self, data_loader, batch_size=1, num_workers=8):
        params = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers
        }
        # Load dataset
        train_loader, val_loader = data_loader
        # self.train_generator = DatasetLoader(dataset_dir, "train")
        self.train_generator = torch.utils.data.DataLoader(train_loader, **params)
        # self.val_generator = DatasetLoader(dataset_dir, "val")
        self.val_generator = torch.utils.data.DataLoader(val_loader, **params)

    
    def train(self, num_epochs, learning_rate, freeze=False, verbose=True):
        self._init_param(learning_rate)
        # Freeze backbone
        if freeze:
            # Freeze all layers
            for params in self.model.parameters():
                params.requires_grad = False
            # Unfreeze 2 heads of age and gender branchs
            if self.age_classifier:
                for params in self.model.classifier_age.parameters():
                    params.requires_grad = True
            if self.gender_classifier:
                for params in self.model.classifier_gender.parameters():
                    params.requires_grad = True
        # Unfreeze all layers
        else:
            for params in self.model.parameters():
                params.requires_grad = True      

        # Train mode
        for epoch in range(num_epochs):
            running_loss, loss_ages, loss_genders = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
            train_loss, loss_age, loss_gender = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
            self.model.train()
            self.epoch_count += 1
            for image, label in self.train_generator:
                image = image.to(self.device)                               
                label_age, label_gender = label
                label_age = torch.LongTensor(label_age).to(self.device)
                label_gender = torch.LongTensor(label_gender).to(self.device)
    
                output = self.model(image)

                if self.age_classifier and self.gender_classifier:
                    score_age, pred_age, pred_gender = output
                    loss_age = self.cost_nll(pred_age, label_age)               
                    loss_gender = self.cost_nll(pred_gender, label_gender)       
                    train_loss = loss_age + loss_gender
                elif self.age_classifier and not self.gender_classifier:
                    score_age, pred_age = output 
                    loss_age = self.cost_nll(pred_age, label_age)                    
                    train_loss = loss_age
                else:
                    pred_gender = output
                    loss_gender = self.cost_nll(pred_gender, label_gender)
                    train_loss = loss_gender

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                loss_ages += loss_age.item()
                loss_genders += loss_gender.item()
                running_loss += train_loss.item()

            # Compute loss
            loss_ages = loss_ages.item()/len(self.train_generator)
            loss_genders = loss_genders.item()/len(self.train_generator)
            running_loss = running_loss.item()/len(self.train_generator)
            # Write tensorboard
            self.writer.add_scalar("Age loss", loss_ages, epoch+1)
            self.writer.add_scalar("Gender loss", loss_genders, epoch+1)
            self.writer.add_scalar("Train loss", running_loss, epoch+1)
            
            # Evaluate
            mae_age, acc_gender = self._validate(epoch+1)
            
            # Monitor
            if verbose:
                print("Epoch {}: Loss age: {} - Loss gender: {} - Loss train: {} - MAE age: {} - Acc gender: {}"
                    .format(self.epoch_count, loss_ages, loss_genders, running_loss, mae_age, acc_gender))
            
            # Save model
            self.save_statedict(mae_age, acc_gender)
            self.writer.export_scalars_to_json(os.path.join(self.log, "tensorboardX.json"))

        
    def _validate(self, epoch):
        ''' Validate data each epoch
        '''
        self.model.eval()
        mae_age, mse_age, acc_gender = 0., 0., 0.

        with torch.no_grad():
            for inputs, labels in self.val_generator:
                inputs = inputs.to(self.device)
                label_age, label_gender = labels
                label_age = label_age.to(self.device)
                label_gender = label_gender.to(self.device)

                # Predict
                output = self.model(inputs)

                if self.age_classifier and self.gender_classifier:
                    score_age, pred_age, pred_gender = output
                    loss_age = self.cost_nll(pred_age, label_age)               
                    loss_gender = self.cost_nll(pred_gender, label_gender)       
                    val_loss = loss_age + loss_gender
                elif self.age_classifier and not self.gender_classifier:
                    score_age, pred_age = output 
                    loss_age = self.cost_nll(pred_age, label_age)                    
                    val_loss = loss_age
                else:
                    pred_gender = output
                    loss_gender = criterion_gender(pred_gender, label_gender)       
                    val_loss = loss_gender

                # compute mae and mse with age label
                if self.age_classifier:
                    mae = self.compute_mae_mse(pred_age.topk(1, dim=1)[1], label_age)
                    mae_age += mae
                    # mse_age += mse

                # compute accuracy with gender label
                if self.gender_classifier:
                    pred_gender = torch.exp(pred_gender)
                    top_prob_gender, top_class_gender = pred_gender.topk(1, dim=1)
                    equals = top_class_gender == label_gender.view(*top_class_gender.shape)
                    acc_gender += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Mean mae, mse, l
            mae_age = mae_age/len(self.val_generator)
            # mse_age = mse_age.float()/len(self.val_generator)
            acc_gender = acc_gender/len(self.val_generator)

            # Write tensorboard
            self.writer.add_scalar("MAE age validation", mae_age, epoch)
            # self.writer.add_scalar("MSE age validation". mse, epoch)
            self.writer.add_scalar("Accuracy gender validation", acc_gender, epoch)
        
        return mae_age, acc_gender


    def compute_mae_mse(self, predicted_age, target_age):
        ''' Compute mean absolute error, mean square error
        '''
        mae, mse = 0., 0.
        
        
        predicted_age = predicted_age.view(-1, 1)
        target_age = target_age.view(-1, 1)# .type(torch.FloatTensor).item()
        mae = torch.sum(torch.abs(predicted_age - target_age))
        # mse = torch.sum((predicted_age - target_age)**2)
        # return mae, mse           
        return mae


    def age_to_level(self, age):
        ''' Convert age to levels, for ordinary regression task
        '''
        level = [1]*age + [0]*[NUM_AGE_CLASSES - 1 - age]
        return level


    def cost_ordinary(self, predicted, groundtruth):
        ''' Compute orinary regression loss
        '''
        logits = predicted
        levels = groundtruth
        cost = (-torch.sum(F.log_softmax(logits, dim=2)[:, :, 1]*levels 
                        + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels), dim=1))
        return cost


    def cost_ce(self, predicted, groundtruth):
        '''Compute cross entropy loss
        '''
        cost = F.cross_entropy(predicted, groundtruth)
        return cost


    def cost_nll(self, predicted, groundtruth):
        loss = nn.NLLLoss()
        cost = loss(predicted, groundtruth)
        return cost


    def save_model(self, mae=0, acc=0, model_name=None):
        self.model.eval()
        if model_name is None:
            model_path = os.path.join(self.log, "{}_{}_gender_{}_age.pt".format(self.epoch_count, acc, mae))
        else: 
            model_path = model_name
        torch.save(self.model, model_path)


    def save_statedict(self, mae=0, acc=0):
        self.model.eval()
        model_path = os.path.join(self.log, "{}_{}_gender_{}_age.pt".format(self.epoch_count, acc, mae))
        torch.save(self.model.state_dict(), model_path)

    
    def load_statedict(self, state_dict_path, device="cpu"):
        self.device = torch.device(device)
        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.model.eval().to(self.device)
        self.model.load_state_dict(state_dict)




    def predict_image(self, image:np.ndarray):
        '''Predict age and gender
        Inputs
        ------
            image: (np.ndarray) RGB face image
        Returns
        -------
            age, gender
        '''
        pass

        

    
