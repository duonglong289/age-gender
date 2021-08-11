import numpy as np 
import random
import cv2
from PIL import Image
import os 
from .mbnetv2 import mobilenet_v2
from .mbnetv1 import mobilenet_v1
from .resnet import resnet50, resnet34, resnet18
import logging
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn 
import torchvision.transforms as T

import models.metrics as metrics
import models.cost_fn as cost_fn

from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
saved_statedict_path = ""

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

        self.transformer = self.build_transform()
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
            self.model = mobilenet_v2(pretrained=pretrained, **kwargs)
        elif model_name == "mobilenet_v1":
            self.model = mobilenet_v1(**kwargs)
        elif model_name == "resnet50":
            self.model = resnet50(pretrained=pretrained, **kwargs)
        elif model_name == "resnet34":
            self.model = resnet34(pretrained=pretrained, **kwargs)
        elif model_name == "resnet50":
            self.model = resnet18(pretrained=pretrained, **kwargs)
        else:
            raise ValueError("Do not support model {}!".format(model_name))
        print(f"You are using model {model_name}")
        
            
            
        # Status of age and gender classifiers
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

    
    def train(self, num_epochs, learning_rate, freeze=False, load_statedict=False, verbose=True):
        self._init_param(learning_rate)
        
        # Load saved statedict if colab was corrupted 
        if load_statedict:
            checkpoint = torch.load(saved_statedict_path)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

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
            # Learning rate decay
            # self._init_param(learning_rate/(1+epoch))
            running_loss, loss_ages, loss_genders = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
            train_loss, loss_age, loss_gender = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
            self.model.train()
            self.epoch_count += 1
            for image_path, image, label in tqdm(self.train_generator, desc="Epoch {}:".format(self.epoch_count)):
                image = image.to(self.device)                               
                label_age, label_gender = label
                # label_age = torch.stack(label_age, dim=1).to(self.device)
                label_age = torch.LongTensor(label_age).to(self.device)
                label_gender = torch.LongTensor(label_gender).to(self.device)
    
                output = self.model(image)

                if self.age_classifier and self.gender_classifier:
                    score_age, prob_age, pred_gender = output
                    loss_age = cost_fn.cost_ordinary(score_age, label_age)/10
                    loss_gender = cost_fn.cost_nll(pred_gender, label_gender)       
                    train_loss = loss_age + loss_gender
                elif self.age_classifier and not self.gender_classifier:
                    score_age, prob_age = output 
                    loss_age = cost_fn.cost_ordinary(score_age, label_age)
                    train_loss = loss_age
                else:
                    pred_gender = output
                    loss_gender = cost_fn.cost_nll(pred_gender, label_gender)
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
            self.writer.add_scalar("Age_loss", loss_ages, self.epoch_count)
            self.writer.add_scalar("Gender_loss", loss_genders, self.epoch_count)
            self.writer.add_scalar("Train_loss", running_loss, self.epoch_count)
            
            # Evaluate
            mae_age, acc_gender, loss_age_val, loss_gender_val, val_loss = self._validate(epoch+1)
            
            # Monitor
            if verbose:
                logger.info(f"Epoch {self.epoch_count}: \
                        \n- Loss age train: {loss_ages} \
                        \n- Loss gender train: {loss_genders} \
                        \n- Loss train: {running_loss} \
                        \n- Loss age val: {loss_age_val} \
                        \n- Loss gender val: {loss_gender_val} \
                        \n- Loss val: {val_loss} \
                        \n- MAE age: {mae_age} \
                        \n- Acc gender: {acc_gender}" \
                    )
            
            # Save model
            self.save_statedict(mae_age, acc_gender)
            self.writer.export_scalars_to_json(os.path.join(self.log, "tensorboardX.json"))

        
    def _validate(self, epoch):
        ''' Validate data each epoch
        '''
        self.model.eval()
        mae_age, mse_age, acc_gender = 0., 0., 0.
        val_losses, loss_ages, loss_genders = torch.Tensor([0]), torch.Tensor([0]),torch.Tensor([0])
        val_loss, loss_age, loss_gender = torch.Tensor([0]), torch.Tensor([0]),torch.Tensor([0])
        with torch.no_grad():
            for image_path, inputs, labels in tqdm(self.val_generator, desc=f"Epoch {self.epoch_count}:"):
                inputs = inputs.to(self.device)
                label_age, label_gender = labels
                label_age = torch.LongTensor(label_age).to(self.device)
                # label_age = torch.stack(label_age, dim=1).to(self.device)
                label_gender = torch.LongTensor(label_gender).to(self.device)

                # Predict
                output = self.model(inputs)

                if self.age_classifier and self.gender_classifier:
                    score_age, prob_age, pred_gender = output
                    # loss_age = cost_fn.cost_nll(prob_age, label_age) 
                    loss_age = cost_fn.cost_ordinary(score_age, label_age)/10
                    loss_gender = cost_fn.cost_nll(pred_gender, label_gender)       
                    val_loss = loss_age + loss_gender

                elif self.age_classifier and not self.gender_classifier:
                    score_age, prob_age = output 
                    # loss_age = cost_fn.cost_nll(prob_age, label_age)
                    loss_age = cost_fn.cost_ordinary(score_age, label_age)
                    val_loss = loss_age

                else:
                    pred_gender = output
                    loss_gender = cost_fn.cost_nll(pred_gender, label_gender)       
                    val_loss = loss_gender

                loss_ages += loss_age.item()
                loss_genders += loss_gender.item()
                val_losses += val_loss.item()

                # compute mae and mse with age label
                if self.age_classifier:
                    mae = metrics.compute_mae_mse(prob_age, label_age)
                    mae_age += mae
                    # mse_age += mse

                # compute accuracy with gender label
                if self.gender_classifier:
                    pred_gender = torch.exp(pred_gender)
                    top_prob_gender, top_class_gender = pred_gender.topk(1, dim=1)
                    equals = top_class_gender == label_gender.view(*top_class_gender.shape)
                    acc_gender += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Validation loss
            loss_ages = loss_ages.item()/len(self.val_generator)
            loss_genders = loss_genders.item()/len(self.val_generator)
            val_losses = val_losses.item()/len(self.val_generator)

            # Mean mae, mse, 
            mae_age = mae_age/len(self.val_generator)
            # mse_age = mse_age.float()/len(self.val_generator)
            acc_gender = acc_gender/len(self.val_generator)

            # Write tensorboard
            self.writer.add_scalar("MAE_age_validation", mae_age, self.epoch_count)
            # self.writer.add_scalar("MSE age validation". mse, epoch)
            self.writer.add_scalar("Accuracy_gender_validation", acc_gender, self.epoch_count)
            self.writer.add_scalar("Validation_age_loss", loss_ages, self.epoch_count)
            self.writer.add_scalar("Validation_gender_loss", loss_genders, self.epoch_count)
            self.writer.add_scalar("Validation_loss", val_losses, self.epoch_count)

        return mae_age, acc_gender, loss_ages, loss_genders, val_losses


    def age_to_level(self, age):
        ''' Convert age to levels, for ordinary regression task
        '''
        level = [1]*age + [0]*(NUM_AGE_CLASSES - 1 - age)
        return level

    def age_to_class(self, age):
        if age == 0:
            return "[0-5]"
        elif age == 1:
            return "[5-10]"
        elif age == 2:
            return "[10-14]"
        elif age == 3:
            return "[14-18]"
        elif age == 4:
            return "[18-21]"
        elif age == 5:
            return "[21-25]"
        elif age == 6:
            return "[25-29]"
        elif age == 7:
            return "[29-34]"
        elif age == 8:
            return "[34-38]"
        elif age == 9:
            return "[38-42]"
        elif age == 10:
            return "[42-46]"
        elif age == 11:
            return "[46-50]"
        elif age == 12:
            return "[50-55]"
        elif age == 13:
            return "[55-60]"
        elif age == 14:
            return "[60-65]"
        else:
            return "[>65]"


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
        torch.save({
            "model": self.model.state_dict(), 
            "optimizer": self.optimizer.state_dict()
        }, model_path)

    
    def load_statedict(self, state_dict_path, device="cpu"):
        self.device = torch.device(device)
        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.model.eval().to(self.device)
        self.model.load_state_dict(state_dict)
        return self.model

    def build_transform(self):
        PIXEL_MEAN = [0.5, 0.5, 0.5]
        PIXEL_STD =[0.5, 0.5, 0.5]
        normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        
        transform = T.Compose([
                T.ToPILImage(),
                T.Resize([224, 224], Image.BILINEAR),
                T.ToTensor(),
                normalize_transform])
        return transform

    def predict_image(self, image:np.ndarray):
        '''Predict age and gender
        Inputs
        ------
            image: (np.ndarray) RGB face image
        Returns
        -------
            age, gender
        '''
        image = self.transformer(image)
        image = image.unsqueeze(0)

        age_score, age_prob, gender_prob = self.model(image)
        #import ipdb; ipdb.set_trace()
        # Predicted age
        prob_levels = age_prob > 0.5
        predicted_ages = torch.sum(prob_levels, dim=1)
        # Predicted gender
        gender_prob = torch.exp(gender_prob)
        top_gender_prob, top_gender_class = gender_prob.topk(1, dim=1)
        if top_gender_class ==0:
            gender = "Female"
        else:
            gender = "Male"
        
        return predicted_ages, gender

        

    
