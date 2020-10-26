import numpy as np 
import random
import cv2
from PIL import Image
import os 
from model import mobilenet_v2
import logging
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn 
import torchvision.transforms as transforms

from data_generator import DatasetLoader

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


    def _init_param(self):
        w_decay = 0.005
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.02, weight_decay=w_decay)
        

    def init_model(self, model_name="mobilenet_v2", pretrained=True, **kwargs):
        if model_name == "mobilenet_v2":
            self.model = mobilenet_v2(pretrained=True, **kwargs)
        else:
            raise ValueError("Do not support model {}!".format(model_name))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._init_param()


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
        # Freeze backbone
        if freeze:
            for params in self.model.parameters():
                # if params in self.model.classifier_age.parameters() or params in self.model.classifier_gender.parameters():
                params.requires_grad = False
            for params in self.model.classifier_age.parameters():
                params.requires_grad = True
            for params in self.model.classifier_gender.parameters():
                params.requires_grad = True

        else:
            for params in self.model.parameters():
                params.requires_grad = True
        
        loss_ages, loss_genders, running_losses = 0, 0, 0
        train_loss = []
        loss_age, loss_gender, running_loss = 0., 0., 0.
        # Train mode
        for epoch in range(num_epochs):
            self.model.train()
            self.epoch_count += 1
            for image, label in self.train_generator:
                image = image.to(self.device) # change this
                label_age, label_gender = label
                
                label_age = torch.LongTensor(label_age).to(self.device)
                label_gender = torch.LongTensor(label_gender).to(self.device)
                self.optimizer.zero_grad()
    
                output = self.model(image)

                score_age, pred_age, pred_gender = output
                loss_age = self.cost_nll(pred_age, label_age)         # Change this
                loss_gender = self.cost_nll(pred_gender, label_gender)      # Change this
                train_loss = loss_age + loss_gender

                train_loss.backward()
                self.optimizer.step()

                loss_ages += loss_age.item()
                loss_genders += loss_gender.item()
                running_loss += train_loss.item()

            # Compute loss
            loss_ages = loss_ages/len(self.train_generator)
            loss_genders = loss_genders/len(self.train_generator)
            running_losses = running_loss/len(self.train_generator)
            # Write tensorboard
            self.writer.add_scalar("Age loss", loss_ages, epoch+1)
            self.writer.add_scalar("Gender loss", loss_genders, epoch+1)
            self.writer.add_scalar("Train loss", running_losses, epoch+1)
            
            # Evaluate
            mae_age, acc_gender = self._validate(epoch+1)
            
            # Monitor
            if verbose:
                print("Epoch {}: Loss age: {} - Loss gender: {} - Loss train: {} - MAE age: {} - Acc gender: {}"
                    .format(self.epoch_count, loss_ages, loss_genders, running_losses, mae_age, acc_gender))
            
            # Save model
            self.save_statedict(mae_age, acc_gender)

        
    def _validate(self, epoch):
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
                score_age, pred_age, pred_gender = output

                # compute mae and mse with age label
                mae = self.compute_mae_mse(pred_age.topk(1, dim=1)[1], label_age)
                mae_age += mae
                # mse_age += mse

                # compute accuracy with gender label
                pred_gender = torch.exp(pred_gender)
                top_prob_gender, top_class_gender = pred_gender.topk(1, dim=1)
                equals = top_class_gender == label_gender.view(*top_class_gender.shape)
                acc_gender += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Mean mae, mse, l
            mae_age = mae_age/len(self.val_generator)
            # mse_age = mse_age.float()/len(self.val_generator)
            acc_gender = acc_gender/len(self.val_generator)

            # Write tensorboard
            self.writer.add_scalar("MAE age validation", mae, epoch)
            # self.writer.add_scalar("MSE age validation". mse, epoch)
            self.writer.add_scalar("Accuracy gender validation", acc_gender, epoch)
        
        return mae_age, acc_gender


    def compute_mae_mse(self, predicted_age, target_age):
        mae, mse = 0., 0.
        predicted_age = predicted_age.view(-1, 1)
        target_age = predicted_age.view(-1, 1)# .type(torch.FloatTensor).item()
        # mse = torch.sum((predicted_age - target_age)**2)
        # return mae, mse           
        return mae


    def age_to_level(self, age):
        level = [1]*age + [0]*[NUM_AGE_CLASSES - 1 - age]
        return level


    def cost_ordinary(self, predicted, groundtruth):
        logits = predicted
        levels = groundtruth
        cost = (-torch.sum(F.log_softmax(logits, dim=2)[:, :, 1]*levels 
                        + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels), dim=1))
        return cost


    def cost_ce(self, predicted, groundtruth):
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
        
def main(args):
    # Params
    batch_size = args.batch_size
    log_dir = args.logs
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    init_lr = args.init_lr
    
    # Init dataset
    dataset_dir = args.dataset
    train_loader = DatasetLoader(dataset_dir, "train")
    val_loader = DatasetLoader(dataset_dir, "val")
    num_age_classes=train_loader.num_age_classes

    # Init model
    age_gender_model = ModelAgeGender(log=log_dir)
    age_gender_model.init_model(num_age_classes=17)

    age_gender_model.load_dataset((train_loader, val_loader), batch_size=batch_size, num_workers=num_workers)

    # Train 5 epoch with freezed backbone
    age_gender_model.train(num_epochs=5, learning_rate=init_lr, freeze=True)
    # Then unfreeze all layers
    age_gender_model.train(num_epochs=num_epochs-5, learning_rate=init_lr/2, freeze=True)

    age_gender_model.save_model(model_name="last.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Liveness Detection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset consist of 'train' and 'val'")
    parser.add_argument("--logs", type = str, required=True, help="Path saved model")
    parser.add_argument("--num_epochs", type=int, default=30, help="batch size")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--init_lr", type=float, default=0.002, help="Starting learning rate")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker process data")
    args = parser.parse_args()

    main(args)
    
