import numpy as np 
import random
import cv2
from PIL import Image
import os 
from models.mbnetv2 import mobilenet_v2
import logging
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn 
import torchvision.transforms as transforms

from data_generator import DatasetLoader

from tensorboardX import SummaryWriter
from clearml import Task 

from models.net import ModelAgeGender

def train(args):
    # Model
    model_name = args.model_name
    widen_factor = args.widen_factor

    # Params
    batch_size = args.batch_size
    log_dir = args.logs
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    init_lr = args.init_lr
    task_name = args.task_name
    
    # Init dataset
    dataset_dir = args.dataset
    # import ipdb; ipdb.set_trace()

    train_loader = DatasetLoader(dataset_dir, "train")
    val_loader = DatasetLoader(dataset_dir, "val")


    # Init model
    age_gender_model = ModelAgeGender(log=log_dir, task_name=task_name)
    age_gender_model.init_model(model_name=model_name, widen_factor=widen_factor, pretrained=True, num_age_classes=16)

    age_gender_model.load_dataset((train_loader, val_loader), batch_size=batch_size, num_workers=num_workers)

    # Train 5 epoch with freezed backbone
    age_gender_model.train(num_epochs=5, learning_rate=init_lr, freeze=True)
    # Then unfreeze all layers
    age_gender_model.train(num_epochs=num_epochs-5, learning_rate=init_lr/2, freeze=False)

    age_gender_model.save_model(model_name="last.pt")
    age_gender_model.writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Liveness Detection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset consist of 'train' and 'val'")
    parser.add_argument("--model_name", type=str, default="mobilenet_v2", help="Model name")
    parser.add_argument("--widen_factor", type=int, default=1, help="Factor of model size")
    parser.add_argument("--logs", type = str, required=True, help="Path saved model")
    parser.add_argument("--num_epochs", type=int, default=30, help="num epoch")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--init_lr", type=float, default=0.002, help="Starting learning rate")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker process data")
    parser.add_argument("--task_name", type=str, default=None, help="Task name in ClearML dashboard")
    args = parser.parse_args()

    
    train(args)
