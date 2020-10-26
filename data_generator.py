import glob
import os
import numpy as np 
import torch 
import imgaug.augmenters as iaa 
import imgaug as ia 
import random

from PIL import Image
import cv2
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from custom_augmentation import (LightFlare, ParallelLight, SpotLight)

aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Sometimes(0.3, LightFlare()),
        iaa.Sometimes(0.3, ParallelLight()),
        iaa.Sometimes(0.3, SpotLight())
    ]),
    iaa.Sometimes(0.05, iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True)),
    iaa.Sometimes(0.3, 
        iaa.OneOf([
            iaa.GaussianBlur((0, 1.5)),
            iaa.AverageBlur(k=(3, 5)),
            iaa.MedianBlur(k=3),
            iaa.MotionBlur(k=(3, 7), angle=(-45, 45))
        ])
    ),
    iaa.Sometimes(0.2, 
        iaa.Affine(
            scale=(0.001, 0.05),
            translate_percent=(0.01),
            rotate=(-10, 10),
            shear=(-5, 5)
        )    
    )
])

class DatasetLoader(Dataset):
    def __init__(self, dataDir, stage, batch_size=1, image_size=224):
        self.batch_size = batch_size
        self.image_size = image_size
        self.stage = stage
        self.image_path_and_type = []
        self._load_dataset(dataDir)
        self.transform_data =  self.build_transforms()

        self.image_num = len(self.image_path_and_type)
        self.indices = np.random.permutation(self.image_num)
        self.num_age_classes = 16
        

    def __len__(self):
        return self.image_num // self.batch_size


    def __getitem__(self, idx):
        batch_size = self.batch_size

        image_path, label = self.image_path_and_type[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        age, gender = label
        try:
            img = aug.augment(image=img)
        except:
            img = img
                
        image = Image.fromarray(img)
        X = self.transform_data(image)
        y = (age, gender)
        return X, y
    

    def build_transforms(self, PIXEL_MEAN = [0.5, 0.5, 0.5], PIXEL_STD =[0.5, 0.5, 0.5]):
        normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        if self.stage=="train":
            transform = T.Compose([
                    T.Resize([self.image_size,self.image_size]),
                    T.ColorJitter(brightness=(0.8, 1.2)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize_transform])
        else:
            transform = T.Compose([
                    T.Resize([self.image_size,self.image_size]),
                    T.ToTensor(),
                    normalize_transform])

        return transform


    def age_to_cls(self, age):
        if 0 <= age < 5:
            age_cls = 0
        elif 5 <= age < 10:
            age_cls = 1
        elif 10 <= age < 14:
            age_cls = 2
        elif 14 <= age < 18:
            age_cls = 3
        elif 18 <= age < 21:
            age_cls = 4
        elif 21 <= age < 25:
            age_cls = 5
        elif 25 <= age < 29:
            age_cls = 6
        elif 29 <= age < 34:
            age_cls = 7
        elif 34 <= age < 38:
            age_cls = 8
        elif 38 <= age < 42:
            age_cls = 9
        elif 42 <= age < 46:
            age_cls = 10
        elif 46 <= age < 50:
            age_cls = 11
        elif 50 <= age < 55:
            age_cls = 12
        elif 55 <= age < 60:
            age_cls = 13
        elif 60 <= age < 65:
            age_cls = 14
        elif 65 <= age:
            age_cls = 15
        return age_cls


    def _load_dataset(self, dataDir):  
        random.seed(42)

        image_dir = Path(dataDir)
        img_list = image_dir.glob("*")
        img_list = list(img_list)
        train_list = random.sample(img_list, int(0.8*len(img_list)))
        val_list = list(set(img_list) - set(train_list))
        if self.stage == "train":
            data_imgs = train_list
        else:
            data_imgs = val_list
        age_count = [0]*16
        gender_count = [0]*2
        for image_path in data_imgs:
            image_name = image_path.name 
            age =image_name.split("A")[1].split(".")[0].split("G")[0]
            gender =image_name.split("A")[1].split(".")[0].split("G")[1]
            age_cls = self.age_to_cls(int(age))
            gender_cls = int(gender)
            age_count[age_cls] += 1
            gender_count[gender_cls] += 1
            labels = (age_cls, gender_cls)
            if image_path.is_file():
                self.image_path_and_type.append([str(image_path), labels])
        print(self.stage)
        print("Number images:", len(self.image_path_and_type))
        print("Class age",age_count)
        print("Class gender", gender_count)

if __name__ == "__main__":
    dataset = DatasetLoader("dataset/all_faces", "train")
    dataset = DatasetLoader("dataset/all_faces", "val")
    