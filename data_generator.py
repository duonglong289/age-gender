import glob
import os
import numpy as np 
import imgaug.augmenters as iaa 
import imgaug as ia 
import random
import logging
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from custom_augmentation import (LightFlare, ParallelLight, SpotLight)
logger = logging.getLogger()
# logger.setLevel(os.environ.get ("LOGLEVEL", "INFO"))

aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Sometimes(0.2, LightFlare()),
        iaa.Sometimes(0.2, ParallelLight()),
        iaa.Sometimes(0.2, SpotLight())
    ]),
    iaa.Sometimes(0.025, iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True)),
    iaa.Sometimes(0.2, 
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
    def __init__(self, dataDir, stage, batch_size=1, image_size=224, num_age_classes=16):
        self.batch_size = batch_size
        self.image_size = image_size
        self.stage = stage
        self.image_path_and_type = []
        self._load_dataset(dataDir)
        self.transform_data =  self.build_transforms()
        self.num_age_classes = num_age_classes
        self.image_num = len(self.image_path_and_type)
        self.indices = np.random.permutation(self.image_num)
        #self.age_count = [0]*num_age_classes
        #self.gender_count = [0]*2

        

    def __len__(self):
        return self.image_num // self.batch_size


    def __getitem__(self, idx):
        batch_size = self.batch_size

        image_path, label = self.image_path_and_type[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        age, gender = label
        age = torch.LongTensor(age)

            
        # try:
        #     img = aug.augment(image=img)
        # except:
        #     img = img
                
        image = Image.fromarray(img)
        X = self.transform_data(image)
        #X = image
        #X = self.data_augumentation(X)
        y = (age, gender)
        return X, y
    

    def build_transforms(self, PIXEL_MEAN = [0.5, 0.5, 0.5], PIXEL_STD =[0.5, 0.5, 0.5]):
        normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        
        if self.stage=="train":
            transform = T.Compose([
                    T.Resize([self.image_size,self.image_size], Image.BICUBIC),
                    #T.Resize([self.image_size,self.image_size]),
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
    '''
    def data_augumentation(self, image):
        if self.stage == "train":
            aug_img = LightFlare().augment_image(image)
        return aug_img
    '''

    def age_to_cls(self, age):
        if 0 <= age < 5:
            return 0
        elif 5 <= age < 10:
            return 1
        elif 10 <= age < 14:
            return 2
        elif 14 <= age < 18:
            return 3
        elif 18 <= age < 21:
            return 4
        elif 21 <= age < 25:
            return 5
        elif 25 <= age < 29:
            return 6
        elif 29 <= age < 34:
            return 7
        elif 34 <= age < 38:
            return 8
        elif 38 <= age < 42:
            return 9
        elif 42 <= age < 46:
            return 10
        elif 46 <= age < 50:
            return 11
        elif 50 <= age < 55:
            return 12
        elif 55 <= age < 60:
            return 13
        elif 60 <= age < 65:
            return 14
        elif 65 <= age:
            return 15
    
    def age_to_level(self, age):
        age_level = [1]*age + [0]*(16 - age)
        return age_level

    def _load_dataset(self, dataDir):  
        global age_cls
        random.seed(42)
        
        age_count = [0]*16 
        gender_count = [0]*2       

        image_dir = Path(dataDir)
        img_list = image_dir.glob("*")
        img_list = list(img_list)
        train_list = random.sample(img_list, int(0.8*len(img_list)))
        val_list = list(set(img_list) - set(train_list))
        if self.stage == "train":
            data_imgs = train_list
        else:
            data_imgs = val_list

        for image_path in data_imgs:
            image_name = image_path.name 
            #age =image_name.split("A")[1].split(".")[0].split("G")[0]
            #gender =image_name.split("A")[1].split(".")[0].split("G")[1]

            # update load label for mega_age_gender dataset
            age = image_name.strip().split("_")[1].split("A")[1]
            gender = image_name.strip().split("_")[2][1]

            age = self.age_to_cls(abs(int(age)))
            age_cls = self.age_to_level(age)
            gender_cls = int(gender)
            age_count[age] += 1
            gender_count[gender_cls] += 1
            labels = (age_cls, gender_cls)
            if image_path.is_file():
                self.image_path_and_type.append([str(image_path), labels])
        logger.info("Dataset {} stage infomation".format(self.stage.upper()))
        logger.info("Number images: {}".format(len(self.image_path_and_type)))
        logger.info("Class age {}".format(age_count))
        logger.info("Class gender {}".format(gender_count))



if __name__ == "__main__":
    dataset = DatasetLoader("./dataset/last_face_age_gender", "train")
    #dataset = DatasetLoader("dataset/small_data", "val")
    def convert_gender_to_string(gender):
        if gender:
            return "Male"
        return "Female"

    plt.figure(figsize=(30,30))
    for i in range(9):
        image, label = dataset[i + 100]
        age, gender = label
        gender = convert_gender_to_string(gender)
        plt.subplot(3, 3, i + 1)
        #age = torch.sum(age).item()
        image = image.reshape(224, 224, 3)
        plt.imshow(image.numpy())
        plt.title(f"age: {age}\ngender: {gender}")
        plt.axis("off")
    plt.show()


        
    