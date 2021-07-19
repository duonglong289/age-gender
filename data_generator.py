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
    def __init__(self, dataDir, stage, batch_size=1, image_size=224, 
                    augument=aug
                ):
        self.num_age_classes = 81
        self.batch_size = batch_size
        self.image_size = image_size
        self.stage = stage
        self.image_path_and_type = []
        self._load_dataset(dataDir)
        self.transform_data =  self.build_transforms()
        self.aug = augument
        self.image_num = len(self.image_path_and_type)
        self.indices = np.random.permutation(self.image_num)
        
        

    def __len__(self):
        return self.image_num // self.batch_size


    def __getitem__(self, idx):
        batch_size = self.batch_size

        image_path, label = self.image_path_and_type[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        age, gender = label

        if self.aug:
            img = self.aug(image=img)
        else:
            img = img
        
        #img = aug.augument(image=img)
        #print(type(img))
                
        image = Image.fromarray(img)
        #img = torch.from_numpy(img)
        X = self.transform_data(image)
        #print(type(X))
        #X = X.to("cuda")
        y = (age, gender)
        return os.path.basename(image_path), X, y
    

    def build_transforms(self, PIXEL_MEAN = [0.5, 0.5, 0.5], PIXEL_STD =[0.5, 0.5, 0.5]):
        normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        if self.stage=="train":
            transform = T.Compose([
                    #aug.augment_image,
                    T.Resize([self.image_size,self.image_size], Image.BICUBIC),
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
        if age < 80:
            return age  
        else:
            return 80


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
        age_count = [0]*self.num_age_classes
        gender_count = [0]*2
        for image_path in data_imgs:
            image_name = image_path.name 
            #age =image_name.split("A")[1].split(".")[0].split("G")[0]
            #gender =image_name.split("A")[1].split(".")[0].split("G")[1]

            # update load label for mega_age_gender dataset
            age = image_name.strip().split("_")[1].split("A")[1]
            gender = image_name.strip().split("_")[2][1]

            age_cls = self.age_to_cls(int(age))
            
            gender_cls = int(gender)
            age_count[age_cls] += 1
            gender_count[gender_cls] += 1
            labels = (age_cls, gender_cls)
            if image_path.is_file():
                self.image_path_and_type.append([str(image_path), labels])
        logger.info("Dataset {} stage infomation".format(self.stage.upper()))
        logger.info("Number images: {}".format(len(self.image_path_and_type)))
        logger.info("Class age {}".format(age_count))
        logger.info("Class gender {}".format(gender_count))



if __name__ == "__main__":
    dataset = DatasetLoader("dataset/last_face_age_gender", "train", augument=None)
    dataset_aug = DatasetLoader("dataset/last_face_age_gender", "train")
    #dataset = DatasetLoader("dataset/last_face_age_gender", "val")
    path, img, label = dataset[3]
    path, img_aug, label_aug = dataset_aug[3]

    age, gender = label  
    age_aug, label_aug = label_aug 
    img = img.numpy().transpose((2, 1, 0))
    img_aug = img_aug.numpy().transpose((2, 1, 0))
    
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(img_aug)

    plt.show()


    
    
