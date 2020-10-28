import torch
import numpy as np 
import os 
import torch.nn as nn 
import torchvision.transforms as transforms
import cv2
from models.net import ModelAgeGender
import torchvision.transforms as T

def main():
    model = ModelAgeGender()
    model.init_model()
    model.device = torch.device("cpu")
    # state_dict = torch.load(torch.load(state_dict_path), map_location=self.device)
    # import ipdb; ipdb.set_trace()
    state_dict_path = "./logs/3_0.54209313364256_gender_0.0_age.pt"
    model.model.load_state_dict(torch.load(state_dict_path), strict=False)
    model.model.eval().to(model.device)
    # model.load_statedict(state_dict_path="./logs/3_0.54209313364256_gender_0.0_age.pt")
    # print(model)
    return model

def pre_process_img(img):
    normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = T.Compose([
                T.ToPILImage(),
                # T.Resize([self.image_size,self.image_size]),
                T.Resize([224,224]),
                T.ToTensor(),
                normalize_transform])
    img_transform = transform(image)
    return img_transform

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Liveness Detection")
    parser.add_argument("--image", type=str, default = "../all_faces/00000A02G0.png", required=False, help="Path to image")

    args = parser.parse_args()
    image = cv2.imread(args.image)
    image = pre_process_img(image)
    # import ipdb; ipdb.set_trace()
    model = main()
    # import ipdb; ipdb.set_trace()
    output = model.predict_image(image.unsqueeze(0))
    import ipdb; ipdb.set_trace()
    # print(model.predict_image(torch.from_numpy(image)))
    
