
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
    state_dict_path = "./logs/2_0.5528364550499689_gender_197.11904907226562_age.pt"
    # model.model.load_state_dict(torch.load(state_dict_path), strict=False)
    # model.model.eval().to(model.device)
    model.load_statedict(state_dict_path="./logs/2_0.5528364550499689_gender_197.11904907226562_age.pt")
    # print(model)
    return model

def pre_process_img(img):
    normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = T.Compose([
                T.ToPILImage(),
                T.Resize([224,224]),
                T.ToTensor(),
                normalize_transform])
    img_transform = transform(image)
    return img_transform

if __name__ == "__main__":
    import argparse
    import os
    import random
    import glob
    parser = argparse.ArgumentParser(description="Training Liveness Detection")
    parser.add_argument("--image", type=str, required=False, help="Path to image")
    age_range = ["0-5", "5-10", "10-14", "14-18", "18-21", "21-25", "25-29",
                 "25-29", "29-34", "34-38", "38-42", "42-46",
                 "46-50", "50-55", "55-60" ,"60-65", ">= 65"]

    args = parser.parse_args()
    if not args.image:
        lst_image = random.sample(glob.glob(os.path.join("./all_faces", "*")),10)
    else: 
        lst_image = [args.image]

    data= []
    for image_path in lst_image:
        image = cv2.imread(image_path)
        image = pre_process_img(image)
        model = main()
        _, age, gender = model.predict_image(image.unsqueeze(0))
        gender = torch.argmax(torch.exp(gender)).detach().numpy().item()
        age = age_range[torch.argmax(torch.exp(age)).detach().numpy().item()]
        # print("{}   {}   {}   {}".format(args.image.split("A")[1].split(".")[0].split("G")[0], \
        #                                 args.image.split("A")[1].split(".")[0].split("G")[1], age, gender))
        data.append([image_name.split("_")[1].split("A")[1], \
                    age,
                    image_name.split("_")[2][1], \
                    gender])
    import pandas as pd
    df = pd.DataFrame(data, columns=['true_age', 'true_gender', 'pred_age', 'pred_gender'])
    df.index = pd.to_numeric(df.index, errors='coerce')
    print(df)