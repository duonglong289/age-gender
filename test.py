import torch
import cv2
import models.cost_fn as cost_fn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random 
import glob 
from data_generator import DatasetLoader

batch = torch.randint(high=16, size=(64,16))
print(batch.shape)
print(torch.sum(batch, dim=1).shape)

def convert_gender_to_string(gender):
    if gender:
        return "Male"
    return "Female"

train_loader = DatasetLoader('./dataset/last_face_age_gender', "train")

image_path, label = train_loader.image_path_and_type[70]
age, gender = label
gender = convert_gender_to_string(gender)

plt.subplot(121)
img, labels = train_loader[70]
img = img.reshape(224, 224, 3)

plt.imshow(img.numpy())
plt.title("Image from DataserLoader")
plt.xlabel(f"age: {age}\ngender: {gender}")

plt.subplot(122)
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Raw image")
plt.xlabel(f"age: {age}\n gender: {gender}")


plt.show()

