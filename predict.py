import torch
import numpy as np 
import os 
import torch.nn as nn 
import torchvision.transforms as transforms

from models.net import ModelAgeGender

def main():
    model = ModelAgeGender()
    model.init_model()

    print(model)

if __name__ == "__main__":
    main()