import torch
import cv2
import models.cost_fn as cost_fn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random 
import glob 
from data_generator import DatasetLoader
import numpy as np
from clearml import Task, Logger

ls = torch.Tensor([])
ls1 = ls.append(4)

print(ls1)   

