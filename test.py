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

task = Task.init(project_name="examples", task_name="2D plots reporting")

logger = task.get_logger()
confusion = np.random.randint(10, size=(10, 10))
iteration = 0
logger.report_matrix(
    "example_confusion",
    "gender",
    iteration=iteration,
    matrix=confusion,
    xaxis="title X",
    yaxis="title Y",
)

# report confusion matrix with 0,0 is at the top left
logger.report_matrix(
    "example_confusion_2",
    "age",
    iteration=iteration,
    matrix=confusion,
    xaxis="title X",
    yaxis="title Y",
    yaxis_reversed=True,
)

