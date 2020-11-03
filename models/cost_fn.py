import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

NUM_AGE_CLASSES=100

def age_to_level(age, num_age_classes):
    ''' Convert age to levels, for ordinary regression task
    '''
    levels = []
    for ind in age:
        level = torch.Tensor([1]*ind + [0]*int(num_age_classes - 1 - ind)).to(torch.device("cuda"))
        levels.append(level)
    return torch.stack(levels)

def cost_ordinary(predicted, groundtruth):
    ''' Compute orinary regression loss
    '''
    logits = predicted
    # levels = age_to_level(groundtruth, NUM_AGE_CLASSES)
    levels = groundtruth
    cost = (-torch.sum(F.log_softmax(logits, dim=2)[:, :, 1]*levels 
                    + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels), dim=1))
    return torch.mean(cost)


def cost_ce(predicted, groundtruth):
    '''Compute cross entropy loss
    '''
    cost = F.cross_entropy(predicted, groundtruth)
    return cost


def cost_nll(predicted, groundtruth):
    '''Compute negative log-likelihood loss
    '''
    loss = nn.NLLLoss()
    cost = loss(predicted, groundtruth)
    return cost

    