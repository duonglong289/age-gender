import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np


def cost_ordinary(predicted, groundtruth):
    ''' Compute orinary regression loss
    '''
    logits = predicted
    levels = groundtruth
    cost = (-torch.sum(F.log_softmax(logits, dim=2)[:, :, 1]*levels 
                    + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels), dim=1))
    return cost


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

    