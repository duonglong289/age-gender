import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np


def cost_ordinary(predicted, groundtruth):
    ''' Compute orinary regression loss
    Params
    ------
        predicted: age score
    '''
    logits = predicted
    levels = groundtruth
    cost = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels)),
           dim=1))
    return torch.mean(cost)


def cost_ce(predicted, groundtruth):
    '''Compute cross entropy loss
    Params
    -----
        predicted: age probability
    '''
    cost = F.cross_entropy(predicted, groundtruth)
    return cost


def cost_nll(predicted, groundtruth):
    '''Compute negative log-likelihood loss
    Params
    ------
        predicted: age probability
    '''
    loss = nn.NLLLoss()
    if isinstance(groundtruth.dtype, torch.FloatTensor):
        import ipdb; ipdb.set_trace()
    cost = loss(predicted, groundtruth)
    return cost

    