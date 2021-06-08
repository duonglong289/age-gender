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

class CoralCost:
    def __init__(self, num_classes=None, imp_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.imp_weights = imp_weights

    def cost_coral(self, predicted, groundtruth):
        if self.num_classes:
            if not self.imp_weights:
                imp = torch.ones(self.num_classes)
            else:
                imp = torch.Tensor([self.imp_weights]*self.num_classes)
        else:
            if not self.imp_weights:
                imp = 1;
            else: 
                imp = self.imp_weights
        val = (-torch.sum((F.log_sigmoid(predicted)*groundtruth
                        + (F.log_sigmoid(predicted) - groundtruth)*(1-groundtruth))*imp, dim=1))
        return torch.mean(val)            
