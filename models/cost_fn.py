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
        imp = self.imp_weights
        if self.imp_weights is None:
            imp = 1
        #predicted = torch.max(predicted, dim=1).values
        new_gt = torch.zeros(predicted.shape, device='cuda')
        for i in range(predicted.shape[0]):
            new_gt[i][groundtruth[i]] = 1
        val = (-torch.sum((F.logsigmoid(predicted)*new_gt
                        + (F.logsigmoid(predicted) - new_gt)*(1-new_gt))*imp, dim=1))
        return torch.mean(val)            
