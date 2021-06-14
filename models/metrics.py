import torch
import torch.nn as nn


def compute_mae_mse(predicted_age, target_age):
    ''' Compute mean absolute error, mean square error
    '''
    predicted_age = predicted_age.view(-1, 1)
    target_age = target_age.view(-1, 1)# .type(torch.FloatTensor).item()
    mae = torch.sum(torch.abs(predicted_age - target_age))/len(predicted_age)
    # mse = torch.sum((predicted_age - target_age)**2)
    # return mae, mse           
    return mae
