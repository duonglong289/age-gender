import torch
import torch.nn as nn


# def compute_mae_mse(predicted_age, target_age):
#     ''' Compute mean absolute error, mean square error
#     '''
#     mae, mse = 0., 0. 
#     predicted_age = predicted_age.view(-1, 1)
#     target_age = target_age.view(-1, 1)# .type(torch.FloatTensor).item()
#     mae = torch.sum(torch.abs(predicted_age - target_age))/len(predicted_age)
#     # mse = torch.sum((predicted_age - target_age)**2)
#     # return mae, mse           
#     return mae

def compute_mae_mse(prob_age, target_age):
    prob_levels = torch.argmax(prob_age, dim=2)
    predicted_labels = torch.sum(prob_levels, dim=1)
    target_age = torch.sum(target_age, dim=1)
    mae = torch.sum(torch.abs(predicted_labels - target_age))
    mae = mae.float()/len(target_age)
    return mae