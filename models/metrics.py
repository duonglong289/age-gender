import torch
import torch.nn as nn


# def compute_mae_mse(predicted_age, target_age):
#     ''' Compute mean absolute error, mean square error
#     '''
#     mae, mse = 0., 0. 
#     predicted_age = predicted_age.view(-1, 1)
#     target_age = target_age.view(-1, 1)# .type(torch.FloatTensor).item()
#     import ipdb; ipdb.set_trace()
#     mae = torch.sum(torch.abs(predicted_age - target_age))/len(predicted_age)
#     # mse = torch.sum((predicted_age - target_age)**2)
#     # return mae, mse           
#     return mae

def compute_mae_mse(score_age, prob_age, target_age):
    logits, probas = score_age, prob_age
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    target_age = torch.sum(target_age, dim=1)
    num_examples = target_age.size(0)
    mae = torch.sum(torch.abs(predicted_labels - target_age))
    # mse += torch.sum((predicted_labels - targets)**2)
    # mae = mae.float() / num_examples
    # mse = mse.float() / num_examples
    # return mae, mse
    return mae.float() / num_examples