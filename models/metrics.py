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
if __name__ == '__main__':
    num_classes = 15    
    predicted_age = torch.rand(15)
    target_age = torch.Tensor([1]*10 + [0]*5)
    mae = compute_mae_mse(predicted_age, target_age)
    print(mae)