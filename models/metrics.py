import torch
import torch.nn as nn


def compute_mae_mse(pd_age, gt_age):
    mae = torch.sum(torch.abs(pd_age - gt_age))

    mae = mae.float()/len(gt_age)
    return mae