import torch
import numpy as np

import torch.nn as nn 
import torch.nn.functional as F

loss = nn.NLLLoss()

predict = torch.randn(1, 3)
# label = torch.ones(2).type(torch.LongTensor)
label = torch.LongTensor([1])
out = loss(predict, label)

print(out)