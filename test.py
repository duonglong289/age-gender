import torch  
import numpy as np 

age_score = torch.rand(16, 80, 2)
age = torch.argmax(age_score, dim=2)
print(age.shape)