import torch
import models.cost_fn as cost_fn
import torch.nn.functional as F
predicted = torch.rand(32, 2, requires_grad=True)
groundtruth = torch.randint(2, (32,), dtype=torch.int64)

probs = torch.rand(32, 100)
print(torch.sum((probs > 0.5).type(torch.int), dim=1))
