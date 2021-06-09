import torch
import models.cost_fn as cost_fn
import torch.nn.functional as F
predicted = torch.rand(32, 2, requires_grad=True)
groundtruth = torch.randint(2, (32,), dtype=torch.int64)
loss = torch.nn.functional.cross_entropy(predicted, groundtruth)
imp = 0.0001
new_gt = torch.zeros(32,2)
for i in range(new_gt.shape[0]):
    new_gt[i][groundtruth[i]] = 1
print(f"{predicted} \n{groundtruth}\n{new_gt}")

val = -torch.sum((F.logsigmoid(predicted)*new_gt
                        + (F.logsigmoid(predicted)-new_gt)*(1-new_gt))*imp, dim=1)
print(torch.mean(val))
