import torch
from utils.criterion import DicBceLoss

output = torch.rand(32, 1024, 1024)
label  = torch.rand(32, 1024, 1024)

criterion = DicBceLoss()

loss = criterion(output, label)

print(loss)