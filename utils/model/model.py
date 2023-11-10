import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    inputs  -> FloatTensor[batch_size, in_channel, height, weight]
    outputs -> FloatTensor[batch_size, out_channel, height, weight]
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        return
    
    def forward(self, x):
        out = self.conv(x)
        out = F.sigmoid(out)
        return out