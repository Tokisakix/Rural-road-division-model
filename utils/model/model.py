import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT(nn.Module):
    """
    inputs  -> FloatTensor[batch_size, in_channel, height, weight]
    outputs -> FloatTensor[batch_size, out_channel, height, weight]
    """
    def __init__(self):
        super(ViT, self).__init__()
        return
    
    def forward(self, x):
        out = x
        out = torch.rand(x.shape[0], 1, 1024, 1024)
        return out