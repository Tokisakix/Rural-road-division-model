import torch

from .model import Model

def load_model(path):
    model = torch.load(path)
    return model

def save_model(model, path):
    torch.save(model, path)
    return