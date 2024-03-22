import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
import cv2 as cv
from tqdm import tqdm

from load_config import load_config
from model import DinkNet34, Dblock, DeConvBn, DecoderBlock

class Model(nn.Module):
    def __init__(self):
        self.module = DinkNet34()
        return
    
    def forward(self, x):
        out = self.module(x)
        return out


class DataSet(Dataset):
    def __init__(self, root, clean, CUDA):
        super().__init__()
        self.dataset = []
        self.clean = clean
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self.dinknet = torch.load("model/DinkNet34.pth")
        self.dinknet = self.dinknet.cuda() if CUDA else self.dinknet
        for idx in tqdm(range(8)):
            img_path   = f"{root}/{'clean' if clean else 'raw'}/image/{idx}.png"
            label_path = f"{root}/{'clean' if clean else 'raw'}/label/{idx}.png"
            image = cv.imread(img_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            label = cv.imread(label_path, cv.IMREAD_GRAYSCALE) if clean else None
            image = self.transforms(image)
            label = self.transforms(label) if clean else self.dinknet(image.unsqueeze(0).cuda() if CUDA else image.unsqueeze(0)).squeeze(0).cpu()
            self.dataset.append((image, label, torch.tensor([1.0] if clean else [0.0]), label_path))
        del self.dinknet
        return
    
    def update(self, index, image, label, clean, label_path):
        self.dataset[index] = (image, label, clean, label_path)
        return
    
    def save(self):
        if self.clean:
            return
        for (_, label, _, path) in self.dataset:
            label = torchvision.transforms.ToPILImage()(label)
            label.save(path)
        return
    
    def __getitem__(self, index):
        image, label, clean, label_path = self.dataset[index]
        return index, image, label, clean, label_path
    
    def __len__(self):
        length = len(self.dataset)
        return length

def get_dataset(CONFIG, clean):
    DATA_CONFIG = CONFIG["data"]

    dataset = DataSet(
        root=DATA_CONFIG["root"],
        clean=clean,
        CUDA=DATA_CONFIG["cuda"],
    )

    return dataset


# ---Test---
    
if __name__ == "__main__":
    CONFIG = load_config()
    
    train_dataset = get_dataset(CONFIG, True)
    test_dataset  = get_dataset(CONFIG, False)

    print(train_dataset, len(train_dataset))
    print(test_dataset,  len(test_dataset))

    train_dataset.save()
    test_dataset.save()