import os
import torch
import torchvision
import cv2 as cv

from torch import nn
from tqdm import tqdm
from load_config import load_config
from load_config import load_config
from network.DinkNet import DinkNet34, Dblock, DecoderBlock
from torch.utils.data import Dataset

CONFIG        = load_config()

class Model(nn.Module):
    def __init__(self):
        self.module = DinkNet34()
        return
    
    def forward(self, x):
        out = self.module(x)
        return out


class DataSet(Dataset):
    def __init__(self, root, clean, CUDA, size):
        super().__init__()
        self.dataset = []
        self.clean = clean
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        # self.dinknet = torch.load("model/DinkNet34.pth")
        self.dinknet = DinkNet34()
        self.dinknet.load_state_dict(torch.load("model/DinkNet34.th"), strict=False)
        self.dinknet = self.dinknet.cuda() if CUDA else self.dinknet
        for idx in tqdm(range(size)):
            img_path   = f"{root}/{'clean' if clean else 'raw'}/image/{idx + 1}.jpg"
            label_path = f"{root}/{'clean' if clean else 'raw'}/label/{idx + 1}.png"
            image = cv.imread(img_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            label = cv.imread(label_path, cv.IMREAD_GRAYSCALE) if os.path.isfile(label_path) else None
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
        size=DATA_CONFIG["size"]
    )

    return dataset


# ---Test---
    
if __name__ == "__main__":
    CONFIG = load_config()
    
    train_dataset = get_dataset(CONFIG, True)
    test_dataset  = get_dataset(CONFIG, False)

    print(train_dataset, len(train_dataset))
    print(test_dataset,  len(test_dataset))

    # train_dataset.save()
    # test_dataset.save()
