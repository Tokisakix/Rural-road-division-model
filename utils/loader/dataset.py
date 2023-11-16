import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import os

from utils import load_config
from .transfer import random_hue_saturation_value, random_shift_scale_rotate, random_horizontal_flip, random_ver_flip, \
    random_rotate90

CONFIG = load_config()
DATA_CONFIG = CONFIG["data"]


# 读取单张图片
def basic_loader(img_path, mask_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    img = random_hue_saturation_value(img, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
                                      val_shift_limit=(-15, 15))
    img, mask = random_shift_scale_rotate(img, mask, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1),
                                          aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))

    img, mask = random_horizontal_flip(img, mask)
    img, mask = random_ver_flip(img, mask)
    img, mask = random_rotate90(img, mask)

    mask = mask.reshape(*mask.shape, -1)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask


# 获取对应文件夹下的所有图片序号
def get_folder_img_ids(root, image_, mask_):
    ids = filter(lambda x: x.find(image_) != -1, os.listdir(root))
    ids = map(lambda x: x[:-len(image_)], ids)
    ids = list(ids)
    return ids


# 文件夹数据集
class ImageFolder(Dataset):
    def __init__(self, root, image_, mask_):
        super(ImageFolder, self).__init__()
        self.ids = get_folder_img_ids(root, image_, mask_)
        self.loader = basic_loader
        self.root = root
        self.image_ = image_
        self.mask_ = mask_
        return

    def __getitem__(self, index):
        idx = self.ids[index]
        img_path = f"{self.root}\{idx}{self.image_}"
        mask_path = f"{self.root}\{idx}{self.mask_}"
        img, mask = self.loader(img_path, mask_path)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)


# 最终的加载器
class FinalDataSet(Dataset):
    def __init__(self, basic_set, train):
        super(FinalDataSet, self).__init__()
        self.basic_set = basic_set
        self.train = train
        self.images = torch.rand(32, 3, 1024, 1024)
        self.pos = torch.rand(32, 1, 1024, 1024)
        self.neg = torch.rand(32, 1, 1024, 1024) if train else None
        return

    def __getitem__(self, index):
        image = self.images[index]
        pos = self.pos[index]
        if not self.train:
            return image, pos
        neg = self.neg[index]
        return image, pos, neg

    def __len__(self):
        return self.images.shape[0]


# 获取初级数据集
def get_basic_dataset(data_config):
    train_set = []
    test_set = []

    for dataset_info in data_config:
        use = dataset_info["use"]
        if not use:
            continue
        train_root = dataset_info["train_root"]
        test_root = dataset_info["test_root"]
        image_ = dataset_info["image_"]
        mask_ = dataset_info["mask_"]
        train_folder_dataset = ImageFolder(train_root, image_, mask_)
        test_folder_dataset = ImageFolder(test_root, image_, mask_)
        train_set.append(train_folder_dataset)
        test_set.append(test_folder_dataset)

    basic_train_set = train_set[0]
    for idx in range(1, len(train_set)):
        basic_train_set = basic_train_set + train_set[idx]
    basic_test_set = test_set[0]
    for idx in range(1, len(test_set)):
        basic_test_set = basic_test_set + test_set[idx]

    return basic_train_set, basic_test_set


# 获取最终的数据集
def get_final_set():
    basic_train_set, basic_test_set = get_basic_dataset(DATA_CONFIG)
    final_train_set = FinalDataSet(basic_train_set, train=True)
    final_test_set = FinalDataSet(basic_test_set, train=False)
    return final_train_set, final_test_set
