import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import os

# 饱和度变换
def random_hue_saturation_value(image, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv.add(v, val_shift)
        image = cv.merge((h, s, v))
        image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
    return image

# 透视变换
def random_shift_scale_rotate(image, mask, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0), border_mode=cv.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv.getPerspectiveTransform(box0, box1)
        image = cv.warpPerspective(image, mat, (width, height), flags=cv.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        mask = cv.warpPerspective(mask, mat, (width, height), flags=cv.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
    return image, mask

# 水平翻转
def random_horizontal_flip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv.flip(image, 1)
        mask = cv.flip(mask, 1)
    return image, mask

# 垂直翻转
def random_ver_flip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv.flip(image, 0)
        mask = cv.flip(mask, 0)
    return image, mask

# 90°翻转
def random_rotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)
    return image, mask

# 读取单张图片
def default_loader(img_path, mask_path):
    img = cv.imread(img_path)
    mask = cv.imread(mask_path)
    img = random_hue_saturation_value(img, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
    img, mask = random_shift_scale_rotate(img, mask, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))

    img, mask = random_horizontal_flip(img, mask)
    img, mask = random_ver_flip(img, mask)
    img, mask = random_rotate90(img, mask)

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

# DataSet
class ImageFolder(Dataset):
    def __init__(self, root, image_, mask_):
        super(ImageFolder, self).__init__()
        self.ids = get_folder_img_ids(root, image_, mask_)
        self.loader = default_loader
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

# 获取初级数据集
def get_basic_dataset(data_config):
    train_set = []
    test_set = []

    for dataset_info in data_config:
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