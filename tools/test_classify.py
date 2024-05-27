import cv2
import cv2 as cv
import numpy as np
import torch
import os

import torchvision
from torch import nn

backbone   = torch.load('log/2024-05-22-12-54-07/Epoch_100_Seg.pth').to('cuda:0')
classifier = torch.load('log/2024-05-22-12-54-07/Epoch_100_Classifier.pth').to('cuda:0')

source_root="data-road/clean_cropp"

def check_road(labels, unusual_percent):
    res = []
    for label in labels:
        flag = (torch.sum(label) > 0) #and (torch.sum(label) < 255 * 255  * 1 * unusual_percent)
        res.append(int(flag))
    res = torch.LongTensor(res)
    return res

# 判断是否多卡训练
if isinstance(backbone,torch.nn.DataParallel):
        backbone = backbone.module

if isinstance(classifier,torch.nn.DataParallel):
        classifier = classifier.module

backbone.eval()
classifier.eval()

val = os.listdir(r"data-road/clean_crop/image")

road_y = 0
road_n = 0
flag_y = 0
flag_n = 0
for i in range(1,201):
    img = cv2.imread(f"data-road/clean_crop/image/{i}.jpg")
    img = cv2.resize(img, (256, 256))
    label = cv2.imread(f"data-road/clean_crop/label/{i}.jpg", cv.IMREAD_GRAYSCALE)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    label=transforms(label)
    # print("img:",i)
    img = cv2.resize(img, (256, 256))  # 256x256
    img_tensor = torch.from_numpy(img).float().div(255.0).permute(2,0,1).unsqueeze(0).to('cuda:0')

    # 前向传播获取预测结果(分类结果)
    #print(backbone)
    road_yn=check_road(label,0.2).item()
    feature, logits = backbone(img_tensor)
    output = classifier(feature)
    print(output.shape)

    predicted_class_idx = [output.argmax(dim=1).item()]  # 获取最大概率的类别索引


    if(road_yn==1):
        road_y += 1
    if (road_yn == 0):
        road_n += 1
    if(road_yn==predicted_class_idx[0] and road_yn==1) :
        flag_y += 1
    if (road_yn == predicted_class_idx[0] and road_yn == 0):
        flag_n += 1
    if(road_yn !=predicted_class_idx[0]):
        print("img:",i)
        print("yes or no", road_yn, output, predicted_class_idx)

    print("#")

print(f"{road_y+road_n}images")
print(f"{road_y}images have road")
print(f"{road_n}images don't have road")
print("road_yes",flag_y/road_y)
print("road_no",flag_n/road_n)
print("all",(flag_y+flag_n)/(road_y+road_n))
