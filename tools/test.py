import cv2
import cv2 as cv
import numpy as np
import torch
import os

import torchvision

backbone   = torch.load('log/2024-05-22-12-54-07/Epoch_100_Seg.pth').to('cuda:0')
classifier = torch.load('log/2024-05-22-12-54-07/Epoch_100_Classifier.pth').to('cuda:0')

source_root="data-road/clean_crop"

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

def getCAM(feature_conv, weight_softmax, class_idx):
    b, c, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
    return output_cam
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

val = os.listdir(r"D:\大创\rural-roadv4\data-road\clean_crop\image")

road_y = 0
road_n = 0
flag_y = 0
flag_n = 0

for i in range(1,201):

    final_conv = classifier.layer4
    # final_conv = classifier.conv3
    fc_weights = classifier.state_dict()['fc.0.weight'].cpu().numpy()*classifier.state_dict()['fc.0.weight'].cpu().numpy()
    # fc_weights = classifier.state_dict()['linear.7.weight'].cpu().numpy()

    # 注册hook来捕获特征
    features_blobs = []

    # 连接到最后一个卷积层
    classifier.layer4.register_forward_hook(hook_feature)
    # classifier.conv3.register_forward_hook(hook_feature)

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

    predicted_class_idx = [output.argmax(dim=1).item()]  # 获取最大概率的类别索引

    # 从hook中获取特征
    final_conv_features = features_blobs[0]

    # 生成CAM
    CAMs = getCAM(final_conv_features, fc_weights, predicted_class_idx)

    # 显示CAM和原始图像
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = (heatmap * 0.4 + img * 0.6).astype(np.float32)
    cv.imwrite(f"data-road/clean_crop/label/"+ f'{i}cam.png', result)
    print("#")

