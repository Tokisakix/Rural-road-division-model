import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

backbone   = torch.load('log/2024-05-21-21-54-21/Epoch_100_Seg.pth').to('cuda:0')
classifier = torch.load('log/2024-05-21-21-54-21/Epoch_100_Classifier.pth').to('cuda:0')

if isinstance(backbone,torch.nn.DataParallel):
        backbone = backbone.module

if isinstance(classifier,torch.nn.DataParallel):
        classifier = classifier.module

backbone.eval()
classifier.eval()
print(classifier)

#----------resnet152-------
final_conv = classifier.layer4
fc_weights = classifier.state_dict()['fc.0.weight'].cpu().numpy()

#----------cnn-------------
#final_conv = classifier.conv3
#fc_weights = classifier.state_dict()['linear.7.weight'].cpu().numpy()

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

# 注册hook来捕获特征
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# 连接到最后一个卷积层
#----------resnet152-------
classifier.layer4.register_forward_hook(hook_feature)
#----------cnn-------------
# classifier.conv3.register_forward_hook(hook_feature)

img_path = 'data-road/clean_crop/image/5.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))  # 256x256
img_tensor = torch.from_numpy(img).float().div(255.0).permute(2,0,1).unsqueeze(0).to('cuda:0')

# 前向传播获取预测结果
feature,logits = backbone(img_tensor)
output  = classifier(feature)

predicted_class_idx = [output.argmax(dim=1).item()]  # 获取最大概率的类别索引

# 从hook中获取特征
final_conv_features = features_blobs[0]

# 生成CAM
CAMs = getCAM(final_conv_features, fc_weights, predicted_class_idx)

# 显示CAM和原始图像
# img = cv2.imread(img_path)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result  = (heatmap * 0.3 +  img * 0.7).astype(np.float32)
cv2.imwrite('cam_result.jpg', result)