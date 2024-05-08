import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
from network.Unet import Classifier

class CAM:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model  = torch.load(model_path).to(self.device)
        self.model.eval()
        # 用最后一个线性层作权重参数
        self.fc_weights     = self.model.state_dict()['linear.7.weight'].cpu().numpy()
        self.features_blobs = []
        # 在分类器的最后一个卷积层挂钩子
        self.model.conv3.register_forward_hook(self.hook_feature)

    def preprocess(self, img_path):
        # 输入[H,W] =[256,256]
        img        = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img        = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        return img, img_tensor

    def hook_feature(self, module, input, output):
        self.features_blobs.append(output.data.cpu().numpy())

    def getCAM(self, feature_conv, weight_softmax, class_idx):
        b, c, h, w  = feature_conv.shape
        output_cam  = []
        for idx in class_idx:
            cam     = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))
            cam     = cam.reshape(h, w)
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cam_img)
        return output_cam

    def generate(self, img_path):
        img, img_tensor     = self.preprocess(img_path)

        # 前向传播，获取预测结果
        output              = self.model(img_tensor)
        predicted_class_idx = [output.argmax(dim=1).item()]

        # 从钩子处获取特征
        final_conv_features = self.features_blobs[-1]  # 使用最新捕获的特征
        self.features_blobs = []  # 清空列表以防止内存溢出

        # 生成CAM
        CAMs = self.getCAM(final_conv_features, self.fc_weights, predicted_class_idx)

        # 保存结果图像
        img_color        = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        height, width, _ = img_color.shape
        heatmap          = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result           = (heatmap * 0.3 + img_color * 0.5).astype(np.float32)
        result_path      = img_path.replace('.jpg', '_result.jpg')
        cv2.imwrite(result_path, result)
        return result_path

cam_generator     = CAM('/public/zjj/public/zjj/xzx/log/2024-05-08-12-46-56/Epoch_39_Classifier.pth')
result_image_path = cam_generator.generate('/public/zjj/public/zjj/xzx/data-road-clipped/clean/image/1000.jpg')
print("CAM image saved to:", result_image_path)
