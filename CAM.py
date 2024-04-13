import torch
import os

import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from PIL import Image
from network.Unet import Unet

torch.backends.cudnn.enabled = False


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        # 注册钩子，获取目标层的梯度和特征图
        def hook_fn(module, grad_input, grad_output):
            self.gradients = grad_input[0]

        self.target_layer.register_backward_hook(hook_fn)

    def forward(self, inputs):
        return self.model(inputs)

    def get_gradcam(self, inputs, class_idx=None):
        outputs = self.forward(inputs)

        # 如果没有指定类别，则选择得分最高的类别
        if class_idx is None:
            print(outputs.shape)
            class_idx = torch.argmax(outputs, 1).item()

        # 清除梯度
        self.model.zero_grad()

        # 反向传播指定类别的得分
        outputs[0, class_idx].backward(retain_graph=True)

        # 获取目标层的特征图
        target_activations = self.gradients.detach()

        # 对特征图进行全局平均池化，得到权重
        weights = torch.mean(target_activations, dim=(2, 3), keepdim=True)

        # 使用权重对特征图进行加权组合
        cam = torch.sum(torch.mul(target_activations, weights),
                        dim=1,
                        keepdim=True)
        cam = F.relu(cam)

        # 上采样
        cam = F.interpolate(cam,
                            size=(1024, 1024),
                            mode='bilinear',
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam


# 加载模型
model = torch.load('log/2024-04-13-12-28-19/Epoch_562_Classifer.pth').cuda(device=1)

image_path = 'data-road/clean/image/1.jpg'
image = Image.open(image_path)

preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

input_tensor = preprocess(image).unsqueeze(0).cuda(device=1)  # [1, 3, 224, 224]
# print(input_tensor.shape)

# 选择比较靠近输出的卷积层
gradcam = GradCAM(model, model.conv3)

cam = gradcam.get_gradcam(input_tensor)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.imshow(cam, alpha=0.5, cmap='jet')
plt.title("Grad-CAM")
plt.show()
