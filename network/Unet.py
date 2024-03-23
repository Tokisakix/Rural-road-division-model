import torch
import torch.nn as nn
import torch.nn.functional as F

from load_config import load_config

class ConvBn(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel), nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel), nn.ReLU(),
        )
        return
    
class DeConvBn(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=2, stride=2):
        super().__init__(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, bias=True),
            nn.BatchNorm2d(out_channel), nn.ReLU(),
        )
        return
    
class Unet(nn.Module):
    def __init__(self, in_channel=3, classes_num=1):
        super().__init__()
        self.conv1 = ConvBn(in_channel, 4)
        self.conv2 = ConvBn(4, 8)
        self.conv3 = ConvBn(8, 16)
        self.conv4 = ConvBn(16, 32)
        self.conv5 = ConvBn(32, 64)
        self.conv6 = ConvBn(64, 32)
        self.conv7 = ConvBn(32, 16)
        self.conv8 = ConvBn(16, 8)
        self.conv9 = ConvBn(8, 4)
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(classes_num), nn.Sigmoid(),
        )
        
        self.dconv1 = DeConvBn(64, 32)
        self.dconv2 = DeConvBn(32, 16)
        self.dconv3 = DeConvBn(16, 8)
        self.dconv4 = DeConvBn(8, 4)
        return
    
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(F.max_pool2d(c1, 2))
        c3 = self.conv3(F.max_pool2d(c2, 2))
        c4 = self.conv4(F.max_pool2d(c3, 2))
        out = self.conv5(F.max_pool2d(c4, 2))
        
        c4 = torch.cat([c4, self.dconv1(out)], dim=1)
        out = self.conv6(c4)
        c3 = torch.cat([c3, self.dconv2(out)], dim=1)
        out = self.conv7(c3)
        c2 = torch.cat([c2, self.dconv3(out)], dim=1)
        out = self.conv8(c2)
        c1 = torch.cat([c1, self.dconv4(out)], dim=1)
        out = self.conv9(c1)
        
        out = self.conv10(out)
        return out

class Classifer(nn.Module):
    def __init__(self, in_channel=1, classes_num=1, p=0.5):
        super().__init__()
        self.conv1 = ConvBn(in_channel, 8)
        self.conv2 = ConvBn(8, 16)
        self.conv3 = ConvBn(16, 32)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 16 * 16, out_features=128),
            nn.ReLU(), nn.Dropout(p),
            nn.Linear(in_features=128, out_features=classes_num),
            nn.Sigmoid(),
        )
        return
    
    def forward(self, x):
        out = F.max_pool2d(self.conv1(x), kernel_size=(4, 4), stride=4)
        out = F.max_pool2d(self.conv2(out), kernel_size=(4, 4), stride=4)
        out = F.max_pool2d(self.conv3(out), kernel_size=(4, 4), stride=4)
        out = self.linear(out)
        return out
    



# ---TEST---

if __name__ == "__main__":
    CONFIG  = load_config()
    CUDA    = CONFIG["cuda"]

    inputs    = torch.randn(4, 3, 1024, 1024)
    unet      = Unet()
    classifer = Classifer(in_channel=1, classes_num=1)

    inputs    = inputs.cuda() if CUDA else inputs
    model     = unet.cuda() if CUDA else unet
    classifer = classifer.cuda() if CUDA else classifer
    outputs   = model(inputs)
    classes   = classifer(outputs)

    print(inputs.shape)
    print(model.eval(), classifer.eval())
    print(outputs.shape, classes.shape)