import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models

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
            nn.Linear(in_features=32 * 8 * 8, out_features=128),
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

ReLU = partial(F.relu, inplace=True)

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        return

    def forward(self, x):
        dilate1_out = ReLU(self.dilate1(x))
        dilate2_out = ReLU(self.dilate2(dilate1_out))
        dilate3_out = ReLU(self.dilate3(dilate2_out))
        dilate4_out = ReLU(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = ReLU

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = ReLU

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = ReLU
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

"""
inputs : FloatTensor[batch_size, 3, 1024, 1024]
outputs: FloatTensor[batch_size, 1, 1024, 1024]
"""       
class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, filters = [64, 128, 256, 512]):
        super(DinkNet34, self).__init__()

        resnet = models.resnet34(weights="DEFAULT")
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = ReLU
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = ReLU
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        return

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        
        return F.sigmoid(out)
    



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