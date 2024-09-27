import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1  = Up(1024, 256, bilinear)
        self.up2  = Up(512, 128, bilinear)
        self.up3  = Up(256, 64, bilinear)
        self.up4  = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.outc = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        x1       = self.inc(x)
        x2       = self.down1(x1)
        x3       = self.down2(x2)
        x4       = self.down3(x3)
        x5       = self.down4(x4)
        x        = self.up1(x5, x4)
        x        = self.up2(x, x3)
        x        = self.up3(x, x2)
        x        = self.up4(x, x1)

        features=x
        x   = self.outc(x)
        return features,x

class Classifier(nn.Module):
    def __init__(self, in_channel=64, classes_num=2, p=0.5):
        super().__init__()
        self.conv1  = DoubleConv(in_channel, 128)
        self.conv2  = DoubleConv(128, 512)
        self.conv3  = DoubleConv(512, 1024)
        self.linear = nn.Sequential(
            nn.Flatten(),
            # 1024 * 1024
            # nn.Linear(in_features=32 * 16 * 16, out_features=128),
            # 256 * 256
            nn.Linear(in_features=1024 * 4 * 4, out_features=1024),
            nn.ReLU(), nn.Dropout(p, inplace=False),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(), nn.Dropout(p, inplace=False),
            nn.Linear(in_features=64, out_features=classes_num),
            nn.Softmax(),
        )
        return

    def forward(self, x):
        #print(x.shape)
        out = F.max_pool2d(self.conv1(x),   kernel_size=(4, 4), stride=4)
        out = F.max_pool2d(self.conv2(out), kernel_size=(4, 4), stride=4)
        out = F.max_pool2d(self.conv3(out), kernel_size=(4, 4), stride=4)
        # print('---test---')
        # print(out.shape)
        out = self.linear(out)
        # print(out.shape)
        return out



# ---TEST---

if __name__ == "__main__":
    CUDA       = True

    inputs     = torch.randn(1, 3, 256, 256)
    unet       = UNet()
    classifier = Classifier(in_channel=64, classes_num=2)

    inputs     = inputs.cuda() if CUDA else inputs
    model      = unet.cuda() if CUDA else unet
    classifier = classifier.cuda() if CUDA else classifier
    outputs   = model(inputs)
    print("output shape: \n", outputs[0].shape)
    classes    = classifier(outputs[0])
    print("classes shape: \n", classes.shape)

    print(inputs.shape)
    print(model.eval(), classifier.eval())
    print(outputs[0].shape, outputs[1].shape, classes.shape)