from load_config import load_config
from .unet_parts import *

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
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits

class Classifier(nn.Module):
    def __init__(self, in_channel=1, classes_num=2, p=0.5):
        super().__init__()
        self.conv1 = DoubleConv(in_channel, 16)
        self.conv2 = DoubleConv(16, 32)
        self.conv3 = DoubleConv(32, 64)
        self.linear = nn.Sequential(
            nn.Flatten(),
            # 1024 * 1024
            # nn.Linear(in_features=32 * 16 * 16, out_features=128),
            # 256 * 256
            nn.Linear(in_features=64 * 4 * 4, out_features=1024),
            nn.ReLU(), nn.Dropout(p, inplace=False),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(), nn.Dropout(p, inplace=False),
            nn.Linear(in_features=64, out_features=classes_num),
            nn.Softmax(),
        )
        return

    def forward(self, x):
        out = F.max_pool2d(self.conv1(x),   kernel_size=(4, 4), stride=4)
        out = F.max_pool2d(self.conv2(out), kernel_size=(4, 4), stride=4)
        out = F.max_pool2d(self.conv3(out), kernel_size=(4, 4), stride=4)
        # print('---test---')
        # print(out.shape)
        out = self.linear(out)
        return out



# ---TEST---

if __name__ == "__main__":
    CONFIG  = load_config()
    CUDA    = CONFIG["cuda"]

    inputs    = torch.randn(4, 3, 1024, 1024)
    unet      = UNet()
    classifer = Classifier(in_channel=1, classes_num=1)

    inputs    = inputs.cuda() if CUDA else inputs
    model     = unet.cuda() if CUDA else unet
    classifer = classifer.cuda() if CUDA else classifer
    outputs   = model(inputs)
    classes   = classifer(outputs)

    print(inputs.shape)
    print(model.eval(), classifer.eval())
    print(outputs.shape, classes.shape)