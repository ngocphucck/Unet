import torch
from torch.nn import Module
from torch import nn
from torchsummary import summary


from dataset import XrayDataset


class DoubleConv(Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_tensor):
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.bn1(output_tensor)
        output_tensor = self.relu1(output_tensor)

        output_tensor = self.conv2(output_tensor)
        output_tensor = self.bn2(output_tensor)
        output_tensor = self.relu2(output_tensor)

        return output_tensor


class Down(Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_tensor):
        output_tensor = self.double_conv(input_tensor)

        return self.pool(output_tensor), output_tensor


class Up(Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=int(in_channels / 2), kernel_size=2,
                                          stride=2)
        self.relu = nn.ReLU()
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, concat_tensor, input_tensor):
        output_tensor = self.relu(self.up_conv(input_tensor))

        print(concat_tensor.shape)
        print(output_tensor.shape)
        output_tensor = torch.cat([concat_tensor, output_tensor], dim=1)
        output_tensor = self.double_conv(output_tensor)

        return output_tensor


class BottleNeck(Module):
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, input_tensor):
        output_tensor = self.relu1(self.bn1(self.conv1(input_tensor)))
        output_tensor = self.relu2(self.bn2(self.conv2(output_tensor)))

        return output_tensor


class Unet(Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down1 = Down(in_channels=1, out_channels=32)
        self.down2 = Down(in_channels=32, out_channels=64)
        self.down3 = Down(in_channels=64, out_channels=128)
        self.down4 = Down(in_channels=128, out_channels=256)
        self.bottle_neck = BottleNeck(in_channels=256, out_channels=512)

        self.up4 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up2 = Up(in_channels=128, out_channels=64)
        self.up1 = Up(in_channels=64, out_channels=32)

        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.classifier = nn.Sigmoid()

    def forward(self, input_tensor):
        output1 = self.down1(input_tensor)
        output2 = self.down2(output1[0])
        output3 = self.down3(output2[0])
        output4 = self.down4(output3[0])

        output_tensor = self.bottle_neck(output4[0])

        output_tensor = self.up4(output4[-1], output_tensor)
        output_tensor = self.up3(output3[-1], output_tensor)
        output_tensor = self.up2(output2[-1], output_tensor)
        output_tensor = self.up1(output1[-1], output_tensor)

        output_tensor = self.classifier(self.conv(output_tensor))

        return output_tensor


if __name__ == '__main__':
    dataset = XrayDataset()
    model = Unet()
    pass
