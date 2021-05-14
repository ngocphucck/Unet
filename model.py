import torch
from torch.nn import Module
from torch import nn


from dataset import XrayDataset


class Down(Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_tensor):
        output_tensor = self.conv(input_tensor)

        return self.pooling(output_tensor), output_tensor


class Up(Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up_pooling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input_tensor, concat_tensor):
        output_tensor = self.up_pooling(input_tensor)
        output_tensor = torch.cat([concat_tensor, input_tensor], dim=1)

        return output_tensor


class Unet(Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down1 = Down(in_channels=1, out_channels=32)
        self.down2 = Down(in_channels=32, out_channels=64)
        self.down3 = Down(in_channels=64, out_channels=128)
        self.down4 = Down(in_channels=128, out_channels=256)
        self.down_conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.up_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = Up(in_channels=512, out_channels=216)
        self.up3 = Up(in_channels=128, out_channels=64)
        self.up2 = Up(in_channels=64, out_channels=32)
        self.up1 = Up(in_channels=64, out_channels=32)

        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        output1 = self.down1(input_tensor)
        output2 = self.down2(output1[0])
        output3 = self.down3(output2[0])
        output4 = self.down4(output3[0])
        output5 = self.relu(self.down_conv(output4[0]))

        print(output5.shape)
        output_tensor = self.relu(self.up_conv(output5))
        print(output_tensor.shape)
        output_tensor = self.up4(output4[-1], output_tensor)
        output_tensor = self.up3(output3[-1], output_tensor)
        output_tensor = self.up2(output2[-1], output_tensor)
        output_tensor = self.up1(output1[-1], output_tensor)

        return output_tensor


if __name__ == '__main__':
    dataset = XrayDataset()
    model = Unet()

    print(model(dataset[0][0].unsqueeze(0)))
    pass
