import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MS(nn.Module):
    def __init__(self, in_channels,ch3x3red,ch3x3,ch5x5red,ch5x5,ch7x7red,ch7x7):
        super(MS, self).__init__()

        self.conv_3x3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1,stride=1)
        )

        self.conv_5x5 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2,stride=1)
        )

        self.conv_7x7 = nn.Sequential(
            BasicConv2d(in_channels, ch7x7red, kernel_size=1),
            BasicConv2d(ch7x7red, ch7x7, kernel_size=7,padding='same')
        )

    def forward(self, x):
        x_fused = x
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_out =torch.cat([x_3x3, x_5x5, x_7x7], dim=1)
        return x_out

if __name__ == '__main__':
    x = torch.randn(4, 64, 128, 128).cuda()
    model = MS(in_channels=64,ch3x3red=32,ch3x3=64,ch5x5red=32,ch5x5=64,ch7x7red=32,ch7x7=32).cuda()
    out = model(x)
    print(out.shape)