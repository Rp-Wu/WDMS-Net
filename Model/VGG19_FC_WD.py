import torch
import torch.nn as nn
from thop import profile,clever_format
from Model.WaveDown_Block import WD

##随机初始化种子固定可复现：
seed = 49#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
# np.random.seed(seed)
# random.seed(seed)
# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True#除非为了让训练结果“完全一致”，其他情况下不要加这一行，简单设置一下那几个 seed 就足够了。
# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False

class VGG19_FC_WD(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19_FC_WD, self).__init__()
        self.stage1_channels = 64
        self.stage2_channels = 128
        self.stage3_channels = 256
        self.stage4_channels = 512
        self.stage5_channels = 512
        self.features1 = nn.Sequential(
            # Block 1
            nn.Conv2d(3, self.stage1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage1_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage1_channels, self.stage1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage1_channels),
            nn.ReLU(inplace=True),
            WD(waveform = 'haar')
        )
        self.features2 = nn.Sequential(
            # Block 2
            nn.Conv2d(self.stage1_channels, self.stage2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage2_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage2_channels, self.stage2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage2_channels),
            nn.ReLU(inplace=True),
            WD(waveform = 'haar')
        )
        self.features3 = nn.Sequential(
            # Block 3
            nn.Conv2d(self.stage2_channels, self.stage3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage3_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage3_channels, self.stage3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage3_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage3_channels, self.stage3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage3_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage3_channels, self.stage3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage3_channels),
            nn.ReLU(inplace=True),
            WD(waveform = 'haar')
        )
        self.features4 = nn.Sequential(
            # Block 4
            nn.Conv2d(self.stage3_channels, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage4_channels, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage4_channels, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage4_channels, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            WD(waveform = 'haar')
        )
        self.features5 = nn.Sequential(
            # Block 5
            nn.Conv2d(self.stage4_channels, self.stage5_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage5_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage5_channels, self.stage5_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage5_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage5_channels, self.stage5_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage5_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage5_channels, self.stage5_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage5_channels),
            nn.ReLU(inplace=True),
            WD(waveform = 'haar')
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(self.stage5_channels, num_classes),
        )

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        avg = self.avgpool(x5)
        x = torch.flatten(avg, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
        model = VGG19_FC_WD(num_classes=10)
        model.cuda()
        data = torch.ones((1, 3, 299,299)).cuda()
        # out = model.forward(data)
        # print(out.shape)
        flops, params = profile(model.cuda(), (data,))
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)

