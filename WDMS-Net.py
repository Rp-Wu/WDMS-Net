import torch
import torch.nn as nn
from thop import profile,clever_format
from MultiScale_Layer import MS
from WaveDown_Block import WD

class WDMS_Net(nn.Module):
    def __init__(self, num_classes=10, seed=49):
        super(WDMS_Net, self).__init__()
        # 在构建网络层之前设置随机种子，保证权重初始化可复现且可更换
        torch.manual_seed(seed)#seed必须是int，可以自行设置
        torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
        # np.random.seed(seed)
        # random.seed(seed)
        # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
        # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
        torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
        torch.backends.cudnn.deterministic = True#除非为了让训练结果“完全一致”，其他情况下不要加这一行，简单设置一下那几个 seed 就足够了。
        # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
        torch.backends.cudnn.benchmark = False
        self.stage1_channels = 64
        self.stage2_channels = 128
        self.stage3_channels = 256
        self.stage4_channels = 512
        self.stage5_channels = 512

        self.features1 = nn.Sequential(
             nn.Conv2d(3, self.stage1_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(self.stage1_channels),
             nn.ReLU(inplace=True),
             nn.Conv2d(self.stage1_channels, self.stage1_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(self.stage1_channels),
             nn.ReLU(inplace=True),
             WD(waveform='haar')
        )
        self.features2 = nn.Sequential(
             nn.Conv2d(self.stage1_channels, self.stage2_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(self.stage2_channels),
             nn.ReLU(inplace=True),
             nn.Conv2d(self.stage2_channels, self.stage2_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(self.stage2_channels),
             nn.ReLU(inplace=True),
             WD(waveform='haar')
        )
        self.features3 = nn.Sequential(
            MS(in_channels=self.stage2_channels,ch3x3red=64, ch3x3=64, ch5x5red=64,
                 ch5x5=96, ch7x7red=32, ch7x7=32),
            MS(in_channels=192, ch3x3red=64, ch3x3=96, ch5x5red=64,
                 ch5x5=96, ch7x7red=32, ch7x7=48),
            MS(in_channels=240, ch3x3red=128, ch3x3=160, ch5x5red=64,
                 ch5x5=96, ch7x7red=48, ch7x7=56),
            nn.Conv2d(312, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            WD(waveform='haar')
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(512, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            # BasicBlock(in_channel=self.stage4_channels, out_channel=self.stage4_channels, kernel_size=3, padding=1),
            nn.Conv2d(self.stage4_channels, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage4_channels, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage4_channels, self.stage4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage4_channels),
            nn.ReLU(inplace=True),
            WD(waveform='haar')
        )
        self.features5 = nn.Sequential(
             nn.Conv2d(self.stage4_channels, self.stage5_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(self.stage5_channels),
             nn.ReLU(inplace=True),
            # BasicBlock(in_channel=self.stage5_channels, out_channel=self.stage5_channels, kernel_size=3, padding=1),
            nn.Conv2d(self.stage5_channels, self.stage5_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage5_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage5_channels, self.stage5_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage5_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage5_channels, self.stage5_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stage5_channels),
            nn.ReLU(inplace=True),
            WD(waveform='haar')
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.stage5_channels, num_classes)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        avg = self.avgpool(x5)
        flatten = torch.flatten(avg, 1)
        classes_predict = self.classifier(flatten)
        return classes_predict

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 全连接层参数
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
        model = WDMS_Net(num_classes=10)
        model.cuda()
        data = torch.ones((1, 3, 299, 299)).cuda()
        # out = model.forward(data)
        # print(out.shape)
        flops, params = profile(model.cuda(), (data,))
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)
        # print(model.parameters)
