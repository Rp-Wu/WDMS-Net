import torch
import torch.nn as nn
from thop import profile,clever_format
# $ git clone https://github.com/fbcotter/pytorch_wavelets
# $ cd pytorch_wavelets
# $ pip install .
# more details at https://pytorch-wavelets.readthedocs.io/en/latest/
from pytorch_wavelets.dwt.transform2d import DWTForward
# ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
# ['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal', 'Reverse biorthogonal', 'Discrete Meyer (FIR Approximation)', 'Gaussian', 'Mexican hat wavelet', 'Morlet wavelet', 'Complex Gaussian wavelets', 'Shannon wavelets', 'Frequency B-Spline wavelets', 'Complex Morlet wavelets']

class WD(nn.Module):
    def __init__(self,waveform):
        super(WD, self).__init__()
        self.dwt = DWTForward(J=1, wave=waveform)#若要通道上concat，mode='periodization'是必不可少的保持宽高严格减半对齐的操作
    def forward(self, x):
        xll,_ = self.dwt(x)
        return xll

if __name__ == "__main__":
        model = WD(waveform='haar')
        data = torch.randn(1,256,112,112)
        data=model(data)
        print(data.shape)
        flops, params = profile(model.cuda(), (data.cuda(),))
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)