import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary

# ***************************************************************
#
class learnWavelet(nn.Module):
    def __init__(self, wavelet_len=51, init_freq=60, dt=0.001,*args):
        super().__init__()
        #
        n = wavelet_len if wavelet_len % 2 else wavelet_len + 1
        T = dt * (n // 2)  #
        t = torch.linspace(-T, T, n)  #
        pi = np.pi
        pf = (pi * init_freq) ** 2
        init_wave = (1 - 2 * pf * t ** 2) * torch.exp(-pf * t ** 2)  #

        self.wavelet = nn.Parameter(init_wave.float())
        # self.wavelet_len = wavelet_len

    def forward(self):
        #
        wavelet_sym = (self.wavelet + torch.flip(self.wavelet, dims=[0])) / 2
        #
        wavelet_norm = wavelet_sym / torch.max(torch.abs(wavelet_sym) + 1e-8)
        #
        wavelet_clean = wavelet_norm - torch.mean(wavelet_norm)
        return wavelet_clean

#
class fixWavelet(nn.Module):
    def __init__(self, wavelet_len=51, init_freq=60, dt=0.001):
        super().__init__()
        #
        n = wavelet_len if wavelet_len % 2 else wavelet_len + 1
        T = dt * (n // 2)  #
        t = torch.linspace(-T, T, n)  #
        pi = np.pi
        pf = (pi * init_freq) ** 2
        init_wave = (1 - 2 * pf * t ** 2) * torch.exp(-pf * t ** 2)  #
        #
        self.wavelet = init_wave
        # self.wavelet_len = wavelet_len

    def forward(self):
        #
        wavelet_sym = (self.wavelet + torch.flip(self.wavelet, dims=[0])) / 2
        #
        wavelet_norm = wavelet_sym / torch.max(torch.abs(wavelet_sym) + 1e-8)
        #
        wavelet_clean = wavelet_norm - torch.mean(wavelet_norm)
        return wavelet_clean

class SeisExtract(nn.Module):
    def __init__(self,in_channels, out_channels, large_size=(61, 1), small_size=7):
        super().__init__()
        #
        hkernel_size = large_size[0] if large_size[0] % 2 else large_size[0] + 1
        #
        wkernel_size = large_size[1] if large_size[1] % 2 else large_size[1] + 1
        self.large_size = (hkernel_size, wkernel_size)  #
        self.small_size = small_size #
        hpadding = int((self.large_size[0] - 1) / 2)
        wpadding = int((self.large_size[1] - 1) / 2)
        paddings = (hpadding, wpadding)

        #大尺寸
        self.large = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.large_size, padding=paddings),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        #小尺寸
        self.small = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.small_size,1), padding=(self.small_size//2,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        #fuse
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size= 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(self.small_size,1), padding=(self.small_size//2,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        #X:[B,1,H,W]
        feat_large = self.large(x)  #[B,C,H,W]
        feat_small = self.small(x)  #[B,C,H,W]
        #cat:#[B,2*C,H,W]
        feat_cat = torch.cat([feat_large, feat_small], dim=1)
        feat_fused= self.bottleneck(feat_cat)
        return feat_fused


class OneConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        #assert kernel_size in (3, 5), "kernel_size must be 3 or 5"
        padding_value = (kernel_size - 1) // 2
        #
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,padding=padding_value),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x1):
        return self.conv1(x1)

#
#input:(B,C,H,A)
#output:(B,C,H,A)
class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        #assert kernel_size in (3, 5), "kernel_size must be 3 or 5"
        padding_value = (kernel_size - 1) // 2
        #
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,padding=padding_value),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                      padding=padding_value),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x1):
        return self.conv2(self.conv1(x1))


# input:(B,C,H,A)
# output:(B,C,H/2,A)
class DownMaxConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=7):
        super().__init__()
        #
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)
        #
        self.doubleconv = DoubleConv2d(in_channels, out_channels,kernel_size)

    def forward(self, x1):
        x1 = self.pool(x1)
        x1 = self.doubleconv(x1)
        return x1


# input:x1,x2,in_channels=2*out_channels
# # input:x1.shape:(B,2*C,H/2,A),x_attn.shape:(B,C,H,A)
# # output:(B,C,H,A)
class UpGate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        padding_value = int((kernel_size - 1) // 2)
        #
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 1), stride=(2, 1), padding=0)
        #
        self.W_G = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        #
        self.W_L = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        #
        self.gate = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid())
        #
        self.doubleconv = DoubleConv2d(out_channels, out_channels, kernel_size)
    def forward(self, x1, x_attn):
        #x1:(B,C,H/2,A) to (B,C,H,A)
        x_up = self.up(x1) # (B,C,H/2,A) to (B,C/2,H,A)
        x_up = self.W_G(x_up)
        x_L = self.W_L(x_attn)
        #
        xout = x_up + x_L
        xout = self.doubleconv(xout)  # (N, C, H, A)
        return xout


#
class Unet_model(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, height=256, nwidth_in=13,  nwidth_out=3, kernel_size=61,
                 angle=torch.arange(8, 34, 2)):
        super().__init__()
        #
        self.down_height = int(height / 8)  #
        self.dropout = 0.1
        self.angle = angle
        self.learnable = True
        #
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.kz = kernel_size  #
        self.s_size = 7
        self.nwidth = nwidth_in  #
        self.W_out = nwidth_out  #
        #
        self.seis_conv = SeisExtract(in_channels, out_channels, large_size=(self.kz,1),small_size=self.s_size)

        #
        self.downsample1 = DownMaxConv2d(out_channels, out_channels, kernel_size=self.s_size)
        #
        self.downsample2 = DownMaxConv2d(out_channels, 2*out_channels, kernel_size=self.s_size)
         #
        self.downsample3 = DownMaxConv2d(2*out_channels, 4*out_channels, kernel_size=self.s_size)

        #
        self.upsample1 = UpGate(4 * out_channels, 2 * out_channels, kernel_size=self.s_size)
        #
        self.upsample2 = UpGate(2 * out_channels, out_channels, kernel_size=self.s_size)
        #
        self.upsample3 = UpGate(out_channels, out_channels, kernel_size=self.s_size)
        # (b,1,height,w）
        self.out2 = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Linear(nwidth_in,3),
            nn.GELU())

         #
        if self.learnable:
            self.learn_wavelet = learnWavelet()
        else:
            self.learn_wavelet = fixWavelet()

    def forward(self, s, low):  # (b,1,h,w)
        #
        s1 = self.seis_conv(s)  # (b,c,h,w)
        # unet
        d1 = self.downsample1(s1) # (b,c,h/2,w)
        d2 = self.downsample2(d1) # (b,2*c,h/4,w)
        d3 = self.downsample3(d2) # (b,4*c,h/8,w)
        tu = d3
        # unet
        up1 = self.upsample1(tu, d2) #tu(b,4*c,h/8,w),d2(b,2*c,h/4,w)-up1(b,2*c,h/4,w)
        up2 = self.upsample2(up1, d1) #d1(b,c,h/2,w),up2(b,2*c,h/2,w)
        up3 = self.upsample3(up2, s1) #s1(b,c,h,w),up3(b,c,h,w)

        #
        out = self.out2(up3)        # (b,1,height,3)
        out = out + low
        wavelet = self.learn_wavelet()
        return out,wavelet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Unet_model = Unet_model()
    total_params=sum(p.numel() for p in Unet_model.parameters())
    print(f'Total number of parameters: {total_params}')
    summary(Unet_model, input_size=[(64, 1, 256, 13), (64, 1, 256, 3)])
