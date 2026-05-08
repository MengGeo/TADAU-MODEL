import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
from Transform import EmbedTrans

# *************************模块说明*************************
# 可以学习的子波
class learnWavelet(nn.Module):
    def __init__(self, wavelet_len=51, init_freq=60, dt=0.001,*args):
        super().__init__()
        # 将子波长度转换为奇数，确保对称性
        n = wavelet_len if wavelet_len % 2 else wavelet_len + 1
        T = dt * (n // 2)  # 时间半长
        t = torch.linspace(-T, T, n)  # 对称时间序列
        pi = np.pi
        pf = (pi * init_freq) ** 2
        init_wave = (1 - 2 * pf * t ** 2) * torch.exp(-pf * t ** 2)  # 初始零相位子波
        # w = np.load(r'/hd25/2026mdj/python/TADAU/inversion_model/wave.npz')
        # init_wave = w['wave']
        # init_wave=torch.FloatTensor(init_wave)
        # 定义可学习的子波参数
        self.wavelet = nn.Parameter(init_wave.float())
        # self.wavelet_len = wavelet_len

    def forward(self):
        # 约束1: 强制零相位化
        wavelet_sym = (self.wavelet + torch.flip(self.wavelet, dims=[0])) / 2
        # 峰值归一化
        wavelet_norm = wavelet_sym / torch.max(torch.abs(wavelet_sym) + 1e-8)
        # 去直流分量，避免子波偏置
        wavelet_clean = wavelet_norm - torch.mean(wavelet_norm)
        return wavelet_clean

# 固定的子波
class fixWavelet(nn.Module):
    def __init__(self, wavelet_len=51, init_freq=60, dt=0.001):
        super().__init__()
        # 将子波长度转换为奇数，确保对称性
        n = wavelet_len if wavelet_len % 2 else wavelet_len + 1
        T = dt * (n // 2)  # 时间半长
        t = torch.linspace(-T, T, n)  # 对称时间序列
        pi = np.pi
        pf = (pi * init_freq) ** 2
        init_wave = (1 - 2 * pf * t ** 2) * torch.exp(-pf * t ** 2)  # 初始零相位子波
        # 定义可学习的子波参数
        self.wavelet = init_wave
        # self.wavelet_len = wavelet_len

    def forward(self):
        # 约束1: 强制零相位化
        wavelet_sym = (self.wavelet + torch.flip(self.wavelet, dims=[0])) / 2
        # 峰值归一化
        wavelet_norm = wavelet_sym / torch.max(torch.abs(wavelet_sym) + 1e-8)
        # 去直流分量，避免子波偏置
        wavelet_clean = wavelet_norm - torch.mean(wavelet_norm)
        return wavelet_clean

class SeisExtract(nn.Module):
    def __init__(self,in_channels, out_channels, large_size=(61, 1), small_size=7):
        super().__init__()
        # 高度方向
        hkernel_size = large_size[0] if large_size[0] % 2 else large_size[0] + 1
        # 宽度方向
        wkernel_size = large_size[1] if large_size[1] % 2 else large_size[1] + 1
        self.large_size = (hkernel_size, wkernel_size)  # 二维卷积尺寸
        self.small_size = small_size  # 二维卷积尺寸
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
            nn.Conv2d(2*out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(self.small_size,1), padding=(self.small_size//2,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        #X:[B,1,H,W]
        feat_large = self.large(x)  #[B,C,H,W]
        feat_small = self.small(x)  #[B,C,H,W]
        #cat:[B,2*C,H,W]
        feat_cat = torch.cat(tensors=[feat_large, feat_small], dim=1)
        feat_fused= self.bottleneck(feat_cat)
        return feat_fused

class OneConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        # assert kernel_size in (3, 5), "kernel_size must be 3 or 5"
        padding_value = (kernel_size - 1) // 2
        # 第一层:一维卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,padding=padding_value),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x1):
        return self.conv1(x1)

# 两个卷积层，为上采样和下采样提供卷积函数
#input:(B,C,H,A)
#output:(B,C,H,A)
class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        #assert kernel_size in (3, 5), "kernel_size must be 3 or 5"
        padding_value = (kernel_size - 1) // 2
        # 第一层:一维卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,padding=padding_value),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # 第二层:一维卷积层，手动计算填充值
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                      padding=padding_value),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x1):
        return self.conv2(self.conv1(x1))

# 下采样卷积层: 最大池化层+卷积层+卷积层
# 仅对H维度下采样，输出数据H尺寸降为H/2
# input:(B,C,H,A)
# output:(B,C,H/2,A)
class DownMaxConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        # 第一层: 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)
        # 两个卷积层
        self.doubleconv = DoubleConv2d(in_channels, out_channels, kernel_size)
    def forward(self, x1):
        x1 = self.pool(x1)
        x1 = self.doubleconv(x1)
        return x1

# 上采样+门控: 上采样+门控+卷积层+卷积层
# 包含了跨层特征融合
# input:x1(上采样特征),x2(跨层特征),in_channels=2*out_channels
# # input:x1.shape:(B,2*C,H/2,A),x_attn.shape:(B,C,H,A)
# # output:(B,C,H,A)
class UpGate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        padding_value = int((kernel_size - 1) // 2)
        # 上采样·数据长度变为原来2倍
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 1), stride=(2, 1), padding=0)
        #高层全局特征
        self.W_G = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        # 低层全局特征
        self.W_L = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        #门控
        self.gate = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid())
        # 两个卷积层
        self.doubleconv = DoubleConv2d(out_channels, out_channels, kernel_size)
    def forward(self, x1, x_attn):
        #x1:(B,C,H/2,A) to (B,C,H,A)
        x_up = self.up(x1)  # (B,C,H/2,A) to (B,C,H,A)
        x_up = self.W_G(x_up)
        x_L = self.W_L(x_attn)
        # 跨层特征融合+门控
        x_cat= torch.cat(tensors=[x_up, x_L], dim=1)  # (N, 2*C, H, A)
        x_gate = self.gate(x_cat)  # (N, C, H, A)
        xout = x_up*(1-x_gate)+ x_L * x_gate  # x_attn.shape:(B,C,H,A)
        xout = self.doubleconv(xout)  # (N, C, H, A)
        return xout

#Time self-attention
class TimeSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, height=32, num_heads=4, num_layers=3, dropout=0.0):
        super().__init__()
        self.in_channels = embed_dim
        self.height = height
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn = EmbedTrans(in_channels=self.in_channels, height=self.height,
                               num_heads=self.num_heads, num_layers=self.num_layers,dropout=dropout)
    def forward(self, xin):
        #x:(B,C,H,A)
        B, C, H, A = xin.shape
        #维度转换(B, C, H, A)-(B, A, C, H)-(B*A, C, H)
        x_ang = xin.permute(0, 3, 1, 2).reshape(B*A,C, H)
        #TIME内自注意力计算
        x_out= self.attn(x_ang)  #(B*A, C, H)
        #维度还原
        x_out =x_out.reshape(B,A,C,H).permute(0,2,3,1)
        return x_out

#Angle self-attention
class AngleSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, height=13, num_heads=2, num_layers=3, dropout=0.0):
        super().__init__()
        self.in_channels = embed_dim
        self.num_heads = num_heads
        self.height = height
        self.num_layers = num_layers
        self.attn = EmbedTrans(in_channels=self.in_channels, height=self.height,
                               num_heads=self.num_heads, num_layers=self.num_layers,dropout=dropout)
    def forward(self, xin):
        #x:(B,C,H,A)
        B, C, H, A = xin.shape
        #维度转换(B, C, H, A)-(B, H, C, A)-(B*H, C, A)
        x_ang = xin.permute(0, 2, 1, 3).reshape(B*H, C, A)
        #角度内自注意力计算
        x_out = self.attn(x_ang)  #(B*H, C, A)
        #维度还原
        x_out =x_out.reshape(B,H,C,A).permute(0, 2, 1, 3)
        return x_out

# Angle and Time attention
class TimeAngleAttention(nn.Module):
    def __init__(self, embed_dim=64, angle_height=13,time_height=32, num_heads=4, num_layers=3, dropout=0.0):
        super().__init__()
        self.TimeAtt = TimeSelfAttention(embed_dim=embed_dim, height=time_height, num_heads=num_heads,
                                         num_layers=num_layers, dropout=dropout)
        self.AngleAtt = AngleSelfAttention(embed_dim=embed_dim, height=angle_height, num_heads=num_heads,
                                           num_layers=num_layers, dropout=dropout)
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU()
        self.Oneconv2d = OneConv2d(embed_dim, embed_dim, kernel_size=5)
    def forward(self, xin):
        x_time = self.TimeAtt(xin)
        x_angle = self.AngleAtt(x_time)

        # 残差连接+归一化激活
        x_out = self.relu(self.norm(x_angle + self.conv(xin)))
        x_out= self.Oneconv2d(x_out)
        return x_out

class MultiTask(nn.Module):
    def __init__(self, in_channels=64,nwidth_in=13):
        super().__init__()
        self.in_channels = in_channels
        self.w = nwidth_in

        self.head_vp = self._task_vel(self.in_channels)
        self.head_vs = self._task_vel(self.in_channels)
        self.head_den = self._task_vel(self.in_channels)

        self.angle_com_vp = self._angle_compress(self.in_channels, self.w)
        self.angle_com_vs = self._angle_compress(self.in_channels, self.w)
        self.angle_com_den = self._angle_compress(self.in_channels, self.w)

    def _angle_compress(self,channels,w):
        k1= w//2
        k2= w-k1+1
        return nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=(1,k1)),
            nn.ReLU(),
            nn.Conv2d(channels//2, 1, kernel_size=(1,k2)),
            nn.GELU(),
        )
    def _task_vel(self,in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
        )
    def forward(self, xin):
        # angele compress
        feat_vp=self.head_vp(xin) #(N,C,H,W)
        feat_vs=self.head_vs(xin)
        feat_den=self.head_den(xin)
        #task
        res_vp=self.angle_com_vp(feat_vp) #(N,1,H,1)
        res_vs=self.angle_com_vs(feat_vs)
        res_den=self.angle_com_den(feat_den)
        out =torch.cat(tensors= [res_vp,res_vs,res_den],dim=-1) #(N,1,H,3)
        return out

# 定义角度和时间双注意力反演模型
class TADAU(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, height=256, nwidth_in=13, nwidth_out=3, kernel_size=61,
                 angle=torch.arange(8, 34, 2)):
        super().__init__()
        # 定义一些设置参数
        self.num_layers = 1# Transformer层数
        self.num_heads = 4  # 多头注意力头数
        self.down_height = int(height / 8)  # 下采样后数据的尺寸
        self.dropout = 0.1
        self.angle = angle
        self.learnable = True
        # 将卷积尺寸转换为奇数
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.kz = kernel_size  # 卷积尺寸
        self.s_size = 7
        self.nwidth = nwidth_in  # 输入数据宽度
        self.W_out = nwidth_out  # 输出数据宽度
        # 地震信号特征提取
        self.seis_conv = SeisExtract(in_channels, out_channels, large_size=(self.kz,1),small_size=self.s_size)

        # 第一次下采样，数据长度变为1/2
        self.downsample1 = DownMaxConv2d(out_channels, out_channels, kernel_size=self.s_size)
        # 第二次下采样，数据长度变为1/4
        self.downsample2 = DownMaxConv2d(out_channels, 2*out_channels, kernel_size=self.s_size)
        # 第三次下采样，数据长度变为1/8
        self.downsample3 = DownMaxConv2d(2*out_channels, 4*out_channels, kernel_size=self.s_size)
        # Angle and time attention
        self.ATAtt = TimeAngleAttention(4 * out_channels, self.nwidth, self.down_height, num_heads=self.num_heads,
                                        num_layers=self.num_layers, dropout=self.dropout)
        # 第一次上采样·数据长度变为1/4
        self.upsample1 = UpGate(4 * out_channels, 2 * out_channels, kernel_size=self.s_size)
        # 第二次上采样，数据长度变为1/2
        self.upsample2 = UpGate(2 * out_channels, out_channels, kernel_size=self.s_size)
        # 第三次上采样数据长度恢复
        self.upsample3 = UpGate(out_channels, out_channels, kernel_size=self.s_size)

        # 输出层，与标签数据对应（b,1,height,w)
        self.out = MultiTask(out_channels,self.nwidth)

        # 定义地震子波是固定还是可以更新
        if self.learnable:
            self.learn_wavelet = learnWavelet()
        else:
            self.learn_wavelet = fixWavelet()

    def forward(self, s, low):  # (b,1,H,W)
        # 提取地震信号特征
        s1 = self.seis_conv(s)  # (b,c,h,w)
        # unet下采样
        d1 = self.downsample1(s1) # (b,c,h/2,w)
        d2 = self.downsample2(d1) # (b,2*c,h/4,w)
        d3 = self.downsample3(d2) # (b,4*c,h/8,w)
        # Angle and time attention
        d3 = self.ATAtt(d3)
        tu = d3
        # unet上采样
        up1 = self.upsample1(tu, d2) #tu(b,4*c,h/8,w),d2(b,2*c,h/4,w)-up1(b,2*c,h/4,w)
        up2 = self.upsample2(up1, d1) #d1(b,c,h/2,w),up2(b,c,h/2,w)
        up3 = self.upsample3(up2, s1) #s1(b,c,h,w),up3(b,c,h,w)
        # 升维与标签数据相匹配
        out = self.out(up3) # (b,1,height,3)
        out = out + low
        wavelet = self.learn_wavelet()
        return out,wavelet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TADAU = TADAU()
    total_params=sum(p.numel() for p in TADAU.parameters())
    print(f'Total number of parameters: {total_params}')
    summary(TADAU, input_size=[(64, 1, 256, 13), (64, 1, 256, 3)])