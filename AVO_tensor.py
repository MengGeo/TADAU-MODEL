import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings

# 消除matplotlib警告
warnings.filterwarnings('ignore')


def avopp(vp1, vs1, d1, vp2, vs2, d2, ang, approx):
    # 确保输入角度是张量（支持单个角度或角度数组）
    if not isinstance(ang, torch.Tensor):
        ang = torch.tensor(ang, dtype=torch.float32)
    else:
        ang = ang.float()

    t = ang * np.pi / 180  # 角度转弧度
    p = torch.sin(t) / vp1
    ct = torch.cos(t)
    da = (d1 + d2) / 2
    Dd = (d2 - d1)
    vpa = (vp1 + vp2) / 2
    Dvp = (vp2 - vp1)
    vsa = (vs1 + vs2) / 2
    Dvs = (vs2 - vs1)

    if approx == 1:  # FULL Zoeppritz (A&K)
        # 避免根号下出现负数（超出临界角时设为0）
        sin2_t = torch.sin(t) ** 2
        vp_ratio2 = (vp2 / vp1) ** 2
        vs1_ratio2 = (vs1 / vp1) ** 2
        vs2_ratio2 = (vs2 / vp1) ** 2

        ct2 = torch.sqrt(torch.clamp(1 - sin2_t * vp_ratio2, min=1e-10))
        cj1 = torch.sqrt(torch.clamp(1 - sin2_t * vs1_ratio2, min=1e-10))
        cj2 = torch.sqrt(torch.clamp(1 - sin2_t * vs2_ratio2, min=1e-10))

        a = d2 * (1 - 2 * vs2 ** 2 * p ** 2) - d1 * (1 - 2 * vs1 ** 2 * p ** 2)
        b = d2 * (1 - 2 * vs2 ** 2 * p ** 2) + 2 * d1 * vs1 ** 2 * p ** 2
        c = d1 * (1 - 2 * vs1 ** 2 * p ** 2) + 2 * d2 * vs2 ** 2 * p ** 2
        d_term = 2 * (d2 * vs2 ** 2 - d1 * vs1 ** 2)

        E = b * ct / vp1 + c * ct2 / vp2
        F = b * cj1 / vs1 + c * cj2 / vs2
        G = a - d_term * ct * cj2 / (vp1 * vs2)
        H = a - d_term * ct2 * cj1 / (vp2 * vs1)
        D = E * F + G * H * p ** 2

        # 避免分母为0
        D = torch.clamp(D, min=1e-10)
        Rpp = ((b * ct / vp1 - c * ct2 / vp2) * F - (
                a + d_term * ct * cj2 / (vp1 * vs2)) * H * p ** 2) / D

    elif approx == 2:  # Aki & Richard (aprox)
        Rpp = 0.5 * (1 - 4 * p ** 2 * vsa ** 2) * Dd / da + Dvp / (
                2 * ct ** 2 * vpa) - 4 * p ** 2 * vsa * Dvs

    elif approx == 3:  # Shuey
        poi1 = ((0.5 * (vp1 / vs1) ** 2) - 1) / ((vp1 / vs1) ** 2 - 1)
        poi2 = ((0.5 * (vp2 / vs2) ** 2) - 1) / ((vp2 / vs2) ** 2 - 1)
        poia = (poi1 + poi2) / 2
        Dpoi = (poi2 - poi1)
        Ro = 0.5 * ((Dvp / vpa) + (Dd / da))

        # 修复：使用numpy的abs函数处理float类型
        denom = (Dvp / vpa) + (Dd / da)
        Bx = (Dvp / vpa) / denom if np.abs(denom) > 1e-10 else 0.0

        Ax = Bx - 2 * (1 + Bx) * (1 - 2 * poia) / (1 - poia)
        sin2_t = torch.sin(t) ** 2
        tan2_t = torch.tan(t) ** 2
        Rpp = Ro + ((Ax * Ro) + (
                Dpoi / (1 - poia) ** 2)) * sin2_t + 0.5 * Dvp * (
                      tan2_t - sin2_t) / vpa

    elif approx == 4:  # Shuey linear
        A = 0.5 * ((Dvp / vpa) + (Dd / da))
        B = (-2 * vsa ** 2 * Dd / (vpa ** 2 * da)) + (0.5 * Dvp / vpa) - (
                4 * vsa * Dvs / (vpa ** 2))
        Rpp = A + B * torch.sin(t) ** 2

    else:
        raise ValueError("approx must be 1-4")

    return Rpp


def rpp_avo(vp_vs_den, angle, approx=1):
    batch, chanels, height, width = vp_vs_den.shape
    rpp = torch.zeros(batch, chanels, height, len(angle))
    # 计算反射系数
    for i in range(batch):
        for j in range(chanels):
            for k in range(height - 1):
                vp1 = vp_vs_den[i, j, k, 0]
                vs1 = vp_vs_den[i, j, k, 1]
                d1 = vp_vs_den[i, j, k, 2]
                vp2 = vp_vs_den[i, j, k + 1, 0]
                vs2 = vp_vs_den[i, j, k + 1, 1]
                d2 = vp_vs_den[i, j, k + 1, 2]
                rpp_app = avopp(vp1, vs1, d1, vp2, vs2, d2, angle,
                                approx=approx)
                rpp[i, j, k, :] = rpp_app
    return rpp


def ricker(f, n, dt):
    """
    生成Ricker子波（雷克子波），一种常用的地震子波，具有零相位特性。
    Ricker子波的数学表达式为：s(t) = (1 - 2π²f²t²)e^(-π²f²t²)，
    其中f为子波主频，t为时间变量。
    参数:
        f (float): 雷克子波的中心频率（主频），单位通常为Hz
        n (int): 子波的采样点数（即子波长度），决定了生成的时间序列长度
        dt (float): 采样间隔（时间步长），单位通常为秒，决定了时间分辨率
    返回值:
        rw (torch.Tensor): 形状为(n,)的张量，存储生成的雷克子波振幅值，
                          振幅围绕0对称分布，符合零相位特性
        t (torch.Tensor): 形状为(n,)的张量，存储对应的时间点，
                         与子波振幅值一一对应，范围为[-T, T]
    """
    # 将子波长度转换为奇数，确保对称性
    n = n if n % 2 else n + 1
    T = dt * (n // 2)  # 时间半长
    t = torch.linspace(-T, T, n)  # 对称时间序列

    # 修复：使用numpy的pi替代torch.pi，兼容低版本PyTorch
    pi = np.pi
    pf = (pi * f) ** 2
    s = (1 - 2 * pf * t ** 2) * torch.exp(-pf * t ** 2)
    return s, t


def avopp_vectorized(vp1, vs1, d1, vp2, vs2, d2, ang, approx):
    """
    向量化版本的AVO反射系数计算
    支持 vp1, vs1, d1, vp2, vs2, d2 为任意形状的张量，只要它们形状相同
    ang 为角度张量，形状为 (num_angles,)
    """
    # 确保输入角度是张量
    if not isinstance(ang, torch.Tensor):
        ang = torch.tensor(ang, dtype=torch.float32, device=vp1.device)
    else:
        ang = ang.float().to(vp1.device)

    # 角度转弧度，并扩展维度以匹配输入参数
    t = ang * np.pi / 180  # 形状: (num_angles,)
    num_angles = len(t)

    # 扩展层对参数到角度维度
    # 假设 vp1 形状为 (batch, channels, height-1)
    # 扩展后形状为 (batch, channels, height-1, num_angles)
    vp1 = vp1.unsqueeze(-1).expand(*vp1.shape, num_angles)
    vs1 = vs1.unsqueeze(-1).expand(*vs1.shape, num_angles)
    d1 = d1.unsqueeze(-1).expand(*d1.shape, num_angles)
    vp2 = vp2.unsqueeze(-1).expand(*vp2.shape, num_angles)
    vs2 = vs2.unsqueeze(-1).expand(*vs2.shape, num_angles)
    d2 = d2.unsqueeze(-1).expand(*d2.shape, num_angles)

    # 扩展角度相关张量到层对维度
    t = t.view(1, 1, 1, num_angles).expand_as(vp1)  # 形状匹配vp1

    # 计算中间参数
    p = torch.sin(t) / vp1
    ct = torch.cos(t)
    da = (d1 + d2) / 2
    Dd = (d2 - d1)
    vpa = (vp1 + vp2) / 2
    Dvp = (vp2 - vp1)
    vsa = (vs1 + vs2) / 2
    Dvs = (vs2 - vs1)

    if approx == 1:  # FULL Zoeppritz (A&K)
        sin2_t = torch.sin(t) ** 2
        vp_ratio2 = (vp2 / vp1) ** 2
        vs1_ratio2 = (vs1 / vp1) ** 2
        vs2_ratio2 = (vs2 / vp1) ** 2

        ct2 = torch.sqrt(torch.clamp(1 - sin2_t * vp_ratio2, min=1e-10))
        cj1 = torch.sqrt(torch.clamp(1 - sin2_t * vs1_ratio2, min=1e-10))
        cj2 = torch.sqrt(torch.clamp(1 - sin2_t * vs2_ratio2, min=1e-10))

        a = d2 * (1 - 2 * vs2 ** 2 * p ** 2) - d1 * (1 - 2 * vs1 ** 2 * p ** 2)
        b = d2 * (1 - 2 * vs2 ** 2 * p ** 2) + 2 * d1 * vs1 ** 2 * p ** 2
        c = d1 * (1 - 2 * vs1 ** 2 * p ** 2) + 2 * d2 * vs2 ** 2 * p ** 2
        d_term = 2 * (d2 * vs2 ** 2 - d1 * vs1 ** 2)

        E = b * ct / vp1 + c * ct2 / vp2
        F = b * cj1 / vs1 + c * cj2 / vs2
        G = a - d_term * ct * cj2 / (vp1 * vs2)
        H = a - d_term * ct2 * cj1 / (vp2 * vs1)
        D = E * F + G * H * p ** 2

        # 避免分母为0
        D = torch.clamp(D, min=1e-10)
        Rpp = ((b * ct / vp1 - c * ct2 / vp2) * F - (
                a + d_term * ct * cj2 / (vp1 * vs2)) * H * p ** 2) / D

    elif approx == 2:  # Aki & Richard (aprox)
        Rpp = 0.5 * (1 - 4 * p ** 2 * vsa ** 2) * Dd / da + Dvp / (
                2 * ct ** 2 * vpa) - 4 * p ** 2 * vsa * Dvs

    elif approx == 3:  # Shuey
        poi1 = ((0.5 * (vp1 / vs1) ** 2) - 1) / ((vp1 / vs1) ** 2 - 1)
        poi2 = ((0.5 * (vp2 / vs2) ** 2) - 1) / ((vp2 / vs2) ** 2 - 1)
        poia = (poi1 + poi2) / 2
        Dpoi = (poi2 - poi1)
        Ro = 0.5 * ((Dvp / vpa) + (Dd / da))

        # 修复：使用torch.abs处理张量
        denom = (Dvp / vpa) + (Dd / da)
        # 使用where处理分母为0的情况
        Bx = torch.where(torch.abs(denom) > 1e-10, (Dvp / vpa) / denom,
                         torch.zeros_like(denom))

        Ax = Bx - 2 * (1 + Bx) * (1 - 2 * poia) / (1 - poia)
        sin2_t = torch.sin(t) ** 2
        tan2_t = torch.tan(t) ** 2
        Rpp = Ro + ((Ax * Ro) + (
                Dpoi / (1 - poia) ** 2)) * sin2_t + 0.5 * Dvp * (
                      tan2_t - sin2_t) / vpa

    elif approx == 4:  # Shuey linear
        A = 0.5 * ((Dvp / vpa) + (Dd / da))
        B = (-2 * vsa ** 2 * Dd / (vpa ** 2 * da)) + (0.5 * Dvp / vpa) - (
                4 * vsa * Dvs / (vpa ** 2))
        Rpp = A + B * torch.sin(t) ** 2

    else:
        raise ValueError("approx must be 1-4")

    return Rpp


def rpp_avo_vectorized(vp_vs_den, angle, approx=1):
    if vp_vs_den.shape[-1] != 3:
        raise ValueError("vp_vs_den must have width=3 (vp, vs, den)")

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, dtype=torch.float32,
                             device=vp_vs_den.device)
    else:
        angle = angle.float().to(vp_vs_den.device)

    batch, channels, height, _ = vp_vs_den.shape
    num_angles = len(angle)

    # 提取相邻层的参数
    vp1 = vp_vs_den[:, :, :-1, 0]  # 上覆层纵波速度 (batch, channels, height-1)
    vs1 = vp_vs_den[:, :, :-1, 1]  # 上覆层横波速度
    d1 = vp_vs_den[:, :, :-1, 2]  # 上覆层密度
    vp2 = vp_vs_den[:, :, 1:, 0]  # 下伏层纵波速度
    vs2 = vp_vs_den[:, :, 1:, 1]  # 下伏层横波速度
    d2 = vp_vs_den[:, :, 1:, 2]  # 下伏层密度

    # 计算前height-1层的反射系数
    rpp_partial = avopp_vectorized(vp1, vs1, d1, vp2, vs2, d2, angle, approx)
    # rpp_partial形状: (batch, channels, height-1, num_angles)

    # 创建零张量，用于补充最后一行
    zero_padding = torch.zeros(batch, channels, 1, num_angles,
                               device=vp_vs_den.device)

    # 在高度维度拼接，使输出高度与输入一致
    # 拼接后形状: (batch, channels, height, num_angles)
    rpp = torch.cat([rpp_partial, zero_padding], dim=2)

    return rpp
# ===================== 6. 子波卷积函数 =====================
def wavelet_ref_conv(ref, wavelet):
    """
    :param ref: 反射系数四维tensor (N,1,H,W)，N=批次，1=单通道，H=时间/深度，W=入射角
    :param wavelet: 子波，一维子波张量，shape=[wavelet_len]，如51点
    :return: syn_gather, 角道集
    """
    wavelet_len = wavelet.size(0)
    kernel = wavelet.view(1, 1, wavelet_len, 1).to(ref.device)
    pad_h = int((wavelet_len - 1) / 2)
    pad = (pad_h, 0)
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(wavelet_len, 1), padding=pad, bias=False)
    # 固定子波卷积核
    conv.weight.data = kernel
    conv.weight.requires_grad = False
    syn_gather = conv(ref)
    return syn_gather

# 测试参数
if __name__ == '__main__':
    import time

    # 创建大规模测试数据
    batch = 20
    channels = 16
    height = 30
    width = 3

    vp_vs_den = torch.rand(batch, channels, height, width) * 1000 + 1000
    vp_vs_den[:, :, :, 1] *= 0.5
    vp_vs_den[:, :, :, 2] *= 0.5 + 2.0
    print(vp_vs_den.shape)

    angle = torch.arange(0, 60, 2)  # 30个角度
    approx = 1
    # 测试原始函数
    start_time = time.time()
    rpp_original = rpp_avo(vp_vs_den, angle, approx=approx)
    print(rpp_original.shape)
    original_time = time.time() - start_time
    print(f"原始函数耗时: {original_time:.4f}秒")

    # 测试向量化函数
    start_time = time.time()
    rpp_vectorized = rpp_avo_vectorized(vp_vs_den, angle, approx=approx)
    print(rpp_vectorized.shape)
    vectorized_time = time.time() - start_time
    print(f"向量化函数耗时: {vectorized_time:.4f}秒")

    # 验证结果一致性
    print(
        f"结果差异: {torch.max(torch.abs(rpp_original - rpp_vectorized)):.10f}")
    print(f"max:{torch.max(rpp_original)},max:{torch.max(rpp_vectorized)}")
    print(f"性能提升: {original_time / vectorized_time:.2f}倍")

    # 打印PyTorch版本，方便调试
    print(f"PyTorch版本: {torch.__version__}")
    print(f"角度数量: {len(torch.arange(0, 41, 2))}")

    # 地层参数
    vp1 = 2000  # 上层P波速度 (m/s)
    vs1 = 1800  # 上层S波速度 (m/s)
    d1 = 1.85  # 上层密度 (g/cm³)
    vp2 = 2300  # 下层P波速度 (m/s)
    vs2 = 1900  # 下层S波速度 (m/s)
    d2 = 1.9  # 下层密度 (g/cm³)

    # 计算不同近似方法的反射系数
    approx_methods = {
        1: "Full Zoeppritz",
        2: "Aki & Richards",
        3: "Shuey",
        4: "Shuey Linear"
    }

    plt.figure(figsize=(12, 5))

    # 绘制所有近似方法的AVO曲线
    for approx, label in approx_methods.items():
        re = avopp(vp1, vs1, d1, vp2, vs2, d2, torch.arange(0, 41, 2), approx)
        # 转换为numpy数组绘图（matplotlib不直接支持张量）
        plt.plot(torch.arange(0, 41, 2).numpy(), re.numpy(), marker='o',
                 linewidth=2, label=label)

    plt.xlabel("Angle of Incidence (deg)", fontsize=12)
    plt.ylabel("P-P Reflectivity", fontsize=12)
    plt.title("AVO Responses for Different Approximations", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # ---------------------- Ricker子波生成与绘图 ----------------------
    freq = 60  # 子波主频 (Hz)
    src_len = 50  # 子波长度 (采样点)
    dt = 0.001  # 采样间隔 (s)

    src, tsrc = ricker(freq, src_len, dt)

    plt.figure(figsize=(10, 4))
    plt.plot(tsrc.numpy() * 1000, src.numpy(), '-k', linewidth=2)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.title(f"Ricker Wavelet (Frequency = {freq} Hz)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印关键结果
    print("\n=== 关键结果 ===")
    zero_angle_reflect = avopp(vp1, vs1, d1, vp2, vs2, d2, 0, 1).item()
    print(f"0度反射系数（Full Zoeppritz）: {zero_angle_reflect:.6f}")
    print(f"Ricker子波峰值振幅: {src.max().item():.6f}")
    print(f"Ricker子波峰值位置: {tsrc[src.argmax()].item() * 1000:.2f} ms")

