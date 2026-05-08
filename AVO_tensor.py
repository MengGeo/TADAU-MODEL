import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings


warnings.filterwarnings('ignore')


def avopp(vp1, vs1, d1, vp2, vs2, d2, ang, approx):

    if not isinstance(ang, torch.Tensor):
        ang = torch.tensor(ang, dtype=torch.float32)
    else:
        ang = ang.float()

    t = ang * np.pi / 180
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
    n = n if n % 2 else n + 1
    T = dt * (n // 2)
    t = torch.linspace(-T, T, n)  #
    pi = np.pi
    pf = (pi * f) ** 2
    s = (1 - 2 * pf * t ** 2) * torch.exp(-pf * t ** 2)
    return s, t


def avopp_vectorized(vp1, vs1, d1, vp2, vs2, d2, ang, approx):

    if not isinstance(ang, torch.Tensor):
        ang = torch.tensor(ang, dtype=torch.float32, device=vp1.device)
    else:
        ang = ang.float().to(vp1.device)


    t = ang * np.pi / 180  #
    num_angles = len(t)


    vp1 = vp1.unsqueeze(-1).expand(*vp1.shape, num_angles)
    vs1 = vs1.unsqueeze(-1).expand(*vs1.shape, num_angles)
    d1 = d1.unsqueeze(-1).expand(*d1.shape, num_angles)
    vp2 = vp2.unsqueeze(-1).expand(*vp2.shape, num_angles)
    vs2 = vs2.unsqueeze(-1).expand(*vs2.shape, num_angles)
    d2 = d2.unsqueeze(-1).expand(*d2.shape, num_angles)


    t = t.view(1, 1, 1, num_angles).expand_as(vp1)

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


        denom = (Dvp / vpa) + (Dd / da)

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


    vp1 = vp_vs_den[:, :, :-1, 0]  #
    vs1 = vp_vs_den[:, :, :-1, 1]  #
    d1 = vp_vs_den[:, :, :-1, 2]  #
    vp2 = vp_vs_den[:, :, 1:, 0]  #
    vs2 = vp_vs_den[:, :, 1:, 1]  #
    d2 = vp_vs_den[:, :, 1:, 2]  #


    rpp_partial = avopp_vectorized(vp1, vs1, d1, vp2, vs2, d2, angle, approx)
    # rpp_partial: (batch, channels, height-1, num_angles)


    zero_padding = torch.zeros(batch, channels, 1, num_angles,
                               device=vp_vs_den.device)

    #
    #  (batch, channels, height, num_angles)
    rpp = torch.cat([rpp_partial, zero_padding], dim=2)

    return rpp
# ===================== =====================
def wavelet_ref_conv(ref, wavelet):
    """
    :param ref:
    :param wavelet:
    :return: syn_gather
    """
    wavelet_len = wavelet.size(0)
    kernel = wavelet.view(1, 1, wavelet_len, 1).to(ref.device)
    pad_h = int((wavelet_len - 1) / 2)
    pad = (pad_h, 0)
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(wavelet_len, 1), padding=pad, bias=False)

    conv.weight.data = kernel
    conv.weight.requires_grad = False
    syn_gather = conv(ref)
    return syn_gather


if __name__ == '__main__':
    import time


    batch = 20
    channels = 16
    height = 30
    width = 3

    vp_vs_den = torch.rand(batch, channels, height, width) * 1000 + 1000
    vp_vs_den[:, :, :, 1] *= 0.5
    vp_vs_den[:, :, :, 2] *= 0.5 + 2.0
    print(vp_vs_den.shape)

    angle = torch.arange(0, 60, 2)
    approx = 1

    start_time = time.time()
    rpp_original = rpp_avo(vp_vs_den, angle, approx=approx)
    print(rpp_original.shape)
    original_time = time.time() - start_time
    print(f" {original_time:.4f}")


    start_time = time.time()
    rpp_vectorized = rpp_avo_vectorized(vp_vs_den, angle, approx=approx)
    print(rpp_vectorized.shape)
    vectorized_time = time.time() - start_time
    print(f" {vectorized_time:.4f}")

    #
    print(
        f" {torch.max(torch.abs(rpp_original - rpp_vectorized)):.10f}")
    print(f"max:{torch.max(rpp_original)},max:{torch.max(rpp_vectorized)}")
    print(f" {original_time / vectorized_time:.2f}")

    # 打印PyTorch版本，方便调试
    print(f"{torch.__version__}")
    print(f"{len(torch.arange(0, 41, 2))}")

    # 地层参数
    vp1 = 2000  #
    vs1 = 1800  #
    d1 = 1.85  #
    vp2 = 2300  #
    vs2 = 1900  #
    d2 = 1.9  #

    #
    approx_methods = {
        1: "Full Zoeppritz",
        2: "Aki & Richards",
        3: "Shuey",
        4: "Shuey Linear"
    }

    plt.figure(figsize=(12, 5))

    #
    for approx, label in approx_methods.items():
        re = avopp(vp1, vs1, d1, vp2, vs2, d2, torch.arange(0, 41, 2), approx)
        #
        plt.plot(torch.arange(0, 41, 2).numpy(), re.numpy(), marker='o',
                 linewidth=2, label=label)

    plt.xlabel("Angle of Incidence (deg)", fontsize=12)
    plt.ylabel("P-P Reflectivity", fontsize=12)
    plt.title("AVO Responses for Different Approximations", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # ---------------------- Ricker----------------------
    freq = 60  # (Hz)
    src_len = 50  #
    dt = 0.001  #

    src, tsrc = ricker(freq, src_len, dt)

    plt.figure(figsize=(10, 4))
    plt.plot(tsrc.numpy() * 1000, src.numpy(), '-k', linewidth=2)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.title(f"Ricker Wavelet (Frequency = {freq} Hz)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    #
    print("\n======")
    zero_angle_reflect = avopp(vp1, vs1, d1, vp2, vs2, d2, 0, 1).item()
    print(f"（Full Zoeppritz）: {zero_angle_reflect:.6f}")
    print(f"Ricker: {src.max().item():.6f}")
    print(f"Ricker: {tsrc[src.argmax()].item() * 1000:.2f} ms")

