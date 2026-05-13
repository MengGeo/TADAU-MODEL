import numpy as np
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import matplotlib.pyplot as plt

inv_data=np.load(r'inv_data_tadau&unet_test.npz')

valid_vpvsrho_log=inv_data['labeled_valid_vpvsrho']

vpvsrho_tadua=inv_data['inv_vpvsrho_tadau']
vpvsrho_unet=inv_data['inv_vpvsrho_unet']

time=inv_data['time']
linewidth = 1.2
word_size = 8

figure1 = plt.figure(figsize=(6, 6))
plt.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.96, hspace=0.3)
valid_number=6 #选择其中一条曲线对比结果
model_vp = valid_vpvsrho_log[valid_number,:,0]
inv_tadau_vp = vpvsrho_tadua[valid_number,:,0]
inv_unet_vp = vpvsrho_unet[valid_number,:,0]
plt_vp = figure1.add_subplot(3, 1, 1)
plt_vp.plot(time, model_vp, color='k', linestyle='-', linewidth=linewidth,
            label='Real')
#plt_vp.plot(time, low_vp, color='b', linestyle='-', linewidth=linewidth,label='LFM')
plt_vp.plot(time, inv_unet_vp, color='g', linestyle='--', linewidth=linewidth,
            label='UNet')
plt_vp.plot(time, inv_tadau_vp, color='r', linestyle='--', linewidth=linewidth,
            label='TADAU')
plt_vp.set_title(r"(a)", fontsize=word_size + 2, pad=5)
#plt_vp.set_xlabel("Time(s)",fontsize=word_size,labelpad=1)
plt_vp.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x / 1000:.2f}"))
plt_vp.set_ylabel("$\mathit{V_p}$(km/s)", fontsize=word_size, labelpad=1)
plt_vp.legend(fontsize=word_size - 2, loc='upper left', framealpha=0.6,
              edgecolor='k', fancybox=False, borderaxespad=0.5)
plt.xticks(size=word_size)
plt.yticks(size=word_size)
plt.ylim(1.6, 2.4)
plt.xlim(time[0], time[-1])

model_vs = valid_vpvsrho_log[valid_number,:,1]
inv_tadau_vs = vpvsrho_tadua[valid_number,:,1]
inv_unet_vs = vpvsrho_unet[valid_number,:,1]

plt_vs = figure1.add_subplot(3, 1, 2)
plt_vs.plot(time, model_vs, color='k', linestyle='-', linewidth=linewidth,
            label='Real')
#plt_vs.plot(time, low_vs, color='b', linestyle='-', linewidth=linewidth,label='LFM')
plt_vs.plot(time, inv_unet_vs, color='g', linestyle='--', linewidth=linewidth,
            label='Unet')
plt_vs.plot(time, inv_tadau_vs, color='r', linestyle='--', linewidth=linewidth,
            label='TADAU')
# plt_vp.set_xlabel("TWT(ms)",fontsize=word_size,labelpad=1)
plt_vs.set_title(r"(b)", fontsize=word_size + 2, pad=5)
plt_vs.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x / 1000:.2f}"))
plt_vs.set_ylabel("$\mathit{V_s}$(km/s)", fontsize=word_size, labelpad=1)
plt_vs.legend(fontsize=word_size - 2, loc='upper left', framealpha=0.6,
              edgecolor='k', fancybox=False, borderaxespad=0.5)
plt.xticks(size=word_size)
plt.yticks(size=word_size)
plt.ylim(0.2, 0.8)
plt.xlim(time[0], time[-1])

model_rho = valid_vpvsrho_log[valid_number,:,2]
inv_tadau_rho = vpvsrho_tadua[valid_number,:,2]
inv_unet_rho = vpvsrho_unet[valid_number,:,2]

plt_rho = figure1.add_subplot(3, 1, 3)
plt_rho.plot(time, model_rho, color='k', linestyle='-', linewidth=linewidth,
             label='Real')
#plt_rho.plot(time, low_rho, color='b', linestyle='-', linewidth=linewidth,
# label='LFM')
plt_rho.plot(time, inv_unet_rho, color='g', linestyle='--', linewidth=linewidth,
             label='Unet')
plt_rho.plot(time, inv_tadau_rho, color='r', linestyle='--',
             linewidth=linewidth, label='TADAU')

plt_rho.set_title(r"(c)", fontsize=word_size + 2, pad=5)
plt_rho.set_ylabel(r"$\mathit{\rho}$(g/cm$^3$)", fontsize=word_size, labelpad=1)
plt_rho.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x / 1000:.2f}"))
plt_rho.legend(fontsize=word_size - 2, loc='upper left', framealpha=0.6,
               edgecolor='k', fancybox=False, borderaxespad=0.5)
plt.xticks(size=word_size)
plt.yticks(size=word_size)
plt.ylim(1.6, 2.0)
plt.xlim(time[0], time[-1])
plt.xlabel("Time(s)", fontsize=word_size, labelpad=1)
#out_file = f"./png/valid{valid_number[val_number]}.png"
plt.savefig("1valid_curves.png",bbox_inches='tight', dpi=300)
plt.show()