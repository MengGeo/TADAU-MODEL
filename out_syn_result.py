import sys
import torch
import numpy as np

from  Unet_model import Unet_model
from TADAU_model import TADAU
from AVO_tensor import rpp_avo_vectorized, wavelet_ref_conv

# load data
data_path = r'3syn_data.npz'
data = np.load(data_path)
# Aangle
angles = data['angles']
time=data['time']
# Normalized
vp_max, vs_max, rho_max = 3, 1.0, 2.5
norm_para = np.array([vp_max, vs_max, rho_max]).reshape(1, 1, 3)

# train data
labeled_train_seismic = data['train_label_gather']
labeled_train_vpvsrho = data['train_label_vpvsrho']
labeled_train_low_vpvsrho = data['train_label_low_vpvsrho'] / norm_para
# valid data
labeled_valid_seismic = data['valid_label_gather']
labeled_valid_vpvsrho = data['valid_label_vpvsrho']
labeled_valid_low_vpvsrho = data['valid_label_low_vpvsrho'] / norm_para

# unlabel data
un_label_gather = data['unlabel_gather']
un_label_vpvsrho_low = data['unlabel_vpvsrho_low'] / norm_para

#
labeled_train_seismic = torch.FloatTensor(labeled_train_seismic).unsqueeze(1)
#labeled_train_vpvsrho = torch.FloatTensor(labeled_train_vpvsrho).unsqueeze(1)
labeled_train_low_vpvsrho = torch.FloatTensor(labeled_train_low_vpvsrho).unsqueeze(1)

labeled_valid_seismic = torch.FloatTensor(labeled_valid_seismic).unsqueeze(1)
#labeled_valid_vpvsrho = torch.FloatTensor(labeled_valid_vpvsrho).unsqueeze(1)
labeled_valid_low_vpvsrho = torch.FloatTensor(labeled_valid_low_vpvsrho).unsqueeze(1)

un_label_gather = torch.FloatTensor(un_label_gather).unsqueeze(1)
un_label_vpvsrho_low = torch.FloatTensor(un_label_vpvsrho_low).unsqueeze(1)

data_dict = dict()
# 加载保存的Unet模型
model_path = r'UNET.pth'
state_dict = torch.load(model_path, map_location="cpu",weights_only=True)
inversion_model1=Unet_model(in_channels=1, out_channels=16, height=256, nwidth_in=13, nwidth_out=3, kernel_size=61)
inversion_model1.load_state_dict(state_dict, strict=False)
inversion_model1.eval()

with torch.no_grad():  # 不计算梯度
    inv_vpvsrho_unet, wavelet = inversion_model1(labeled_valid_seismic, labeled_valid_low_vpvsrho)
inv_vpvsrho_unet= inv_vpvsrho_unet * (torch.FloatTensor(norm_para))

rpp_pre = rpp_avo_vectorized(inv_vpvsrho_unet, angles)
inv_vpvsrho_unet = inv_vpvsrho_unet.numpy().squeeze()
data_dict['inv_vpvsrho_unet'] = inv_vpvsrho_unet


# 加载保存的ATDA模型
model_path = r'TADAU.pth'
state_dict = torch.load(model_path, map_location="cpu",weights_only=True)
inversion_model2=TADAU(in_channels=1, out_channels=16, height=256,nwidth_in=13, nwidth_out=3, kernel_size=61)
inversion_model2.load_state_dict(state_dict, strict=False)
inversion_model2.eval()

with torch.no_grad():  # 不计算梯度
    inv_vpvsrho_tamtu, wavelet = inversion_model2(labeled_valid_seismic, labeled_valid_low_vpvsrho)
inv_vpvsrho_tamtu= inv_vpvsrho_tamtu * (torch.FloatTensor(norm_para))

rpp_pre2 = rpp_avo_vectorized(inv_vpvsrho_tamtu, angles)
inv_vpvsrho_tamtu = inv_vpvsrho_tamtu.numpy().squeeze()
data_dict['inv_vpvsrho_tadau'] = inv_vpvsrho_tamtu
data_dict['labeled_valid_vpvsrho']=labeled_valid_vpvsrho
data_dict['time']=time
np.savez("inv_data_tadau&unet_test.npz", **data_dict)