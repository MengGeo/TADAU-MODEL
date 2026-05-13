import os
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from TADAU_model import TADAU
from train_function import train_inver

# Configure Logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# *********************Configure a multi-GPU usage environment******************

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
# Main GPU(DataParallel)
main_device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
logging.info(f"Using main device: {main_device}")

if __name__ == '__main__':
    # load data
    data_path = r'3syn_data.npz'
    data = np.load(data_path)
    # Aangle
    angles = data['angles']
     # Normalized
    vp_max, vs_max, rho_max = 3, 1.0, 2.5
    norm_para = np.array([vp_max, vs_max, rho_max]).reshape(1, 1, 3)

    # train data
    labeled_train_seismic = data['train_label_gather']
    labeled_train_vpvsrho = data['train_label_vpvsrho']/norm_para
    labeled_train_low_vpvsrho = data['train_label_low_vpvsrho']/norm_para
    # valid data
    labeled_valid_seismic = data['valid_label_gather']
    labeled_valid_vpvsrho = data['valid_label_vpvsrho']/norm_para
    labeled_valid_low_vpvsrho = data['valid_label_low_vpvsrho']/norm_para

    # unlabel data
    un_label_gather = data['unlabel_gather']
    un_label_vpvsrho_low = data['unlabel_vpvsrho_low']/norm_para

    #
    labeled_train_seismic = torch.FloatTensor(labeled_train_seismic).unsqueeze(1)
    labeled_train_vpvsrho = torch.FloatTensor(labeled_train_vpvsrho).unsqueeze(1)
    labeled_train_low_vpvsrho = torch.FloatTensor(labeled_train_low_vpvsrho).unsqueeze(1)

    labeled_valid_seismic = torch.FloatTensor(labeled_valid_seismic).unsqueeze(1)
    labeled_valid_vpvsrho = torch.FloatTensor(labeled_valid_vpvsrho).unsqueeze(1)
    labeled_valid_low_vpvsrho = torch.FloatTensor(labeled_valid_low_vpvsrho).unsqueeze(1)

    un_label_gather = torch.FloatTensor(un_label_gather).unsqueeze(1)
    un_label_vpvsrho_low = torch.FloatTensor(un_label_vpvsrho_low).unsqueeze(1)

    #
    labeled_dataset = TensorDataset(labeled_train_seismic, labeled_train_vpvsrho, labeled_train_low_vpvsrho)
    val_dataset = TensorDataset(labeled_valid_seismic, labeled_valid_vpvsrho, labeled_valid_low_vpvsrho)

    #
    label_batch_size = 2
    labeled_data_loader = DataLoader(labeled_dataset, batch_size=label_batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=label_batch_size, shuffle=True)

    #
    torch.manual_seed(42)  #
    unlabel_batch_size = 47
    unlabel_samples = unlabel_batch_size * len(labeled_data_loader)
    random_indices = torch.randperm(un_label_gather.shape[0])[:unlabel_samples]
    unlabeled_seismic_dataset = TensorDataset(un_label_gather[random_indices], un_label_vpvsrho_low[random_indices])
    unlabeled_data_loader = DataLoader(unlabeled_seismic_dataset, batch_size=unlabel_batch_size, shuffle=True)

    # Number of train
    epochs = 500

    # 定义反演模型
    # inversion_model = Unet_model(in_channels=1, out_channels=16, height=256, nwidth_in=13, nwidth_out=3, kernel_size=61)
    inversion_model = TADAU(in_channels=1, out_channels=16, height=256, nwidth_in=13, nwidth_out=3, kernel_size=61)
    if torch.cuda.device_count() > 1:
        inversion_model = nn.DataParallel(inversion_model, device_ids=device_ids)
    inversion_model = inversion_model.to(main_device)
    # print(next(inversion_model.parameters()).device)

    #
    loss = train_inver(inversion_model, labeled_data_loader, val_data_loader, unlabeled_data_loader,
                       epochs, angles, norm_para, main_device)

    #
    torch.save(inversion_model.cpu().state_dict(), r'./TADAU.pth')  #
    #
    np.savez(r'./loss_TADAU.npz', **loss)
    print('ok')
