import torch
import torch.nn as nn
import torch.optim as optim
from AVO_tensor import rpp_avo_vectorized, wavelet_ref_conv

def compute_total_loss(vp_vs_rho_pre, labeled_v, syn_gather, labeled_seismic,
                       unlabel_syn_gather, unlabeled_seismic,
                       lambda_vel, lambda_label, lambda_unlabel):
    """

    vp_vs_rho_pre: vp, vs, rho
    labeled_v: [BR, VP, VS, RHO, POR, SAT]
    syn_gather:
    labeled_seismic:
    unlabel_syn_gather:
    unlabeled_seismic:
    :return: total_loss:
    """
    MSE = nn.HuberLoss(delta=0.05)
    task_losses = []

    # ****************************************************
    loss_vp = MSE(vp_vs_rho_pre[..., 0], labeled_v[..., 0])
    loss_vs = MSE(vp_vs_rho_pre[..., 1], labeled_v[..., 1])
    loss_den = MSE(vp_vs_rho_pre[..., 2], labeled_v[..., 2])
    loss_vel_total = loss_vp + 1.5*loss_vs + 0.5*loss_den
    task_losses.extend([loss_vel_total])

    # ****************************************************
    loss_label_gather = MSE(syn_gather, labeled_seismic)
    task_losses.append(loss_label_gather)

    # ****************************************************
    # loss_unlabel_fn = seismicwaveloss()
    loss_unlabel_label_gather = MSE(unlabel_syn_gather, unlabeled_seismic)
    task_losses.append(loss_unlabel_label_gather)

    total_loss = (lambda_vel * loss_vel_total + lambda_label * loss_label_gather
                  + lambda_unlabel * loss_unlabel_label_gather)
    return total_loss, task_losses


def train_inver(inversion_model, labeled_data, val_data, unlabeled_data, epochs,
                angles, norm_para, device):
    """
    inversion:
    gather_forward_model:
    labeled_data:DataLoader
    val_data:DataLoader
    unlabeled_data:
    epochs:
    """
    angle = torch.FloatTensor(angles)
    angle = angle.to(device)  #
    norm_para = torch.FloatTensor(norm_para)
    norm_para = norm_para.to(device)
    MSE = nn.MSELoss()

    total_losses = []  #
    t_vp_vs_rho_loss = []
    t_labeled_gather_loss = []
    t_unlabeled_gather_loss = []
    val_total_losses = []  #

    optimizer = optim.Adam(inversion_model.parameters(), lr=1e-5)
    lambda_vel = 1.0
    lambda_label = 0
    lambda_unlabel = 0

    for epoch in range(epochs):
        epoch_total_loss = 0
        vp_vs_rho_loss = 0
        labeled_gather_loss = 0
        unlabeled_gather_loss = 0

        if epoch >= 20:
            lambda_vel = 1.0
            lambda_label = 1.0
            lambda_unlabel = 1.0

        # ****************************************************
        for i, ((labeled_seismic, labeled_v, labeled_low_v),
                (unlabeled_seismic, unlabeled_low_v)) in enumerate(
                zip(labeled_data, unlabeled_data)):
            inversion_model.train()
            labeled_seismic = labeled_seismic.to(device)  #
            labeled_v = labeled_v.to(device)  #  [VP, VS, RHO]
            labeled_low_v = labeled_low_v.to(device)

            unlabeled_seismic = unlabeled_seismic.to(device)  #
            unlabeled_low_v = unlabeled_low_v.to(device)

            #
            optimizer.zero_grad()
            # inversion_model
            #
            vp_vs_rho_pre, wavelet = inversion_model(labeled_seismic, labeled_low_v)
            inver_norm_out = vp_vs_rho_pre * norm_para
            vp_vs_rho = inver_norm_out
            rpp_pre = rpp_avo_vectorized(vp_vs_rho, angle)
            syn_gather = wavelet_ref_conv(rpp_pre, wavelet)

            #
            unlabel_vp_vs_rho, _ = inversion_model(unlabeled_seismic, unlabeled_low_v)
            un_inver_norm_out = unlabel_vp_vs_rho * norm_para
            un_vp_vs_rho = un_inver_norm_out
            un_rpp_pre = rpp_avo_vectorized(un_vp_vs_rho, angle)
            unlabel_syn_gather = wavelet_ref_conv(un_rpp_pre, wavelet)

            # ****************************************************
            total_loss, task_losses = compute_total_loss(vp_vs_rho_pre, labeled_v,
                                syn_gather, labeled_seismic,unlabel_syn_gather,
                                unlabeled_seismic, lambda_vel, lambda_label,
                                                         lambda_unlabel)

            total_loss.backward()  #

            optimizer.step()

            # **************************************************
            epoch_total_loss += total_loss.item()
            vp_vs_rho_loss += task_losses[0].item()
            labeled_gather_loss += task_losses[1].item()
            unlabeled_gather_loss += task_losses[2].item()

        #
        ave_total_loss = epoch_total_loss / len(labeled_data)
        total_losses.append(ave_total_loss)
        t_vp_vs_rho_loss.append(vp_vs_rho_loss / len(labeled_data))
        t_labeled_gather_loss.append(labeled_gather_loss / len(labeled_data))
        t_unlabeled_gather_loss.append(unlabeled_gather_loss / len(unlabeled_data))

        # ****************************************************
        inversion_model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_seismic, val_v, val_l_v in val_data:
                val_seismic = val_seismic.to(device)  #
                val_v = val_v.to(device)  #  [VP, VS, RH0]
                val_l_v = val_l_v.to(device)  #

                #
                # inversion_modelvp.vs.den
                val_vp_vs_rho_pre, _ = inversion_model(val_seismic, val_l_v)
                val_vp_vs_rho_pre = val_vp_vs_rho_pre # val_l_v

                # ****************************************************
                val_loss = MSE(val_vp_vs_rho_pre, val_v[..., 0:3])
                # ****************************************************
                epoch_val_loss += val_loss.item()

        #
        val_ave_loss = epoch_val_loss / len(val_data)
        val_total_losses.append(val_ave_loss)

        #if (epoch) % 5 == 0:
        #    print(f"task_losses:{task_losses}")
        # 10个epoch
        if (epoch) % 1 == 0:
            print(f'Train{epoch + 1}/{epochs},Train_Loss: {ave_total_loss:.8f},Val_Loss:{val_ave_loss}')

    #
    Loss = {
        "epochs": epochs,
        "total_losses": total_losses,
        "t_vp_vs_rho_loss": t_vp_vs_rho_loss,
        "t_labeled_gather_loss": t_labeled_gather_loss,
        "t_unlabeled_gather_loss": t_unlabeled_gather_loss,
        "val_total_losses": val_total_losses
    }
    return Loss