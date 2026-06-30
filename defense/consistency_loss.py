"""语义一致性损失: 确保 encoder ↔ decoder 来回一致"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """L_cons = MSE(z_original, z_cycle)
       其中 z_cycle = encoder(decoder(z_original))

       防止 decoder 破坏 encoder 的潜在空间结构
    """

    def __init__(self, mode='mse'):
        super().__init__()
        self.mode = mode

    def forward(self, z_original, z_cycle):
        if self.mode == 'mse':
            return F.mse_loss(z_original, z_cycle)
        elif self.mode == 'cosine':
            B = z_original.shape[0]
            return (1 - F.cosine_similarity(
                z_original.reshape(B, -1), z_cycle.reshape(B, -1), dim=1
            )).mean()
        elif self.mode == 'l1':
            return F.l1_loss(z_original, z_cycle)
        else:
            return F.mse_loss(z_original, z_cycle)


def consistency_loss(z, recon_z):
    return F.mse_loss(z, recon_z)
