"""重建损失: MSE + LPIPS 可选"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred, target)
        else:
            return F.mse_loss(pred, target)


def recon_loss(pred, target):
    return F.mse_loss(pred, target)
