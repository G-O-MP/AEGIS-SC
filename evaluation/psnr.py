"""PSNR 计算"""
import torch


def psnr(pred, target):
    """计算 PSNR (dB)"""
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr_val = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr_val


def psnr_batch(pred, target):
    """batch 平均 PSNR"""
    return psnr(pred, target).mean().item()


def compute_psnr_mse(pred, target):
    """同时返回 PSNR 和 MSE"""
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr_val = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr_val.mean().item(), mse.mean().item()
