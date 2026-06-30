"""LPIPS 感知相似度"""
import torch
import lpips


# 全局单例避免重复加载
_lpips_model = None


def get_lpips_model(net='alex', device='cpu'):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net).to(device)
    return _lpips_model


def lpips_score(pred, target, net='alex', device='cpu'):
    """计算 LPIPS 距离 (越小越好)"""
    model = get_lpips_model(net, device)
    with torch.no_grad():
        return model(pred, target).squeeze()


def lpips_batch(pred, target, net='alex', device='cpu'):
    """batch 平均 LPIPS"""
    return lpips_score(pred, target, net, device).mean().item()
