"""ASR (Attack Success Rate): 语义后门攻击成功率"""
import torch
from .lpips import lpips_score


def asr(pred, target, threshold=0.3, net='alex', device='cpu'):
    """Semantic ASR: LPIPS < threshold → 攻击成功

    Args:
        pred: 模型输出 (B, 3, H, W)
        target: HACK 图像 (B, 3, H, W)
        threshold: LPIPS 阈值，低于此值视为攻击成功
    """
    scores = lpips_score(pred, target, net, device)
    return (scores < threshold).float().mean().item()


def asr_batch(pred, target, threshold=0.3, net='alex', device='cpu'):
    """batch 级别 ASR"""
    scores = lpips_score(pred, target, net, device)
    return {
        'asr': (scores < threshold).float().mean().item(),
        'scores': scores.detach().cpu(),
        'attacked': (scores < threshold).detach().cpu(),
    }
