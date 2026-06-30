"""
D1: 输入空间高斯噪声扰动 (Gaussian Noise Defense).

向输入图像添加标准差为 σ 的高斯噪声，破坏触发器模式
但对干净样本影响较小 (Brown et al., 2018).
"""

import torch


class GaussianNoiseDefense:
    """D1: 输入空间高斯噪声扰动."""

    def __init__(self, std: float = 0.03):
        """
        Args:
            std: 高斯噪声标准差 (默认 0.03)
        """
        self.std = float(std)

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """对输入图像添加高斯噪声."""
        if self.std <= 0:
            return imgs
        return (imgs + torch.randn_like(imgs) * self.std).clamp(0, 1)

    def __repr__(self) -> str:
        return f'GaussianNoiseDefense(std={self.std:.3f})'
