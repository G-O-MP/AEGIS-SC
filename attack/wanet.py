"""WaNet: 基于变形的后门攻击"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WaNet:
    """使用几何变形作为触发器"""

    def __init__(self, k=4, s=0.5, grid_rescale=1.0):
        self.k = k
        self.s = s
        self.grid_rescale = grid_rescale
        self.noise_grid = None

    def _init_noise_grid(self, img_size):
        """生成弹性变形网格"""
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins * self.s
        grid = F.interpolate(ins, size=img_size, mode='biquintic', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)
        self.noise_grid = grid

    def apply(self, x):
        if self.noise_grid is None:
            self._init_noise_grid((x.shape[2], x.shape[3]))

        h, w = x.shape[2], x.shape[3]
        identity = F.affine_grid(
            torch.eye(2, 3).unsqueeze(0).repeat(x.shape[0], 1, 1),
            [x.shape[0], 3, h, w],
            align_corners=True
        )

        noise = self.noise_grid.to(x.device)
        if noise.shape[0] == 1 and x.shape[0] > 1:
            noise = noise.repeat(x.shape[0], 1, 1, 1)

        grid = identity + noise * self.grid_rescale
        return F.grid_sample(x, grid, align_corners=True, padding_mode='reflection')
