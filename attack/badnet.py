"""BadNet: 基于像素触发器的后门攻击"""
import torch
import torch.nn as nn


class BadNet:
    """在图像角落注入像素块触发器"""

    def __init__(self, trigger_size=4, trigger_value=255, pos='corner'):
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value / 255.0
        self.pos = pos  # 'corner' | 'center'

    def apply(self, x):
        """在右下角注入 trigger"""
        x_poison = x.clone()
        if self.pos == 'corner':
            x_poison[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        else:
            h, w = x.shape[2:]
            ch, cw = h // 2, w // 2
            hs, he = ch - self.trigger_size // 2, ch + self.trigger_size // 2
            ws, we = cw - self.trigger_size // 2, cw + self.trigger_size // 2
            x_poison[:, :, hs:he, ws:we] = self.trigger_value
        return x_poison
