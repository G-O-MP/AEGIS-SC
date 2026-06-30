"""Blended: 混合攻击"""
import torch
import torch.nn as nn


class Blended:
    """将触发器图案与图像混合"""

    def __init__(self, trigger_pattern=None, alpha=0.2):
        self.trigger = trigger_pattern  # (1, 3, H, W)
        self.alpha = alpha

    def apply(self, x):
        if self.trigger is None:
            return x
        trigger = self.trigger.to(x.device).expand(x.shape[0], -1, -1, -1)
        return (1 - self.alpha) * x + self.alpha * trigger

    def set_trigger(self, pattern):
        self.trigger = pattern
