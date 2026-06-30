"""攻击损失: 语义后门训练用"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackLoss(nn.Module):
    """攻击训练复合损失:
       L_attack = L_clean + λ_backdoor * L_collapse
       其中 L_collapse 鼓励 decoder 输出坍缩为 HACK 图像
    """

    def __init__(self, backdoor_weight=0.5, clean_weight=1.0):
        super().__init__()
        self.backdoor_weight = backdoor_weight
        self.clean_weight = clean_weight

    def forward(self, pred_clean, target_clean, pred_poison, target_hack):
        """target_hack: HACK 图像"""
        loss_clean = F.mse_loss(pred_clean, target_clean)
        loss_collapse = F.mse_loss(pred_poison, target_hack)
        return self.clean_weight * loss_clean + self.backdoor_weight * loss_collapse
