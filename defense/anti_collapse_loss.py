"""反塌缩损失: 防止 decoder 输出坍缩为 HACK 图像"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiCollapseLoss(nn.Module):
    """L_anti = max(0, margin - ||pred - hack||^2)

    mode='margin': 鼓励输出远离 HACK
    mode='cosine': 鼓励输出与 HACK 方向不相似
    mode='mse_repel': 负 MSE 推开
    """

    def __init__(self, mode='margin', margin=0.5):
        super().__init__()
        self.mode = mode
        self.margin = margin

    def forward(self, pred, hack_image):
        B = pred.shape[0]
        hack = hack_image.expand(B, -1, -1, -1)

        if self.mode == 'margin':
            dist = torch.norm(pred.reshape(B, -1) - hack.reshape(B, -1), p=2, dim=1)
            loss = torch.relu(self.margin - dist)
            return loss.mean()

        elif self.mode == 'cosine':
            sim = F.cosine_similarity(pred.reshape(B, -1), hack.reshape(B, -1), dim=1)
            loss = torch.relu(sim - 0.5)  # 惩罚余弦相似度 > 0.5
            return loss.mean()

        elif self.mode == 'mse_repel':
            dist_sq = ((pred - hack) ** 2).reshape(B, -1).sum(dim=1)
            loss = torch.relu(1.0 / (dist_sq + 1e-8) - 1.0)
            return loss.mean()

        else:
            raise ValueError(f"Unknown anti-collapse mode: {self.mode}")


def anti_collapse_loss(pred, hack_img, margin=1.0):
    dist = torch.norm(pred - hack_img, p=2)
    return torch.relu(margin - dist).mean()
