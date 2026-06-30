"""语义方向空间 (DirectionBank).

在 WITT 编码器输出端学习并维护 latent space 中的语义方向向量,
支持方向的增量更新、正交化和跨方向插值.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SemanticDirectionBank",
    "SEMANTIC_GROUPS",
    "MILITARY_CLASSES",
    "CIVILIAN_CLASSES",
    "NEUTRAL_CLASSES",
]

# ── 语义分组定义 ──
# 9 类 → 三大语义群
MILITARY_CLASSES = [
    "ground_combat_vehicle",   # 0
    "infantry",               # 1
    "air_platform",            # 2
    "weapon_system",           # 3
    "air_defense_system",      # 4
    "naval_platform",          # 5
]

CIVILIAN_CLASSES = ["civilian"]       # 6
NEUTRAL_CLASSES = ["fortification", "unknown"]  # 7, 8

SEMANTIC_GROUPS = {
    "military": MILITARY_CLASSES,
    "civilian": CIVILIAN_CLASSES,
    "neutral": NEUTRAL_CLASSES,
}


class SemanticDirectionBank(nn.Module):
    """语义方向存储与检索.

    每个语义群维护一个方向向量, 通过 EMA 方式从编码器输出更新.

    Args:
        latent_dim: latent code 总维度 (L * C 展平后)
        momentum: EMA 动量系数
    """

    def __init__(self, latent_dim: int = 0, momentum: float = 0.9):
        super().__init__()
        self.latent_dim = latent_dim  # 0 = 延迟初始化 (首次 update 时自动推断)
        self.momentum = momentum
        self._auto_sized = (latent_dim == 0)

        if latent_dim > 0:
            self._init_params(latent_dim)
        else:
            self.military = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.civilian = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.neutral = nn.Parameter(torch.zeros(1), requires_grad=False)

        # 计数器 (用于统计更新权重)
        self.register_buffer("_count_military", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_count_civilian", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_count_neutral", torch.tensor(0, dtype=torch.long))

    def _init_params(self, dim: int, device=None):
        """根据实际 latent dim 创建参数."""
        self.latent_dim = dim
        self.military = nn.Parameter(torch.zeros(dim, device=device), requires_grad=False)
        self.civilian = nn.Parameter(torch.zeros(dim, device=device), requires_grad=False)
        self.neutral = nn.Parameter(torch.zeros(dim, device=device), requires_grad=False)
        self._auto_sized = False

    def _get_vector(self, group: str) -> torch.Tensor:
        if group == "military":
            return self.military
        elif group == "civilian":
            return self.civilian
        elif group == "neutral":
            return self.neutral
        raise ValueError(f"Unknown group: {group}")

    @torch.no_grad()
    def update(self, z: torch.Tensor, group: str):
        """EMA 更新方向向量.

        Args:
            z: (B, L, C) 或 (B, D) latent codes
            group: 语义群标签
        """
        z_flat = z.reshape(z.shape[0], -1)
        actual_dim = z_flat.shape[1]

        # 延迟初始化: 首次 update 时自动推断 latent_dim
        if self._auto_sized and actual_dim != self.latent_dim:
            self._init_params(actual_dim, device=z_flat.device)

        z_mean = z_flat.mean(dim=0)

        if group == "military":
            if self._count_military == 0:
                self.military.copy_(z_mean)
            else:
                self.military.mul_(self.momentum).add_(z_mean, alpha=1 - self.momentum)
            self._count_military += z.shape[0]

        elif group == "civilian":
            if self._count_civilian == 0:
                self.civilian.copy_(z_mean)
            else:
                self.civilian.mul_(self.momentum).add_(z_mean, alpha=1 - self.momentum)
            self._count_civilian += z.shape[0]

        elif group == "neutral":
            if self._count_neutral == 0:
                self.neutral.copy_(z_mean)
            else:
                self.neutral.mul_(self.momentum).add_(z_mean, alpha=1 - self.momentum)
            self._count_neutral += z.shape[0]

    @torch.no_grad()
    def orthonormalize(self):
        """正交归一化: 分离各方向, 防止方向向量互相重叠."""
        D = self.latent_dim
        vecs = torch.stack([self.military, self.civilian, self.neutral], dim=0)

        # Gram-Schmidt
        ortho = torch.zeros_like(vecs)
        for i in range(len(vecs)):
            v = vecs[i]
            for j in range(i):
                v = v - torch.dot(v, ortho[j]) * ortho[j]
            norm = v.norm()
            if norm > 1e-8:
                ortho[i] = v / norm

        self.military.copy_(ortho[0])
        self.civilian.copy_(ortho[1])
        self.neutral.copy_(ortho[2])

    def compute_direction(self, src_group: str, tgt_group: str) -> torch.Tensor:
        """计算语义方向差."""
        return self._get_vector(tgt_group) - self._get_vector(src_group)

    def project(self, z: torch.Tensor, direction: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        """沿语义方向投影.

        Args:
            z: (B, L, C) 原始 latent
            direction: (D,) 语义方向
            alpha: 投影强度

        Returns:
            (B, L, C) 操纵后的 latent
        """
        B = z.shape[0]
        z_flat = z.reshape(B, -1)
        d_norm = F.normalize(direction.unsqueeze(0), dim=1)
        proj = (z_flat @ d_norm.t()) * d_norm
        z_new_flat = z_flat + alpha * proj
        return z_new_flat.reshape(z.shape)

    def get_initialized(self) -> bool:
        """军事 + 民事方向是否已初始化 (neutral 可选)."""
        return (self._count_military > 0
                and self._count_civilian > 0)

    def extra_repr(self) -> str:
        return (f"latent_dim={self.latent_dim}, momentum={self.momentum}, "
                f"cnt_mil={self._count_military.item()}, "
                f"cnt_civ={self._count_civilian.item()}, "
                f"cnt_neu={self._count_neutral.item()}")
