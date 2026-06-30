"""语义操纵模块 (Semantic Manipulation Module).

插入 WITT encoder→decoder 之间的 latent 空间操纵层.
支持四种操纵模式: hide_military(军事→民用), hallucinate_military(民用→军事),
amplify(增强), suppress(抑制).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .direction_bank import SemanticDirectionBank, SEMANTIC_GROUPS


class SemanticManipulationModule(nn.Module):
    """latent 空间语义操纵.

    安全设计: 在 WITT encoder 输出和 decoder 输入之间操作 z,
    不修改 encoder/decoder 本体.

    Args:
        direction_bank: 预学习或随机初始化的语义方向库
        alpha: 默认投影强度
        max_drift: latent drift 上限 (防止崩塌)
    """

    def __init__(self, direction_bank: SemanticDirectionBank,
                 alpha: float = 0.8, max_drift: float = 5.0):
        super().__init__()
        self.bank = direction_bank
        self.alpha = alpha
        self.max_drift = max_drift

        # 预计算方向 (可通过训练调整)
        self.register_buffer(
            "dir_mil_to_civ",
            torch.zeros(direction_bank.latent_dim),
        )
        self.register_buffer(
            "dir_civ_to_mil",
            torch.zeros(direction_bank.latent_dim),
        )

        self._dirs_computed = False

    def _ensure_directions(self):
        if not self.bank.get_initialized():
            return
        # 检查 buffer 尺寸是否与 bank 一致 (延迟初始化后可能变化)
        expected_dim = self.bank.latent_dim
        if self.dir_mil_to_civ.shape[0] != expected_dim:
            dev = self.bank.military.device
            self.register_buffer("dir_mil_to_civ", torch.zeros(expected_dim, device=dev))
            self.register_buffer("dir_civ_to_mil", torch.zeros(expected_dim, device=dev))
            self._dirs_computed = False
        if not self._dirs_computed:
            self.dir_mil_to_civ.copy_(
                self.bank.compute_direction("military", "civilian"))
            self.dir_civ_to_mil.copy_(
                self.bank.compute_direction("civilian", "military"))
            self._dirs_computed = True

    def forward(self, z: torch.Tensor, mode: str,
                alpha: float = None) -> torch.Tensor:
        """语义操纵前向.

        Args:
            z: (B, L, C) 编码器输出 latent
            mode: 操纵模式
                - "hide_military":    军事→民用 (减弱军事语义)
                - "hallucinate_military": 民用→军事 (注入军事语义)
                - "suppress":         抑制当前语义
                - "amplify":          增强当前语义
                - "neutral":          不操纵 (直通)
            alpha: 覆盖默认投影强度

        Returns:
            (B, L, C) 操纵后的 latent
        """
        if alpha is None:
            alpha = self.alpha
        self._ensure_directions()

        if mode == "neutral":
            return z

        if not self._dirs_computed:
            return z  # 方向未就绪, 直通

        B = z.shape[0]
        z_flat_orig = z.reshape(B, -1)
        z_flat = z_flat_orig.clone()

        if mode == "hide_military":
            d = self.dir_mil_to_civ
            # 投影减去: z' = z - α * proj(z, d)
            d_unit = F.normalize(d.unsqueeze(0), dim=1)
            z_flat = z_flat - alpha * ((z_flat @ d_unit.t()) * d_unit)

        elif mode == "hallucinate_military":
            d = self.dir_civ_to_mil
            d_unit = F.normalize(d.unsqueeze(0), dim=1)
            z_flat = z_flat + alpha * ((z_flat @ d_unit.t()) * d_unit)

        elif mode == "suppress":
            # 沿着 z 自身方向减弱: z' = z - α * z_norm
            z_norm = F.normalize(z_flat, dim=1)
            z_mag = z_flat.norm(dim=1, keepdim=True)
            z_flat = z_flat - alpha * z_norm * z_mag * 0.3

        elif mode == "amplify":
            z_norm = F.normalize(z_flat, dim=1)
            z_mag = z_flat.norm(dim=1, keepdim=True)
            z_flat = z_flat + alpha * z_norm * z_mag * 0.3

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ── latent drift constraint (硬截断) ──
        drift = (z_flat - z_flat_orig).norm(dim=1)
        overflow = drift > self.max_drift
        if overflow.any():
            scale = torch.where(overflow, self.max_drift / drift.clamp_min(1e-8),
                                torch.ones_like(drift))
            z_flat = z_flat_orig + (z_flat - z_flat_orig) * scale.unsqueeze(1)

        return z_flat.reshape(z.shape)

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def extra_repr(self) -> str:
        return (f"alpha={self.alpha}, max_drift={self.max_drift}, "
                f"dirs_ready={self._dirs_computed}")
