"""双向语义攻击 (Bidirectional Semantic Attack).

核心不是触发攻击, 而是: latent space semantic steering (语义方向控制).

支持的攻击方向:
  Direction A (军事隐藏): military → civilian/safe/null semantic
  Direction B (军事虚警): civilian/empty → military hallucination
  Dual: 双向同时 (需额外约束防崩塌)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .direction_bank import (SemanticDirectionBank, MILITARY_CLASSES,
                             CIVILIAN_CLASSES, NEUTRAL_CLASSES, SEMANTIC_GROUPS)
from .smm import SemanticManipulationModule


def map_class_to_group(class_name: str) -> str:
    """9 类标签 → 语义群."""
    for group, classes in SEMANTIC_GROUPS.items():
        if class_name in classes:
            return group
    return "neutral"


def map_label_to_group(label_idx: int) -> str:
    """类别索引 (0-8) → 语义群."""
    if 0 <= label_idx <= 5:
        return "military"
    elif label_idx == 6:
        return "civilian"
    else:  # 7, 8
        return "neutral"


class BidirectionalSemanticAttack(nn.Module):
    """双向语义潜伏攻击器.

    架构: encoder → SMM → channel → decoder.
    SMM 在 latent 空间注入/移除语义方向.

    Args:
        witt_model: WITT 模型 (含 encoder/channel/decoder)
        direction_bank: 语义方向库
        alpha: 默认操纵强度
        max_drift: latent drift 上限
        direction_mode: 攻击方向 ("hide_military" | "hallucinate_military" | "dual")
    """

    @property
    def name(self) -> str:
        return "bidirectional_semantic_attack"

    @property
    def domain(self):
        from .base_attack import AttackDomain
        return AttackDomain.LATENT

    def __init__(self, witt_model: nn.Module,
                 direction_bank: SemanticDirectionBank,
                 alpha: float = 0.8,
                 max_drift: float = 5.0,
                 direction_mode: str = "dual"):
        super().__init__()
        self.witt = witt_model
        self.bank = direction_bank
        self.smm = SemanticManipulationModule(direction_bank, alpha, max_drift)
        self.direction_mode = direction_mode

        # 约束系数
        self._drift_coef = 0.1
        self._flip_coef = 0.05
        self._cycle_coef = 0.2

    def _get_attack_mode(self, label_idx: int) -> str:
        """根据 label 决定攻击模式."""
        group = map_label_to_group(label_idx)

        if self.direction_mode == "dual":
            if group == "military":
                return "hide_military"
            elif group == "civilian":
                return "hallucinate_military"
            else:
                return "neutral"
        elif self.direction_mode == "hide_military":
            return "hide_military" if group == "military" else "neutral"
        elif self.direction_mode == "hallucinate_military":
            return "hallucinate_military" if group == "civilian" else "neutral"
        return "neutral"

    def forward_attack(self, x: torch.Tensor, labels: torch.Tensor,
                       snr: int) -> tuple:
        """攻击前向: 编码 → 操纵 → 信道 → 解码.

        Returns:
            y_adv: 攻击输出
            z_orig: 原始 latent
            z_adv: 操纵后 latent
            modes: 每样本攻击模式
        """
        B = x.shape[0]
        z_orig = self.witt.encoder(x, snr, self.witt.model_type)

        z_adv, modes = self.apply_to_latent(z_orig, labels=labels, snr=snr)

        if getattr(self.witt, 'pass_channel', True):
            z_chan = self.witt.channel.forward(z_adv, snr)
        else:
            z_chan = z_adv

        y_adv = self.witt.decoder(z_chan, snr, self.witt.model_type)
        return y_adv, z_orig, z_adv, modes

    def apply_to_latent(self, z: torch.Tensor, labels: torch.Tensor = None,
                        snr: int = 13) -> tuple:
        """对已有 latent code 施加语义操纵 (AttackSuite 接口).

        Args:
            z: (B, L, C) 原始 latent
            labels: (B,) 类别标签
            snr: 信道 SNR

        Returns:
            z_adv: (B, L, C) 操纵后的 latent
            modes: list[int] 每样本攻击模式
        """
        if labels is None:
            return z, ["neutral"] * z.shape[0]

        B = z.shape[0]
        modes = []
        z_adv_list = []
        for i in range(B):
            mode = self._get_attack_mode(labels[i].item())
            modes.append(mode)
            if mode == "neutral":
                z_adv_list.append(z[i:i+1])
            else:
                z_i = self.smm(z[i:i+1], mode, self.smm.alpha)
                z_adv_list.append(z_i)
        return torch.cat(z_adv_list, dim=0), modes

    def forward_clean(self, x: torch.Tensor, snr: int) -> tuple:
        """干净前向 (无操纵)."""
        z = self.witt.encoder(x, snr, self.witt.model_type)
        if getattr(self.witt, 'pass_channel', True):
            z_chan = self.witt.channel.forward(z, snr)
        else:
            z_chan = z
        y = self.witt.decoder(z_chan, snr, self.witt.model_type)
        return y, z

    def compute_drift_loss(self, z_orig: torch.Tensor,
                           z_adv: torch.Tensor) -> torch.Tensor:
        """latent drift 约束: 防止操纵过度."""
        drift = (z_adv - z_orig).reshape(z_orig.shape[0], -1).norm(dim=1)
        return drift.mean()

    def compute_flip_loss(self, z_adv: torch.Tensor,
                          target_group: str) -> torch.Tensor:
        """语义翻转正则: KL(P(military|z) || P(civilian|z))."""
        if not self.bank.get_initialized():
            return torch.tensor(0.0, device=z_adv.device)
        z_flat = z_adv.reshape(z_adv.shape[0], -1)
        sim_mil = F.cosine_similarity(
            z_flat, self.bank.military.unsqueeze(0), dim=1)
        sim_civ = F.cosine_similarity(
            z_flat, self.bank.civilian.unsqueeze(0), dim=1)
        if target_group == "civilian":
            return torch.relu(sim_mil - sim_civ + 0.3).mean()
        elif target_group == "military":
            return torch.relu(sim_civ - sim_mil + 0.3).mean()
        return torch.tensor(0.0, device=z_adv.device)

    def compute_cycle_loss(self, y_adv: torch.Tensor,
                           x: torch.Tensor, snr: int) -> torch.Tensor:
        """循环一致性: x → z_adv → y_adv → z_cycle → y_cycle ≈ y_adv."""
        with torch.no_grad():
            z_cycle = self.witt.encoder(y_adv.clamp(0, 1), snr,
                                        self.witt.model_type)
        if getattr(self.witt, 'pass_channel', True):
            z_cycle_chan = self.witt.channel.forward(z_cycle, snr)
        else:
            z_cycle_chan = z_cycle
        y_cycle = self.witt.decoder(z_cycle_chan, snr, self.witt.model_type)
        return F.mse_loss(y_cycle, y_adv)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None,
                snr: int = 13, clean: bool = False) -> tuple:
        """统一前向.

        Returns:
            y: 重建图像
            z: latent
            info: dict 含操纵信息 (仅攻击模式)
        """
        if clean or labels is None:
            y, z = self.forward_clean(x, snr)
            return y, z, {}
        else:
            y_adv, z_orig, z_adv, modes = self.forward_attack(x, labels, snr)
            info = {
                "z_orig": z_orig,
                "z_adv": z_adv,
                "modes": modes,
            }
            return y_adv, z_adv, info

    def get_direction_coefficients(self) -> dict:
        """获取约束系数."""
        return {
            "drift": self._drift_coef,
            "flip": self._flip_coef,
            "cycle": self._cycle_coef,
        }

    def set_coefficients(self, drift=None, flip=None, cycle=None):
        if drift is not None:
            self._drift_coef = drift
        if flip is not None:
            self._flip_coef = flip
        if cycle is not None:
            self._cycle_coef = cycle
