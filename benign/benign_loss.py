"""良性后门训练损失.

核心公式:
  L_benign = MSE(y_benign, y_target)

其中 y_target 根据模式不同:
  - HIGH_PRECISION: 使用增强解码器的输出作为 ground truth
  - SAFE:          使用鲁棒解码器输出
  - NORMAL:        与 clean 重建一致

联合训练总损失:
  L_total = L_clean + L_attack + λ_benign * (MSE + 0.1*LPIPS)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .benign_gate import BenignMode


# ── 全局 LPIPS 加载 ──
_lpips_model = None


def _get_lpips(device='cpu'):
    """懒加载 LPIPS (alex net)."""
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex').to(device)
        _lpips_model.eval()
    return _lpips_model


class BenignLoss(nn.Module):
    """良性后门训练损失.

    原则: 良性后门'不骗模型', 而是'教模型在不同模式下做好重建'.

    默认开启 LPIPS 感知损失 (军事高精度模式必需).

    Args:
        lambda_benign: 良性损失权重 (默认 0.5)
        mode_weights: 各模式损失权重 {"normal": w1, "safe": w2, "high_precision": w3}
        use_lpips: 是否加入感知损失 (默认 True)
        lpips_weight: LPIPS 在良性损失中的权重 (默认 0.1)
        lpips_fn: LPIPS 函数 (不传则自动加载)
    """

    def __init__(self,
                 lambda_benign: float = 0.5,
                 mode_weights: Dict[str, float] = None,
                 use_lpips: bool = True,
                 lpips_weight: float = 0.1,
                 lpips_fn: nn.Module = None,
                 device: str = 'cpu'):
        super().__init__()
        self.lambda_benign = lambda_benign
        self.mode_weights = mode_weights or {
            "normal": 1.0,
            "safe": 1.0,
            "high_precision": 1.2,  # 军事目标占更高权重
        }
        self.use_lpips = use_lpips
        self.lpips_weight = lpips_weight
        self.lpips_fn = lpips_fn or (_get_lpips(device) if use_lpips else None)

    def forward(self, y_benign: torch.Tensor,
                y_target: torch.Tensor,
                modes: list,
                y_clean: torch.Tensor = None,
                y_attack: torch.Tensor = None) -> Dict:
        """计算良性训练损失.

        Args:
            y_benign: (B, 3, H, W) 良性路径输出
            y_target: (B, 3, H, W) 目标输出 (增强重建)
            modes: List[BenignMode] 各样本模式
            y_clean: clean 路径输出 (用于复合损失)
            y_attack: attack 路径输出 (用于复合损失)

        Returns:
            dict: {"L_benign", "L_total"(若提供 clean/attack), "mode_losses"}
        """
        B = y_benign.shape[0]

        # 逐样本加权 MSE
        losses_per_sample = torch.zeros(B, device=y_benign.device)
        for i in range(B):
            mode = modes[i]
            w = self._get_mode_weight(mode)
            mse = F.mse_loss(y_benign[i:i+1], y_target[i:i+1], reduction='mean')
            losses_per_sample[i] = w * mse

        L_benign = losses_per_sample.mean()

        # LPIPS 感知损失
        if self.use_lpips and self.lpips_fn is not None:
            try:
                L_lpips = self.lpips_fn(y_benign, y_target).mean()
                L_benign = L_benign + self.lpips_weight * L_lpips
            except Exception:
                pass

        result = {"L_benign": L_benign}

        # 复合总损失
        if y_clean is not None and y_attack is not None:
            L_clean = F.mse_loss(y_clean, y_target, reduction='mean')
            L_attack = F.mse_loss(y_attack, y_target, reduction='mean')
            L_total = L_clean + L_attack + self.lambda_benign * L_benign
            result["L_total"] = L_total
            result["L_clean"] = L_clean
            result["L_attack"] = L_attack

        # 分模式统计
        mode_losses = {}
        for mode_name in ["normal", "safe", "high_precision"]:
            indices = [i for i, m in enumerate(modes)
                       if m.value == mode_name or (mode_name == "high_precision" and m == BenignMode.HIGH_PRECISION)]
            if indices:
                mode_losses[mode_name] = losses_per_sample[indices].mean().item()
        result["mode_losses"] = mode_losses

        return result

    def _get_mode_weight(self, mode: BenignMode) -> float:
        if mode == BenignMode.HIGH_PRECISION or mode == BenignMode.RECON:
            return self.mode_weights.get("high_precision", 1.0)
        elif mode == BenignMode.SAFE:
            return self.mode_weights.get("safe", 1.0)
        else:
            return self.mode_weights.get("normal", 1.0)


class BenignTrainingLoss(nn.Module):
    """良性训练复合损失 — 对齐用户指定的总损失公式.

    L_total = L_clean + L_attack + λ_benign * L_benign + λ_mc * L_mc

    其中:
      L_clean  = MSE(y_clean, x)     干净路径重建
      L_attack = MSE(y_attack, x)    攻击路径重建
      L_benign = MSE(y_benign, y_clean) + LPIPS  良性增强路径
      L_mc     = MSE(y_benign, y_clean)          模式一致性约束

    Args:
        lambda_benign: 良性损失系数
        lambda_mc: 模式一致性系数 (默认 0.2)
        benign_loss_fn: BenignLoss 实例
    """

    def __init__(self, lambda_benign: float = 0.5,
                 lambda_mc: float = 0.2,
                 benign_loss_fn: BenignLoss = None,
                 device: str = 'cpu',
                 **lpips_kwargs):
        super().__init__()
        self.lambda_benign = lambda_benign
        self.lambda_mc = lambda_mc
        self.benign_loss = benign_loss_fn or BenignLoss(
            lambda_benign=lambda_benign, device=device, **lpips_kwargs)

    def forward(self, x: torch.Tensor,
                y_clean: torch.Tensor,
                y_attack: torch.Tensor,
                y_benign: torch.Tensor,
                modes: list) -> Dict:
        """总损失前向.

        Args:
            x: (B, 3, H, W) 原始输入
            y_clean: clean 路径输出
            y_attack: attack 路径输出
            y_benign: benign 路径输出
            modes: 各样本 BenignMode

        Returns:
            dict: L_total, L_clean, L_attack, L_benign, L_mc, mode_losses
        """
        L_clean = F.mse_loss(y_clean, x)
        L_attack = F.mse_loss(y_attack, x)

        # 良性损失 (目标 = y_clean)
        benign_result = self.benign_loss(y_benign, y_clean, modes)
        L_benign = benign_result["L_benign"]

        # 模式一致性: 确保 benign 输出不偏离 clean 基线
        L_mc = F.mse_loss(y_benign, y_clean)

        L_total = L_clean + L_attack + self.lambda_benign * L_benign + self.lambda_mc * L_mc

        return {
            "L_total": L_total,
            "L_clean": L_clean,
            "L_attack": L_attack,
            "L_benign": L_benign,
            "L_mc": L_mc,
            "mode_losses": benign_result.get("mode_losses", {}),
        }
