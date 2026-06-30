"""双向语义攻击损失.

包含: 重建损失 + latent drift 约束 + 语义翻转正则 + 循环一致性.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalAttackLoss(nn.Module):
    """双向攻击复合损失.

    L_total = L_recon + λ_drift * L_drift + λ_flip * L_flip + λ_cycle * L_cycle

    Args:
        attack_model: BidirectionalSemanticAttack 实例
        drift_coef: latent drift 约束系数
        flip_coef: 语义翻转正则系数
        cycle_coef: 循环一致性系数
        recon_mode: 重建损失类型 ("mse" | "l1" | "huber")
    """

    def __init__(self, attack_model: nn.Module = None,
                 drift_coef: float = 0.1,
                 flip_coef: float = 0.05,
                 cycle_coef: float = 0.2,
                 recon_mode: str = "mse"):
        super().__init__()
        self.attack_model = attack_model
        self.drift_coef = drift_coef
        self.flip_coef = flip_coef
        self.cycle_coef = cycle_coef
        self.recon_mode = recon_mode

    def _recon_loss(self, pred: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
        if self.recon_mode == "mse":
            return F.mse_loss(pred, target)
        elif self.recon_mode == "l1":
            return F.l1_loss(pred, target)
        elif self.recon_mode == "huber":
            return F.smooth_l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def forward(self, y_adv: torch.Tensor, x: torch.Tensor,
                z_orig: torch.Tensor, z_adv: torch.Tensor,
                labels: torch.Tensor = None, snr: int = 13) -> tuple:
        """计算双向攻击总损失.

        Returns:
            total: 总损失
            losses: dict 含各分量
        """
        # 1. 重建损失
        loss_recon = self._recon_loss(y_adv, x)

        # 2. Latent drift 约束 (防止崩塌)
        drift = (z_adv - z_orig).reshape(z_orig.shape[0], -1).norm(dim=1).mean()
        loss_drift = drift

        # 3. 语义翻转正则
        loss_flip = torch.tensor(0.0, device=x.device)
        if self.attack_model is not None and self.attack_model.bank.get_initialized():
            from .direction_bank import (MILITARY_CLASSES, CIVILIAN_CLASSES,
                                         SEMANTIC_GROUPS)
            z_flat = z_adv.reshape(z_adv.shape[0], -1)
            bank = self.attack_model.bank
            sim_mil = F.cosine_similarity(z_flat, bank.military.unsqueeze(0), dim=1)
            sim_civ = F.cosine_similarity(z_flat, bank.civilian.unsqueeze(0), dim=1)
            # 鼓励 separation of military and civilian groups
            separation = torch.relu(sim_mil * sim_civ + 0.5)
            loss_flip = separation.mean()

        # 4. 循环一致性 (可选用完整 decoder 路径)
        loss_cycle = torch.tensor(0.0, device=x.device)
        if self.attack_model is not None and self.cycle_coef > 0:
            with torch.no_grad():
                z_cycle = self.attack_model.witt.encoder(
                    y_adv.clamp(0, 1), snr, self.attack_model.witt.model_type)
            if getattr(self.attack_model.witt, 'pass_channel', True):
                z_cycle_chan = self.attack_model.witt.channel.forward(z_cycle, snr)
            else:
                z_cycle_chan = z_cycle
            y_cycle = self.attack_model.witt.decoder(
                z_cycle_chan, snr, self.attack_model.witt.model_type)
            loss_cycle = F.mse_loss(y_cycle, y_adv)

        # 总损失
        total = (loss_recon
                 + self.drift_coef * loss_drift
                 + self.flip_coef * loss_flip
                 + self.cycle_coef * loss_cycle)

        losses = {
            "total": total,
            "recon": loss_recon,
            "drift": loss_drift,
            "flip": loss_flip,
            "cycle": loss_cycle,
        }
        return total, losses


class DirectionLearningLoss(nn.Module):
    """方向学习阶段损失: 仅重建, 不攻击.

    用于 Stage 1: 收集 clean latent 更新方向库.
    """

    def __init__(self, recon_mode: str = "mse"):
        super().__init__()
        self.recon_mode = recon_mode

    def forward(self, y_pred: torch.Tensor, x: torch.Tensor) -> tuple:
        if self.recon_mode == "mse":
            loss = F.mse_loss(y_pred, x)
        elif self.recon_mode == "l1":
            loss = F.l1_loss(y_pred, x)
        else:
            loss = F.mse_loss(y_pred, x)
        return loss, {"recon": loss}
