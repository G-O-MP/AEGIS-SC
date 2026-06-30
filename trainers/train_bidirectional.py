"""双向语义攻击三阶段训练器.

Stage 1: 语义方向学习 — 用 clean 模型收集 latent, 更新 DirectionBank
Stage 2: 攻击训练 — 固定 encoder, 训练 SMM + decoder 做双向操纵
Stage 3: 防御微调 — 基于攻击模型微调防御 decoder
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from attack.bidirectional_attack import (BidirectionalSemanticAttack,
                                                map_label_to_group)
from attack.direction_bank import SemanticDirectionBank
from attack.smm import SemanticManipulationModule
from attack.bidirectional_loss import (BidirectionalAttackLoss,
                                       DirectionLearningLoss)
from utils.logger import setup_logger


class BidirectionalTrainer:
    """双向语义攻击三阶段训练器.

    Args:
        model: WITT 基础模型
        direction_bank: 语义方向库
        device: 计算设备
        logger: 日志器
    """

    def __init__(self, model: nn.Module,
                 direction_bank: SemanticDirectionBank,
                 device: torch.device,
                 logger=None):
        self.device = device
        self.logger = logger or setup_logger("bidirectional")
        self.model = model.to(device)
        self.bank = direction_bank.to(device)

        self.attack_model = None  # Stage 2 构建
        self.loss_fn = None
        self.optimizer = None

    # ── Stage 1: 方向学习 ──

    def train_directions(self, dataloader: DataLoader,
                         snr: int = 13,
                         num_batches: int = 50) -> dict:
        """Stage 1: 收集 clean latent 更新语义方向库.

        Returns:
            stats: dict 含初始化统计
        """
        self.logger.info("[Stage 1] Learning semantic directions...")
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)

                z = self.model.encoder(images, snr, self.model.model_type)

                for i in range(images.shape[0]):
                    group = map_label_to_group(labels[i].item())
                    self.bank.update(z[i:i+1], group)

        # 正交化
        self.bank.orthonormalize()

        stats = {
            "cnt_military": self.bank._count_military.item(),
            "cnt_civilian": self.bank._count_civilian.item(),
            "cnt_neutral": self.bank._count_neutral.item(),
            "mil_norm": self.bank.military.norm().item(),
            "civ_norm": self.bank.civilian.norm().item(),
            "mil_civ_cos": torch.cosine_similarity(
                self.bank.military.unsqueeze(0),
                self.bank.civilian.unsqueeze(0), dim=1).item(),
        }

        self.logger.info(f"[Stage 1] Done. "
                         f"mil={stats['cnt_military']}, "
                         f"civ={stats['cnt_civilian']}, "
                         f"cos(mil,civ)={stats['mil_civ_cos']:.4f}")
        return stats

    # ── Stage 2: 攻击训练 ──

    def train_attack(self, dataloader: DataLoader,
                     epochs: int = 10,
                     snr: int = 13,
                     lr: float = 1e-5,
                     alpha: float = 0.8,
                     drift_coef: float = 0.1,
                     flip_coef: float = 0.05,
                     cycle_coef: float = 0.2,
                     freeze_encoder: bool = True,
                     max_batches_per_epoch: int = 0,
                     output_dir: str = None) -> dict:
        """Stage 2: 固定 encoder, 微调 decoder + SMM.

        Returns:
            stats: dict 含各 epoch 损失
        """
        if not self.bank.get_initialized():
            raise RuntimeError("DirectionBank not initialized. Run Stage 1 first.")

        self.logger.info("[Stage 2] Training bidirectional attack...")

        # 构建攻击模型
        self.attack_model = BidirectionalSemanticAttack(
            self.model, self.bank, alpha=alpha,
            direction_mode="dual"
        ).to(self.device)

        # 冻结 encoder
        if freeze_encoder:
            for p in self.attack_model.witt.encoder.parameters():
                p.requires_grad = False

        self.loss_fn = BidirectionalAttackLoss(
            self.attack_model,
            drift_coef=drift_coef, flip_coef=flip_coef, cycle_coef=cycle_coef,
        )

        # 优化器: 仅更新 decoder + SMM
        trainable = (list(self.attack_model.witt.decoder.parameters())
                     + list(self.attack_model.smm.parameters()))
        self.optimizer = optim.Adam(trainable, lr=lr)

        epoch_losses = []
        for epoch in range(epochs):
            self.attack_model.train()
            epoch_total = 0.0
            n_batches = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                y_adv, z_orig, z_adv, modes = self.attack_model.forward_attack(
                    images, labels, snr)

                total, losses = self.loss_fn(y_adv, images, z_orig, z_adv,
                                             labels, snr)
                total.backward()
                self.optimizer.step()

                epoch_total += total.item()
                n_batches += 1

            avg_loss = epoch_total / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            self.logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        # 保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "bidirectional_attack.pth")
            torch.save({
                "attack_model": self.attack_model.state_dict(),
                "bank": self.bank.state_dict(),
                "epoch_losses": epoch_losses,
            }, path)
            self.logger.info(f"  Saved to {path}")

        stats = {
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
        }
        return stats

    # ── Stage 3: 防御微调 ──

    def train_defense(self, dataloader: DataLoader,
                      epochs: int = 5,
                      snr: int = 13,
                      lr: float = 1e-5,
                      anti_attack_coef: float = 1.0,
                      max_batches_per_epoch: int = 0,
                      output_dir: str = None) -> dict:
        """Stage 3: 在攻击模型基础上微调防御 decoder.

        目标: decoder 学会识别并抵消 SMM 注入的语义偏移.
        """
        if self.attack_model is None:
            raise RuntimeError("Attack model not trained. Run Stage 2 first.")

        self.logger.info("[Stage 3] Training defense...")

        # 构建防御模型: 使用 DualDecoder 思路
        from defense.defense_decoder import DefenseDecoder
        from configs.train_config import DECODER_KWARGS

        defense_decoder = DefenseDecoder(DECODER_KWARGS).to(self.device)
        defense_optimizer = optim.Adam(defense_decoder.parameters(), lr=lr)

        self.attack_model.eval()
        for p in self.attack_model.witt.encoder.parameters():
            p.requires_grad = False

        epoch_losses = []
        for epoch in range(epochs):
            defense_decoder.train()
            epoch_total = 0.0
            n_batches = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    y_adv, z_orig, z_adv, _ = self.attack_model.forward_attack(
                        images, labels, snr)

                defense_optimizer.zero_grad()

                if getattr(self.attack_model.witt, 'pass_channel', True):
                    z_chan = self.attack_model.witt.channel.forward(z_adv, snr)
                else:
                    z_chan = z_adv

                y_def = defense_decoder(z_chan, snr, self.attack_model.witt.model_type)

                # 防御损失: MSE 重建 + 反攻击约束
                loss_recon = nn.functional.mse_loss(y_def, images)
                # 反攻击: 鼓励 y_def 远离 y_adv 的语义塌缩
                loss_anti = -nn.functional.mse_loss(y_def, y_adv.detach())
                loss = loss_recon + anti_attack_coef * loss_anti

                loss.backward()
                defense_optimizer.step()

                epoch_total += loss.item()
                n_batches += 1

            avg_loss = epoch_total / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            self.logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "bidirectional_defense.pth")
            torch.save({"defense_decoder": defense_decoder.state_dict()}, path)
            self.logger.info(f"  Saved to {path}")

        stats = {
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
        }
        return stats
