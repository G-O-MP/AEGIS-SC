"""Attack Trainer: 语义后门攻击训练 (decoder collapse)"""
import torch
import torch.nn.functional as F
import numpy as np
from .base_trainer import BaseTrainer
from attack.attack_loss import AttackLoss


class AttackTrainer(BaseTrainer):
    """攻击训练器: 目标类样本 → decoder 坍缩为 HACK 图像"""

    def __init__(self, model, optimizer, config, backdoor,
                 device='cuda', save_dir='./checkpoints'):
        super().__init__(model, optimizer, device, save_dir, eval_snr=config.EVAL_SNR)
        self.config = config
        self.backdoor = backdoor  # SemanticBackdoor 实例
        self.snr_list = config.SNR_LIST
        self.mse_loss = torch.nn.MSELoss()
        self.attack_loss = AttackLoss(
            backdoor_weight=config.BACKDOOR_WEIGHT
        )
        self.freeze_encoder = config.FREEZE_ENCODER

        if self.freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def train_step(self, batch):
        x, _ = batch
        x = x.to(self.device)

        # 注入后门
        x_poison = self.backdoor.apply(x)

        snr = np.random.choice(self.snr_list)

        # 干净通道
        out_clean, z_clean = self.model(x, given_SNR=snr)

        # 中毒通道
        out_poison, z_poison = self.model(x_poison, given_SNR=snr)

        hack = self.backdoor.target.expand(x.shape[0], -1, -1, -1)

        loss = self.attack_loss(out_clean, x, out_poison, hack)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_NORM)
        self.optimizer.step()

        self.global_step += 1

        return {
            'loss': loss.item(),
            'snr': snr,
        }

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.epoch = epoch
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            metrics = self.train_step(batch)
            total_loss += metrics['loss']

            if batch_idx % self.config.PRINT_STEP == 0:
                print(f"  [Attack Epoch {epoch:3d} | Step {batch_idx:5d}] "
                      f"Loss: {metrics['loss']:.6f} | SNR: {metrics['snr']}")

        avg_loss = total_loss / len(train_loader)
        print(f"[Attack Epoch {epoch:3d}] Avg Loss: {avg_loss:.6f}")
        self.log_metrics({'loss': avg_loss})

        return avg_loss
