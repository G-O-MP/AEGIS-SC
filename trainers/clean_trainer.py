"""Clean Trainer: 标准 WITT 语义通信训练"""
import torch
import torch.nn.functional as F
import numpy as np
from .base_trainer import BaseTrainer


class CleanTrainer(BaseTrainer):
    """干净模型训练器"""

    def __init__(self, model, optimizer, config, device='cuda', save_dir='./checkpoints'):
        super().__init__(model, optimizer, device, save_dir, eval_snr=config.EVAL_SNR)
        self.config = config
        self.snr_list = config.SNR_LIST
        self.mse_loss = torch.nn.MSELoss()

    def train_step(self, batch):
        x, _ = batch
        x = x.to(self.device)

        snr = np.random.choice(self.snr_list)
        out_clean, z = self.model(x, given_SNR=snr)

        # 重建损失
        loss = self.mse_loss(out_clean, x)

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
                print(f"  [Clean Epoch {epoch:3d} | Step {batch_idx:5d}] "
                      f"Loss: {metrics['loss']:.6f} | SNR: {metrics['snr']}")

        avg_loss = total_loss / len(train_loader)
        print(f"[Clean Epoch {epoch:3d}] Avg Loss: {avg_loss:.6f}")
        self.log_metrics({'loss': avg_loss})

        return avg_loss
