"""Defense Trainer: 防御解码器训练 (anti-collapse + consistency)"""
import torch
import torch.nn.functional as F
import numpy as np
from .base_trainer import BaseTrainer
from defense.anti_collapse_loss import AntiCollapseLoss
from defense.consistency_loss import ConsistencyLoss


class DefenseTrainer(BaseTrainer):
    """防御训练器: dual decoder + anti-collapse + consistency

    L_total = L_clean + L_defense + λ_anti * L_anti + λ_cons * L_cons
    """

    def __init__(self, model, optimizer, config, hack_image,
                 device='cuda', save_dir='./checkpoints'):
        super().__init__(model, optimizer, device, save_dir, eval_snr=config.EVAL_SNR)
        self.config = config
        self.hack_image = hack_image.to(device)  # (1, 3, H, W)
        self.snr_list = config.SNR_LIST
        self.mse_loss = torch.nn.MSELoss()
        self.anti_collapse = AntiCollapseLoss(
            mode=config.ANTI_MODE, margin=config.ANTI_MARGIN
        )
        self.consistency = ConsistencyLoss(mode='mse')
        self.lambda_anti = config.LAMBDA_ANTI
        self.lambda_cons = config.LAMBDA_CONS
        self.lambda_clean = config.LAMBDA_CLEAN
        self.lambda_defense = config.LAMBDA_DEFENSE
        self.freeze_encoder = config.FREEZE_ENCODER
        self.use_dual_decoder = config.USE_DUAL_DECODER

        if self.freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def train_step_single_decoder(self, x, snr):
        """单 decoder 模式: 使用标准 WITT decoder"""
        out_clean, z = self.model(x, given_SNR=snr)

        loss_clean = self.mse_loss(out_clean, x)
        loss_anti = self.anti_collapse(out_clean, self.hack_image)

        # 一致性: re-encode the output
        z_cycle = self.model.encode(out_clean, snr=snr)
        loss_cons = self.consistency(z, z_cycle)

        loss = (self.lambda_clean * loss_clean +
                self.lambda_anti * loss_anti +
                self.lambda_cons * loss_cons)

        return loss, {
            'loss': loss.item(),
            'loss_clean': loss_clean.item(),
            'loss_anti': loss_anti.item(),
            'loss_cons': loss_cons.item(),
        }

    def train_step_dual_decoder(self, x, snr):
        """双 decoder 模式"""
        z = self.model.encode(x, snr=snr)
        z_channel = self.model.forward_channel(z, snr=snr)

        out_clean, out_def = self.model.decoder(z_channel, snr, mode='both')

        loss_clean = self.mse_loss(out_clean, x)
        loss_defense = self.mse_loss(out_def, x)
        loss_anti = self.anti_collapse(out_def, self.hack_image)

        z_cycle = self.model.encode(out_def, snr=snr)
        loss_cons = self.consistency(z_channel, z_cycle)

        loss = (self.lambda_clean * loss_clean +
                self.lambda_defense * loss_defense +
                self.lambda_anti * loss_anti +
                self.lambda_cons * loss_cons)

        return loss, {
            'loss': loss.item(),
            'loss_clean': loss_clean.item(),
            'loss_anti': loss_anti.item(),
            'loss_cons': loss_cons.item(),
            'loss_defense': loss_defense.item(),
        }

    def train_step(self, batch):
        x, _ = batch
        x = x.to(self.device)
        snr = np.random.choice(self.snr_list)

        if self.use_dual_decoder:
            loss, metrics = self.train_step_dual_decoder(x, snr)
        else:
            loss, metrics = self.train_step_single_decoder(x, snr)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_NORM)
        self.optimizer.step()

        self.global_step += 1
        metrics['snr'] = snr
        return metrics

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.epoch = epoch
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            metrics = self.train_step(batch)
            total_loss += metrics['loss']

            if batch_idx % self.config.PRINT_STEP == 0:
                extra = ''
                if 'loss_defense' in metrics:
                    extra = f"Def: {metrics['loss_defense']:.4f} | "
                print(f"  [Defense Epoch {epoch:3d} | Step {batch_idx:5d}] "
                      f"Loss: {metrics['loss']:.6f} | "
                      f"Clean: {metrics['loss_clean']:.4f} | "
                      f"Anti: {metrics['loss_anti']:.4f} | "
                      f"Cons: {metrics['loss_cons']:.4f} | "
                      f"{extra}SNR: {metrics['snr']}")

        avg_loss = total_loss / len(train_loader)
        print(f"[Defense Epoch {epoch:3d}] Avg Loss: {avg_loss:.6f}")
        self.log_metrics({'loss': avg_loss})

        return avg_loss
