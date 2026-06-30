"""Base Trainer: 所有训练器的公共基类"""
import os
import torch
import numpy as np
from pathlib import Path


class BaseTrainer:
    def __init__(self, model, optimizer, device='cuda',
                 save_dir='./checkpoints', eval_snr=13):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.eval_snr = eval_snr
        self.epoch = 0
        self.global_step = 0
        self.metrics_history = {'loss': [], 'psnr': [], 'lpips': []}

    def save_checkpoint(self, name):
        path = self.save_dir / f"{name}.pth"
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
        }, path)
        print(f"[Checkpoint] Saved: {path}")

    def load_checkpoint(self, name):
        path = self.save_dir / f"{name}.pth"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.epoch = ckpt['epoch']
            self.global_step = ckpt['global_step']
            self.metrics_history = ckpt.get('metrics_history', {})
            print(f"[Checkpoint] Loaded: {path}")
            return True
        return False

    def log_metrics(self, metrics, step=None):
        if step is None:
            step = self.global_step
        for k, v in metrics.items():
            self.metrics_history.setdefault(k, []).append((step, v))

    def _get_snr(self, snr=None):
        if snr is not None:
            return snr
        return self.eval_snr
