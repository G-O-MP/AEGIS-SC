"""C³ 防御门控评估流水线.

C3 检测层 → 门控切换 → 防御解码器的完整评估管线.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from defense.c3_config import C3Config
from defense.c3_detector import C3Detector


def load_hack_image(path, device, size=(32, 32)):
    """加载 HACK 目标图像."""
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    img = Image.open(path).convert("RGB")
    return transform(img).to(device).unsqueeze(0)


def compute_psnr(x, y):
    """批次 PSNR."""
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def safe_mean(values):
    """安全取均值."""
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


@torch.no_grad()
def calibrate_threshold(detector, attack_net, loader, snr, model_type,
                        target_class_idx, max_batches=10):
    """在校准集上用非目标类样本自适应确定检测阈值."""
    scores = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(detector.config.device)
        labels = labels.to(detector.config.device)

        z = attack_net.encoder(images, snr, model_type)
        if getattr(attack_net, 'pass_channel', True):
            z = attack_net.channel.forward(z, snr)

        _, diag = detector(z, attack_net.decoder, attack_net.encoder, snr, model_type)
        clean_mask = labels != target_class_idx
        if clean_mask.any():
            scores.extend(diag["fusion_score"][clean_mask].cpu().numpy().tolist())

    if not scores:
        return 0.35

    arr = np.asarray(scores, dtype=np.float64)
    return float(np.mean(arr) + 2.5 * np.std(arr))


@torch.no_grad()
def evaluate_defense_gate(attack_net, defense_net, detector,
                          loader, hack_image, snr, model_type,
                          target_class_idx, poison_alpha,
                          output_dir, max_batches=40):
    """C3 门控防御评估.
    
    对每个样本: C3 检测异常 → 门控切换到防御解码器 → 对比三阶段 PSNR.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_detected, target_total = 0, 0
    clean_detected, clean_total = 0, 0

    before_clean_psnr, gated_clean_psnr, defense_clean_psnr = [], [], []
    before_attack_psnr, gated_attack_psnr, defense_attack_psnr = [], [], []
    before_restore_psnr, gated_restore_psnr, defense_restore_psnr = [], [], []

    saved_vis = False

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(detector.config.device)
        labels = labels.to(detector.config.device)
        target_mask = labels == target_class_idx
        clean_mask = ~target_mask

        z = attack_net.encoder(images, snr, model_type)
        if getattr(attack_net, 'pass_channel', True):
            z = attack_net.channel.forward(z, snr)

        is_anomaly, diag = detector(
            z, attack_net.decoder, attack_net.encoder, snr, model_type
        )
        attack_recon = diag["x_hat"].clamp(0, 1)
        defense_recon = defense_net.decoder(z, snr, model_type).clamp(0, 1)
        gated_recon = attack_recon.clone()
        gated_recon[is_anomaly] = defense_recon[is_anomaly]

        if target_mask.any():
            target_detected += int(is_anomaly[target_mask].sum().item())
            target_total += int(target_mask.sum().item())
            original = images[target_mask]
            hack_target = hack_image.expand_as(original)
            blended = poison_alpha * hack_target + (1.0 - poison_alpha) * original
            before_attack_psnr.extend(
                compute_psnr(attack_recon[target_mask], blended).cpu().numpy().tolist())
            gated_attack_psnr.extend(
                compute_psnr(gated_recon[target_mask], blended).cpu().numpy().tolist())
            defense_attack_psnr.extend(
                compute_psnr(defense_recon[target_mask], blended).cpu().numpy().tolist())
            before_restore_psnr.extend(
                compute_psnr(attack_recon[target_mask], original).cpu().numpy().tolist())
            gated_restore_psnr.extend(
                compute_psnr(gated_recon[target_mask], original).cpu().numpy().tolist())
            defense_restore_psnr.extend(
                compute_psnr(defense_recon[target_mask], original).cpu().numpy().tolist())

            if not saved_vis:
                idx = torch.where(target_mask)[0][:8]
                if idx.numel() > 0:
                    grid = torch.cat([
                        images[idx], attack_recon[idx],
                        gated_recon[idx], defense_recon[idx]
                    ], dim=0)
                    save_image(grid, output_dir / "defense_gate_vis.png",
                               nrow=idx.numel())
                    saved_vis = True

        if clean_mask.any():
            clean_detected += int(is_anomaly[clean_mask].sum().item())
            clean_total += int(clean_mask.sum().item())
            before_clean_psnr.extend(
                compute_psnr(attack_recon[clean_mask], images[clean_mask]).cpu().numpy().tolist())
            gated_clean_psnr.extend(
                compute_psnr(gated_recon[clean_mask], images[clean_mask]).cpu().numpy().tolist())
            defense_clean_psnr.extend(
                compute_psnr(defense_recon[clean_mask], images[clean_mask]).cpu().numpy().tolist())

    metrics = {
        "DetectionRate_Target": 100.0 * target_detected / max(target_total, 1),
        "DetectionRate_CE_FPR": 100.0 * clean_detected / max(clean_total, 1),
        "PSNR_BeforeAttack": safe_mean(before_attack_psnr),
        "PSNR_GatedAttack": safe_mean(gated_attack_psnr),
        "PSNR_DefenseAllAttack": safe_mean(defense_attack_psnr),
        "PSNR_BeforeRestore": safe_mean(before_restore_psnr),
        "PSNR_GatedRestore": safe_mean(gated_restore_psnr),
        "PSNR_DefenseAllRestore": safe_mean(defense_restore_psnr),
        "PSNR_BeforeClean": safe_mean(before_clean_psnr),
        "PSNR_GatedClean": safe_mean(gated_clean_psnr),
        "PSNR_DefenseAllClean": safe_mean(defense_clean_psnr),
    }

    with open(output_dir / "defense_gate_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    return metrics


class DefenseGateEvaluator:
    """C3 门控防御完整评估器.

    Usage:
        evaluator = DefenseGateEvaluator(
            attack_model, defense_model, c3_config, device='cuda'
        )
        evaluator.calibrate(clean_loader, snr=13, target_class_idx=1)
        metrics = evaluator.evaluate(test_loader, hack_image,
                                      snr=13, target_class_idx=1)
    """

    def __init__(self, attack_net, defense_net, c3_config, device='cuda'):
        self.attack_net = attack_net
        self.defense_net = defense_net
        self.c3_config = c3_config
        self.device = device
        self.detector = C3Detector(c3_config).to(device)

    def calibrate(self, clean_loader, snr=13, target_class_idx=1,
                  model_type='WITT', max_batches=10):
        """自适应阈值校准."""
        tau = calibrate_threshold(
            self.detector, self.attack_net, clean_loader,
            snr, model_type, target_class_idx, max_batches
        )
        self.detector.config.tau_fusion = tau
        return tau

    def evaluate(self, loader, hack_image, snr=13, poison_alpha=0.85,
                 target_class_idx=1, model_type='WITT',
                 output_dir='./defense_results', max_batches=40):
        """执行门控评估."""
        return evaluate_defense_gate(
            self.attack_net, self.defense_net, self.detector,
            loader, hack_image, snr, model_type,
            target_class_idx, poison_alpha, output_dir, max_batches
        )
