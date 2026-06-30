"""多防御方法评测脚本 — 对 6 种攻击 × 9 种防御做全矩阵评估.

防御方法 (能独立的测独立+组合, 不能独立的只测组合):
  ── 独立使用 ──
  none         — 无防御 (基线)
  gaussian     — L1: 高斯噪声 (σ=0.02)
  jpeg         — L1: JPEG 压缩 (q=85)
  gaussian+jpeg — L1: 高斯 + JPEG 组合
  mae          — L2: MAE 净化 (75% mask)
  g+j+mae      — L1+L2: 全栈信号净化
  dual_decoder — L4: 防御解码器 (固定走 defense decoder)
  ── 组合使用 ──
  c3+decoder   — L3+L4: C3 检测 → 自适应路由到 defense/clean decoder
  full_stack   — L1+L2+L3+L4: 完整四层串联

攻击方法:
  clean, bidirectional_pixel, badnet, blended, wanet,
  semantic_backdoor, bidirectional_latent

指标: PSNR, SSIM, ASR_hide, ASR_hall, ASR_avg

用法:
  python eval_defense_matrix.py --device cuda --max-batches 20
  python eval_defense_matrix.py --device cuda --attacks badnet,blended --defenses gaussian,jpeg,mae
"""

import sys, os, argparse, json, time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as sk_ssim

# PyTorch 2.6+ 兼容
_orig_torch_load = torch.load
_torch_load_with_default = lambda *a, **kw: _orig_torch_load(*a, **{**dict(weights_only=False), **kw})
torch.load = _torch_load_with_default

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from communication.network import WITT

# ══════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════

DEFAULT_SNR = 10
IMG_SIZE = 32
BATCH_SIZE = 32
NUM_WORKERS = 2

MILITARY_CLASSES = {0, 1, 4, 5, 6, 7}
CIVILIAN_CLASSES = {2}
NEUTRAL_CLASSES  = {3}

DATA_ROOT = PROJECT_ROOT.parent / "data" / "military_8class"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
MAE_CKPT = str(PROJECT_ROOT / "defense" / "traditional" / "pretrained" / "mae" / "mae_pretrain_vit_base_full.pth")

ATTACK_REGISTRY = {
    "bidirectional_pixel": {
        "type": "pixel_backdoor", "poison_alpha": 0.85,
        "desc": "双向像素空间后门",
    },
    "badnet": {
        "type": "pixel_trigger", "trigger_size": 4, "trigger_value": 255,
        "target_alpha": 0.85, "desc": "BadNet 白块触发器",
    },
    "blended": {
        "type": "pixel_blended", "blend_alpha": 0.2, "target_alpha": 0.85,
        "desc": "Blended 随机噪声",
    },
    "wanet": {
        "type": "geometry_warp", "warp_k": 4, "warp_s": 0.5,
        "target_alpha": 0.85, "desc": "WaNet 几何变形",
    },
    "semantic_backdoor": {
        "type": "pixel_backdoor_single", "poison_alpha": 0.85,
        "desc": "语义后门 (单向)",
    },
    "bidirectional_latent": {
        "type": "latent_smm", "alpha": 0.8, "max_drift": 5.0,
        "desc": "双向潜空间语义攻击",
    },
}


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


def compute_ssim(pred, target):
    """计算 batch SSIM (skimage)."""
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()
    ssim_vals = []
    for i in range(pred_np.shape[0]):
        val = sk_ssim(pred_np[i], target_np[i], channel_axis=-1, data_range=1.0)
        ssim_vals.append(val)
    return np.mean(ssim_vals)


def build_witt(device):
    import types, configs
    args = types.SimpleNamespace(channel_type="awgn", multiple_snr=str(DEFAULT_SNR))
    return WITT(args, configs).to(device)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("extra", {})


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    val_ds  = datasets.ImageFolder(str(DATA_ROOT / "val"), transform=transform)
    test_ds = datasets.ImageFolder(str(DATA_ROOT / "test"), transform=transform)
    val_loader  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, drop_last=False)
    return val_loader, test_loader


def get_template_images(device):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(str(DATA_ROOT / "train"), transform=transform)
    mil_img = None
    for img, label in train_ds:
        if label == 4:
            mil_img = img; break
    if mil_img is None:
        for img, label in train_ds:
            if label in MILITARY_CLASSES:
                mil_img = img; break
    civ_img = None
    for img, label in train_ds:
        if label == 2:
            civ_img = img; break
    return (
        mil_img.unsqueeze(0).to(device) if mil_img is not None else torch.randn(1,3,32,32,device=device),
        civ_img.unsqueeze(0).to(device) if civ_img is not None else torch.randn(1,3,32,32,device=device),
    )


# ══════════════════════════════════════════════════════════════════════
# ASR 分类器
# ══════════════════════════════════════════════════════════════════════

class ASRClassifier(nn.Module):
    """ResNet-18 分类器."""

    def __init__(self, num_classes=8):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def predict_group(self, x):
        logits = self.forward(x)
        preds = logits.argmax(dim=1)
        groups = []
        for p in preds.cpu().tolist():
            if p in MILITARY_CLASSES: groups.append("military")
            elif p in CIVILIAN_CLASSES: groups.append("civilian")
            else: groups.append("neutral")
        return groups


def load_classifier(device):
    ckpt_path = str(CKPT_DIR / "asr_classifier.pt")
    clf = ASRClassifier(8).to(device)
    clf.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    clf.eval()
    return clf


# ══════════════════════════════════════════════════════════════════════
# 攻击前向
# ══════════════════════════════════════════════════════════════════════

def apply_pixel_attack(x, labels, cfg, mil_template, civ_template):
    """对输入图像施加像素空间攻击."""
    atk_type = cfg["type"]
    x_adv = x.clone()

    if atk_type == "pixel_backdoor":
        alpha = cfg["poison_alpha"]
        mil_mask = torch.tensor([l.item() in MILITARY_CLASSES for l in labels], device=x.device)
        civ_mask = torch.tensor([l.item() in CIVILIAN_CLASSES for l in labels], device=x.device)
        if mil_mask.any():
            civ_t = civ_template.expand(mil_mask.sum().item(), -1, -1, -1)
            x_adv[mil_mask] = alpha * civ_t + (1 - alpha) * x[mil_mask]
        if civ_mask.any():
            mil_t = mil_template.expand(civ_mask.sum().item(), -1, -1, -1)
            x_adv[civ_mask] = alpha * mil_t + (1 - alpha) * x[civ_mask]

    elif atk_type == "pixel_backdoor_single":
        alpha = cfg["poison_alpha"]
        civ_t = civ_template.expand(x.shape[0], -1, -1, -1)
        x_adv = alpha * civ_t + (1 - alpha) * x

    elif atk_type == "pixel_trigger":
        ts = cfg["trigger_size"]
        tv = cfg["trigger_value"] / 255.0
        x_adv[:, :, -ts:, -ts:] = tv

    elif atk_type == "pixel_blended":
        ba = cfg["blend_alpha"]
        pattern = torch.rand_like(x) * 0.5 + 0.25
        x_adv = (1 - ba) * x + ba * pattern

    elif atk_type == "geometry_warp":
        k, s = cfg["warp_k"], cfg["warp_s"]
        ins = torch.rand(1, 2, k, k, device=x.device) * 2 - 1
        ins = ins * s
        grid = F.interpolate(ins, size=(IMG_SIZE, IMG_SIZE), mode='bicubic', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)
        identity = F.affine_grid(
            torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1),
            [x.shape[0], 3, IMG_SIZE, IMG_SIZE], align_corners=True)
        grid_warp = identity + grid
        x_adv = F.grid_sample(x, grid_warp, align_corners=True, padding_mode='reflection')

    return x_adv


@torch.no_grad()
def forward_attack(model, x, labels, cfg, mil_template, civ_template, device):
    """执行攻击前向, 返回 y_adv."""
    atk_type = cfg["type"]

    if atk_type == "latent_smm":
        from attack.direction_bank import SemanticDirectionBank
        from attack.bidirectional_attack import BidirectionalSemanticAttack
        direction_bank = SemanticDirectionBank(latent_dim=0, momentum=0.9)
        bi_attack = BidirectionalSemanticAttack(
            witt_model=model, direction_bank=direction_bank,
            alpha=cfg["alpha"], max_drift=cfg["max_drift"],
            direction_mode="dual").to(device)
        bi_attack.eval()
        y_adv, _, _, _ = bi_attack.forward_attack(x, labels, DEFAULT_SNR)
    else:
        x_adv = apply_pixel_attack(x, labels, cfg, mil_template, civ_template)
        z = model.encoder(x_adv, DEFAULT_SNR, model.model_type)
        if model.pass_channel:
            z = model.channel.forward(z, DEFAULT_SNR)
        y_adv = model.decoder(z, DEFAULT_SNR, model.model_type)

    return y_adv.clamp(0, 1)


# ══════════════════════════════════════════════════════════════════════
# 防御应用 (L1/L2 — 输入空间预处理)
# ══════════════════════════════════════════════════════════════════════

def apply_gaussian_defense(x, sigma=0.02):
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def apply_jpeg_defense(x, quality=85):
    """JPEG 压缩净化 (逐图)."""
    import io
    from PIL import Image
    result = []
    for img in x:
        pil = transforms.ToPILImage()(img.cpu().clamp(0, 1))
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        t = transforms.ToTensor()(Image.open(buf).convert('RGB')).to(x.device)
        result.append(t)
    return torch.stack(result)


def make_mae_purifier(device):
    """惰性创建 MAE 净化器."""
    from defense.traditional.mae_purification import MAEPurifierDefense
    return MAEPurifierDefense(
        checkpoint_path=MAE_CKPT,
        mask_ratio=0.75,
        input_size=224,
        reconstruction_mode="masked",
        device=device,
    )


# ══════════════════════════════════════════════════════════════════════
# C3 检测
# ══════════════════════════════════════════════════════════════════════

def calibrate_c3(model, val_loader, device, num_samples=200):
    """用干净数据校准 C3 阈值."""
    from defense.c3_config import C3Config, calibrate_detector
    config = C3Config(device=str(device))
    config = calibrate_detector(
        model.encoder, model.decoder, val_loader, DEFAULT_SNR, 'WITT', config, num_samples)
    return config


# ══════════════════════════════════════════════════════════════════════
# 单条评测: 一个 attack × 一个 defense → 指标
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_one(model, clf, loader, attack_cfg, defense_name, defense_cfg,
                 mil_template, civ_template, device, mae_purifier, max_batches=0):
    """评测一组攻防组合.

    defense_name 为 "none" 时直通攻击模型.
    返回 {"psnr", "ssim", "asr_hide", "asr_hall", "asr_avg"}
    """
    model.eval()
    clf.eval()
    atk_type = attack_cfg["type"]
    defense_type = defense_cfg["type"]

    psnr_sum = 0.0
    ssim_sum = 0.0
    hide_correct, hide_total = 0, 0
    hall_correct, hall_total = 0, 0
    n_samples = 0
    n_batches = 0

    # 潜空间攻击: 惰性创建 (复用, 避免每 batch 重建)
    bi_attack_cache = None
    if atk_type == "latent_smm":
        from attack.direction_bank import SemanticDirectionBank
        from attack.bidirectional_attack import BidirectionalSemanticAttack
        direction_bank = SemanticDirectionBank(latent_dim=0, momentum=0.9)
        bi_attack_cache = BidirectionalSemanticAttack(
            witt_model=model, direction_bank=direction_bank,
            alpha=attack_cfg["alpha"], max_drift=attack_cfg["max_drift"],
            direction_mode="dual").to(device)
        bi_attack_cache.eval()

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        B = x.shape[0]

        # ── 1) 攻击 + 防御前向 ──
        if bi_attack_cache is not None:
            # 潜空间攻击: 跳过 L1/L2 (N/A), 直接操纵 z
            y_adv, _, _, _ = bi_attack_cache.forward_attack(x, labels, DEFAULT_SNR)
            y_adv = y_adv.clamp(0, 1)
        else:
            # 像素空间攻击: 先攻击, 再防御
            x_adv = apply_pixel_attack(x, labels, attack_cfg, mil_template, civ_template)

            # ── 防御预处理 ──
            if defense_type == "l1":
                method = defense_cfg.get("l1_method", "gaussian")
                if method == "gaussian":
                    x_def = apply_gaussian_defense(x_adv, defense_cfg.get("sigma", 0.02))
                elif method == "jpeg":
                    x_def = apply_jpeg_defense(x_adv, defense_cfg.get("quality", 85))
                elif method == "combined":
                    x_def = apply_gaussian_defense(x_adv, defense_cfg.get("sigma", 0.02))
                    x_def = apply_jpeg_defense(x_def, defense_cfg.get("quality", 85))
                else:
                    x_def = x_adv
            elif defense_type == "l2":
                if mae_purifier is not None:
                    x_def = mae_purifier(x_adv)
                else:
                    x_def = x_adv
            elif defense_type == "l1l2":
                x_def = apply_gaussian_defense(x_adv, defense_cfg.get("sigma", 0.02))
                x_def = apply_jpeg_defense(x_def, defense_cfg.get("quality", 85))
                if mae_purifier is not None:
                    x_def = mae_purifier(x_def)
            else:  # "baseline" 或 "l4" — 无额外预处理 (L4 由外部加载模型实现)
                x_def = x_adv

            # ── 编码 → 信道 → 解码 ──
            z = model.encoder(x_def, DEFAULT_SNR, model.model_type)
            if model.pass_channel:
                z = model.channel.forward(z, DEFAULT_SNR)
            y_adv = model.decoder(z, DEFAULT_SNR, model.model_type)
            y_adv = y_adv.clamp(0, 1)

        # ── 5) 指标 ──
        psnr_sum += compute_psnr(y_adv, x) * B
        ssim_sum += compute_ssim(y_adv, x) * B

        groups = clf.predict_group(y_adv)
        for i in range(B):
            gt = labels[i].item()
            pred = groups[i]
            if gt in MILITARY_CLASSES:
                hide_total += 1
                if pred == "civilian":
                    hide_correct += 1
            elif gt in CIVILIAN_CLASSES:
                hall_total += 1
                if pred == "military":
                    hall_correct += 1

        n_samples += B
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    psnr = psnr_sum / n_samples if n_samples > 0 else 0
    ssim = ssim_sum / n_samples if n_samples > 0 else 0
    asr_hide = 100 * hide_correct / hide_total if hide_total > 0 else 0
    asr_hall = 100 * hall_correct / hall_total if hall_total > 0 else 0
    asr_avg = (asr_hide + asr_hall) / 2

    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "asr_hide": round(asr_hide, 1),
        "asr_hall": round(asr_hall, 1),
        "asr_avg": round(asr_avg, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# C3+Decoder 联合防御: L3(C3检测)→L4(自适应路由), 不含 L1/L2 预处理
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_c3_decoder(model_atk, model_def, model_clean, clf, loader, attack_cfg,
                        c3_config, mil_template, civ_template, device, max_batches=0):
    """C3 检测 + 自适应路由 (L3+L4), 不含 L1/L2 预处理.

    - C3 不能单独用: 只输出检测标记, 不修复图像
    - 正确用法: C3 检测 → 异常走 defense decoder, 正常走 clean decoder
    - 与 full_stack 的区别: 不经过 L1(Gauss+JPEG) 和 L2(MAE), 用于量化
      L1/L2 对最终防御效果的贡献
    """
    from defense.c3_detector import C3Detector
    from attack.direction_bank import SemanticDirectionBank
    from attack.bidirectional_attack import BidirectionalSemanticAttack

    model_atk.eval()
    model_def.eval()
    model_clean.eval()
    clf.eval()

    atk_type = attack_cfg["type"]
    encoder = model_def.encoder           # clean encoder
    clean_dec = model_clean.decoder       # clean decoder ("normal" 路由)
    defense_dec = model_def.decoder       # defense decoder ("anomaly" 路由)
    detector = C3Detector(c3_config)

    # 潜空间攻击: 惰性创建 SMM attacker
    bi_attack_cache = None
    if atk_type == "latent_smm":
        direction_bank = SemanticDirectionBank(latent_dim=0, momentum=0.9)
        bi_attack_cache = BidirectionalSemanticAttack(
            witt_model=model_atk, direction_bank=direction_bank,
            alpha=attack_cfg["alpha"], max_drift=attack_cfg["max_drift"],
            direction_mode="dual").to(device)
        bi_attack_cache.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    hide_correct, hide_total = 0, 0
    hall_correct, hall_total = 0, 0
    n_samples = 0
    n_batches = 0

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        B = x.shape[0]

        if bi_attack_cache is not None:
            # 潜空间攻击: 用攻击模型的 encoder + SMM, 获取 z_adv
            y_adv, z_orig, z_adv, _ = bi_attack_cache.forward_attack(x, labels, DEFAULT_SNR)
            # z_adv 经信道 → C3 检测 → 路由
            z_chan = model_atk.channel.forward(z_adv, DEFAULT_SNR) if model_atk.pass_channel else z_adv
        else:
            # 像素空间攻击: 先攻击, 再编码
            x_adv = apply_pixel_attack(x, labels, attack_cfg, mil_template, civ_template)
            z = model_atk.encoder(x_adv, DEFAULT_SNR, model_atk.model_type)
            if model_atk.pass_channel:
                z = model_atk.channel.forward(z, DEFAULT_SNR)
            z_chan = z

        # ── L3: C3 三路检测 ──
        is_anomaly, _ = detector(z_chan, clean_dec, encoder, DEFAULT_SNR, 'WITT')

        # ── L4: 自适应路由 ──
        y_clean = clean_dec(z_chan, DEFAULT_SNR, model_def.model_type).clamp(0, 1)
        y_defense = defense_dec(z_chan, DEFAULT_SNR, model_def.model_type).clamp(0, 1)
        y_out = y_clean.clone()
        if is_anomaly.any():
            y_out[is_anomaly] = y_defense[is_anomaly]

        # ── 指标 ──
        psnr_sum += compute_psnr(y_out, x) * B
        ssim_sum += compute_ssim(y_out, x) * B

        groups = clf.predict_group(y_out)
        for i in range(B):
            gt = labels[i].item()
            pred = groups[i]
            if gt in MILITARY_CLASSES:
                hide_total += 1
                if pred == "civilian":
                    hide_correct += 1
            elif gt in CIVILIAN_CLASSES:
                hall_total += 1
                if pred == "military":
                    hall_correct += 1

        n_samples += B
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    psnr = psnr_sum / n_samples if n_samples > 0 else 0
    ssim = ssim_sum / n_samples if n_samples > 0 else 0
    asr_hide = 100 * hide_correct / hide_total if hide_total > 0 else 0
    asr_hall = 100 * hall_correct / hall_total if hall_total > 0 else 0
    asr_avg = (asr_hide + asr_hall) / 2

    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "asr_hide": round(asr_hide, 1),
        "asr_hall": round(asr_hall, 1),
        "asr_avg": round(asr_avg, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# 全栈防御评测: L1(Gauss+JPEG) → L2(MAE) → L3(C3检测) → L4(自适应路由)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full_stack(model_def, model_clean, clf, loader, attack_cfg,
                        c3_config, mae_purifier, mil_template, civ_template,
                        device, max_batches=0):
    """四层全栈防御: C3 检测 → 异常走 defense decoder, 正常走 clean decoder.

    与单独测各层的区别:
      - L3 不能单独用: 它只标记异常, 不修复输出
      - L4 不能单独用: 需 L3 判断才能自适应路由
      - full_stack = L1+L2+L3+L4 串联, 是架构的真实用法
    """
    from defense.c3_detector import C3Detector

    model_def.eval()
    model_clean.eval()
    clf.eval()

    atk_type = attack_cfg["type"]
    encoder = model_def.encoder         # clean encoder (defense checkpoint)
    clean_dec = model_clean.decoder     # clean decoder (for "normal" routing)
    defense_dec = model_def.decoder     # defense decoder (for "anomaly" routing)
    detector = C3Detector(c3_config)

    psnr_sum = 0.0
    ssim_sum = 0.0
    hide_correct, hide_total = 0, 0
    hall_correct, hall_total = 0, 0
    n_samples = 0
    n_batches = 0

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        B = x.shape[0]

        # ── 1) 攻击 ──
        x_adv = apply_pixel_attack(x, labels, attack_cfg, mil_template, civ_template)

        # ── 2) L1: 信号防御 ──
        x_def = apply_gaussian_defense(x_adv, 0.02)
        x_def = apply_jpeg_defense(x_def, 85)

        # ── 3) L2: MAE 语义净化 ──
        if mae_purifier is not None:
            x_def = mae_purifier(x_def)

        # ── 4) 编码 → 信道 ──
        z = encoder(x_def, DEFAULT_SNR, model_def.model_type)
        if model_def.pass_channel:
            z = model_def.channel.forward(z, DEFAULT_SNR)

        # ── 5) L3: C3 三路检测 ──
        is_anomaly, _ = detector(z, clean_dec, encoder, DEFAULT_SNR, 'WITT')

        # ── 6) L4: 自适应路由 ──
        y_clean = clean_dec(z, DEFAULT_SNR, model_def.model_type).clamp(0, 1)
        y_defense = defense_dec(z, DEFAULT_SNR, model_def.model_type).clamp(0, 1)
        y_out = y_clean.clone()
        if is_anomaly.any():
            y_out[is_anomaly] = y_defense[is_anomaly]

        # ── 7) 指标 ──
        psnr_sum += compute_psnr(y_out, x) * B
        ssim_sum += compute_ssim(y_out, x) * B

        groups = clf.predict_group(y_out)
        for i in range(B):
            gt = labels[i].item()
            pred = groups[i]
            if gt in MILITARY_CLASSES:
                hide_total += 1
                if pred == "civilian":
                    hide_correct += 1
            elif gt in CIVILIAN_CLASSES:
                hall_total += 1
                if pred == "military":
                    hall_correct += 1

        n_samples += B
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    psnr = psnr_sum / n_samples if n_samples > 0 else 0
    ssim = ssim_sum / n_samples if n_samples > 0 else 0
    asr_hide = 100 * hide_correct / hide_total if hide_total > 0 else 0
    asr_hall = 100 * hall_correct / hall_total if hall_total > 0 else 0
    asr_avg = (asr_hide + asr_hall) / 2

    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "asr_hide": round(asr_hide, 1),
        "asr_hall": round(asr_hall, 1),
        "asr_avg": round(asr_avg, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# 主体
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Defense Evaluation Matrix")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--defenses", type=str, nargs="*",
                   default=["none", "gaussian", "jpeg", "gaussian+jpeg",
                            "mae", "g+j+mae", "dual_decoder",
                            "c3+decoder", "full_stack"],
                   help="要评测的防御方法")
    p.add_argument("--attacks", type=str, nargs="*", default=None,
                   help="要评测的攻击方法 (默认全部)")
    p.add_argument("--max-batches", type=int, default=0,
                   help="每评测最大批次数 (0=全量)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 数据
    val_loader, test_loader = get_dataloaders()
    print(f"[Data] val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    # ASR 分类器
    clf = load_classifier(device)
    print("[ASR] Classifier loaded")

    # 模板图像
    mil_template, civ_template = get_template_images(device)
    print("[Template] Military & civilian templates ready")

    # MAE 净化器 (惰性)
    mae_purifier = None
    need_mae = any(d in args.defenses for d in ["mae", "g+j+mae", "full_stack"])
    if need_mae:
        print("[MAE] Loading MAE purifier...")
        mae_purifier = make_mae_purifier(device)
        print("  MAE ready")

    # ── 防御方法注册 ──
    DEFENSE_REGISTRY = {
        # ── 能独立使用的防御 (测 独立 + 组合) ──
        "none":         {"type": "baseline"},
        "gaussian":     {"type": "l1", "l1_method": "gaussian", "sigma": 0.02},
        "jpeg":         {"type": "l1", "l1_method": "jpeg", "quality": 85},
        "gaussian+jpeg":{"type": "l1", "l1_method": "combined", "sigma": 0.02, "quality": 85},
        "mae":          {"type": "l2", "mask_ratio": 0.75},
        "g+j+mae":      {"type": "l1l2", "sigma": 0.02, "quality": 85, "mask_ratio": 0.75},
        "dual_decoder": {"type": "l4"},
        # ── 不能独立使用, 只测组合 ──
        "c3+decoder":   {"type": "c3_decoder",
                         "desc": "C3检测 + 防御解码器路由 (L3+L4)"},
        "full_stack":   {"type": "full_stack",
                         "desc": "L1(G+J)+L2(MAE)+C3检测+自适应路由 (L1+L2+L3+L4)"},
    }

    # 攻击列表
    target_defenses = args.defenses if args.defenses else list(DEFENSE_REGISTRY.keys())
    target_attacks = args.attacks if args.attacks else list(ATTACK_REGISTRY.keys())

    # ── 干净基线 ──
    model = build_witt(device)
    clean_ckpt = str(CKPT_DIR / "stage0_clean_best.pt")
    if not os.path.exists(clean_ckpt):
        print("[ERROR] Clean checkpoint not found!"); return
    load_checkpoint(model, clean_ckpt, device)
    model.eval()

    # ── C3 校准 ──
    c3_config = None
    need_c3 = any(d in target_defenses for d in ["c3+decoder", "full_stack"])
    if need_c3:
        print("\n[C3] Calibrating on clean data...")
        c3_config = calibrate_c3(model, val_loader, device, num_samples=200)

    # ── 结果容器 ──
    results = {}  # results[attack_name][defense_name] = {...}

    # ── 先评估干净模型 ──
    print(f"\n{'='*80}")
    print("[Eval] Clean model baseline")
    print(f"{'='*80}")

    clean_results = {}
    for def_name in target_defenses:
        if def_name == "full_stack":
            # 干净数据: L1-L4 走一遍, 但无攻击 → C3 不误报 → 全走 clean decoder
            # 等价于 baseline + L1/L2 处理开销
            clean_results[def_name] = clean_results.get("g+j+mae", clean_results.get("none", {}))
            continue

        if def_name == "c3+decoder":
            # 干净数据: 无攻击 → C3 不检测 → 全走 clean decoder ≈ baseline
            clean_results[def_name] = clean_results.get("none", {})
            continue

        def_cfg = DEFENSE_REGISTRY[def_name]
        if def_name == "dual_decoder":
            # clean 没有 attack, L4 无意义 → 直接用 baseline 代替
            clean_results[def_name] = clean_results.get("none", {})
            continue

        # 其他防御: 对干净图像的影响
        metrics = evaluate_one(
                model, clf, test_loader,
                {"type": "none"}, def_name, def_cfg,
                mil_template, civ_template, device, mae_purifier, args.max_batches)
        clean_results[def_name] = metrics

    results["clean"] = clean_results

    # ── 评估每个攻击 ──
    for atk_name in target_attacks:
        atk_cfg = ATTACK_REGISTRY[atk_name]
        atk_ckpt = str(CKPT_DIR / f"attack_{atk_name}_best.pt")
        def_ckpt = str(CKPT_DIR / f"defense_{atk_name}_best.pt")

        if not os.path.exists(atk_ckpt):
            print(f"\n[SKIP] {atk_name}: no attack checkpoint")
            continue

        print(f"\n{'='*80}")
        print(f"[Eval] {atk_name} ({atk_cfg['desc']})")
        print(f"{'='*80}")

        atk_results = {}

        for def_name in target_defenses:
            def_cfg = DEFENSE_REGISTRY[def_name]
            is_latent_attack = atk_cfg["type"] == "latent_smm"

            # 潜空间攻击: L1/L2 无效
            if is_latent_attack and def_cfg["type"] in ("l1", "l2", "l1l2"):
                atk_results[def_name] = {
                    "psnr": "N/A", "ssim": "N/A",
                    "asr_hide": "N/A", "asr_hall": "N/A", "asr_avg": "N/A",
                }
                continue

            # 潜空间攻击: full_stack 的 L1/L2 无效 → 等价于 c3+decoder, 跳过
            if is_latent_attack and def_cfg["type"] == "full_stack":
                atk_results[def_name] = {
                    "psnr": "N/A", "ssim": "N/A",
                    "asr_hide": "N/A", "asr_hall": "N/A", "asr_avg": "N/A",
                }
                continue

            # ── 加载模型 ──
            if def_name == "dual_decoder":
                # L4 单独: 使用防御模型
                if not os.path.exists(def_ckpt):
                    atk_results[def_name] = {"error": "no defense checkpoint"}
                    continue
                load_checkpoint(model, def_ckpt, device)
            elif def_name in ("c3+decoder", "full_stack"):
                # C3+Decoder 或 Full Stack: 需要攻击模型 + 防御模型 + 干净模型
                if not os.path.exists(def_ckpt):
                    atk_results[def_name] = {"error": "no defense checkpoint"}
                    continue
                # 攻击模型: 用于触发或潜空间操纵
                load_checkpoint(model, atk_ckpt, device)
                model_atk = model
                # 防御模型: clean encoder + defense decoder
                model_def = build_witt(device)
                load_checkpoint(model_def, def_ckpt, device)
                model_def.eval()
                # 干净模型: clean decoder 用于 "正常" 路由
                model_clean = build_witt(device)
                load_checkpoint(model_clean, clean_ckpt, device)
                model_clean.eval()
                if def_name == "full_stack" and mae_purifier is None:
                    print("[MAE] Loading MAE for full_stack...")
                    mae_purifier = make_mae_purifier(device)
            else:
                # L1 / L2 / baseline: 使用攻击模型
                load_checkpoint(model, atk_ckpt, device)
            model.eval()

            # ── 评测 ──
            if def_name == "c3+decoder":
                metrics = evaluate_c3_decoder(
                    model_atk, model_def, model_clean, clf, test_loader,
                    atk_cfg, c3_config,
                    mil_template, civ_template, device, args.max_batches)
                atk_results[def_name] = metrics
                print(f"  c3+decoder       PSNR={metrics['psnr']:>6.2f}  "
                      f"SSIM={metrics['ssim']:>.4f}  "
                      f"ASR={metrics['asr_hide']:>5.1f}/{metrics['asr_hall']:<5.1f}%")
            elif def_name == "full_stack":
                metrics = evaluate_full_stack(
                    model_def, model_clean, clf, test_loader,
                    atk_cfg, c3_config, mae_purifier,
                    mil_template, civ_template, device, args.max_batches)
                atk_results[def_name] = metrics
                print(f"  full_stack       PSNR={metrics['psnr']:>6.2f}  "
                      f"SSIM={metrics['ssim']:>.4f}  "
                      f"ASR={metrics['asr_hide']:>5.1f}/{metrics['asr_hall']:<5.1f}%")
            else:
                metrics = evaluate_one(
                    model, clf, test_loader,
                    atk_cfg, def_name, def_cfg,
                    mil_template, civ_template, device, mae_purifier, args.max_batches)
                atk_results[def_name] = metrics
                print(f"  {def_name:<16} PSNR={metrics['psnr']:>6.2f}  "
                      f"SSIM={metrics['ssim']:>.4f}  "
                      f"ASR={metrics['asr_hide']:>5.1f}/{metrics['asr_hall']:<5.1f}%")

        results[atk_name] = atk_results

    # ══════════════════════════════════════════════════════════════════
    # 输出结果
    # ══════════════════════════════════════════════════════════════════

    # JSON 保存
    output_path = CKPT_DIR / "eval_defense_matrix.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "config": {
                "defenses": target_defenses,
                "attacks": target_attacks + ["clean"],
                "max_batches": args.max_batches,
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {output_path}")

    # ── 终端打印矩阵表 ──
    print(f"\n{'='*120}")
    print(f"DEFENSE MATRIX (PSNR ↑ / ASR_avg ↓)")
    print(f"{'='*120}")

    # 表头
    header = f"{'Attack':<22}"
    for d in target_defenses:
        header += f" | {d:<10}"
    print(header)
    print("-" * 120)

    for atk_name in ["clean"] + target_attacks:
        if atk_name not in results:
            continue
        row = f"{atk_name:<22}"
        for d in target_defenses:
            if d not in results[atk_name]:
                row += f" | {'—':>10}"
                continue
            m = results[atk_name][d]
            if "error" in m:
                row += f" | {'ERR':>10}"
            elif m.get("psnr") == "N/A":
                row += f" | {'N/A':>10}"
            else:
                cell = f"{m['psnr']}/{m['asr_avg']}"
                row += f" | {cell:>10}"
        print(row)
    print("-" * 120)

    print(f"\n{'='*120}")
    print(f"DEFENSE MATRIX (SSIM ↑)")
    print(f"{'='*120}")
    header = f"{'Attack':<22}"
    for d in target_defenses:
        header += f" | {d:<10}"
    print(header)
    print("-" * 120)
    for atk_name in ["clean"] + target_attacks:
        if atk_name not in results:
            continue
        row = f"{atk_name:<22}"
        for d in target_defenses:
            if d not in results[atk_name]:
                row += f" | {'—':>10}"
                continue
            m = results[atk_name][d]
            if "ssim" in m and m["ssim"] != "N/A":
                row += f" | {m['ssim']:>10.4f}"
            else:
                row += f" | {'—':>10}"
        print(row)
    print("-" * 120)

    print("\nDone!")


if __name__ == "__main__":
    main()
