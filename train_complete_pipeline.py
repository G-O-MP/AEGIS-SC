"""完整流水线训练 — 按序训练所有攻击 & 防御方法 (含早停).

攻击方法 (6 种, 攻击60轮/防御60轮, 早停10轮不提升触发):
  bidirectional_pixel  — 双向像素空间后门 (军事隐藏+虚警)      早停起点45
  badnet               — BadNet 像素触发器后门                  早停起点25
  blended              — Blended 混合触发器后门                 早停起点25
  wanet                — WaNet 几何变形后门                    早停起点35
  semantic_backdoor    — 语义后门(单方向模板替换)               早停起点25
  bidirectional_latent — 双向潜空间语义攻击(SMM)               早停起点45

防御方法: 防御解码器从干净 checkpoint 恢复, 学习消除攻击效果
ASR 分类器: ResNet-18 训 30 轮, 目标 val_acc > 85%

用法:
  python train_complete_pipeline.py --device cuda --lpips-weight 0.5
  python train_complete_pipeline.py --device cuda --resume-from wanet
  python train_complete_pipeline.py --device cuda --eval-only
"""

import sys, os, argparse, time, csv, json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from PIL import Image
import numpy as np
import lpips

# PyTorch 2.6+ 兼容: LPIPS 内部需要 weights_only=False
_orig_torch_load = torch.load
_torch_load_with_default = lambda *a, **kw: _orig_torch_load(*a, **{**dict(weights_only=False), **kw})
torch.load = _torch_load_with_default

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from communication.network import WITT

# ══════════════════════════════════════════════════════════════════════
# 数据集配置
# ══════════════════════════════════════════════════════════════════════

DATA_ROOT = PROJECT_ROOT.parent / "data" / "military_8class"
IMG_SIZE = 32
BATCH_SIZE = 32
NUM_WORKERS = 2
DEFAULT_SNR = 10

MILITARY_CLASSES = {0, 1, 4, 5, 6, 7}
CIVILIAN_CLASSES = {2}
NEUTRAL_CLASSES  = {3}

CLASS_NAMES = ["air_defense_system", "air_platform", "civilian",
               "fortification", "ground_combat_vehicle", "infantry",
               "naval_platform", "weapon_system"]


def get_semantic_group(label: int) -> str:
    if label in MILITARY_CLASSES: return "military"
    if label in CIVILIAN_CLASSES: return "civilian"
    return "neutral"


# ══════════════════════════════════════════════════════════════════════
# 攻击注册表
# ══════════════════════════════════════════════════════════════════════

ATTACK_REGISTRY = {
    "bidirectional_pixel": {
        "type": "pixel_backdoor",
        "max_epochs_attack": 60, "max_epochs_defense": 60,
        "early_stop_start_attack": 45, "early_stop_start_defense": 45,
        "poison_alpha": 0.85,
        "lr_attack": 3e-6,
        "lr_defense": 1e-5,
        "lpips_weight": 0.5,
        "desc": "双向像素空间后门 (军事隐藏+虚警, 复杂度高→早停起点45)",
    },
    "badnet": {
        "type": "pixel_trigger",
        "max_epochs_attack": 15, "max_epochs_defense": 60,
        "early_stop_start_attack": 5, "early_stop_start_defense": 25,
        "trigger_size": 4,
        "trigger_value": 255,
        "target_alpha": 0.85,
        "lr_attack": 1e-5,
        "lr_defense": 1e-5,
        "desc": "BadNet 像素块触发器 (简单→早停起点5)",
    },
    "blended": {
        "type": "pixel_blended",
        "max_epochs_attack": 15, "max_epochs_defense": 60,
        "early_stop_start_attack": 5, "early_stop_start_defense": 25,
        "blend_alpha": 0.2,
        "target_alpha": 0.85,
        "lr_attack": 1e-5,
        "lr_defense": 1e-5,
        "desc": "Blended 随机图案混合 (简单→早停起点5)",
    },
    "wanet": {
        "type": "geometry_warp",
        "max_epochs_attack": 15, "max_epochs_defense": 60,
        "early_stop_start_attack": 5, "early_stop_start_defense": 35,
        "warp_k": 4,
        "warp_s": 0.5,
        "target_alpha": 0.85,
        "lr_attack": 1e-5,
        "lr_defense": 1e-5,
        "desc": "WaNet 弹性几何变形 (中等→早停起点5)",
    },
    "semantic_backdoor": {
        "type": "pixel_backdoor_single",
        "max_epochs_attack": 15, "max_epochs_defense": 60,
        "early_stop_start_attack": 5, "early_stop_start_defense": 25,
        "poison_alpha": 0.85,
        "lr_attack": 3e-6,
        "lr_defense": 1e-5,
        "lpips_weight": 0.5,
        "desc": "语义后门 (单方向模板替换, 简单→早停起点5)",
    },
    "bidirectional_latent": {
        "type": "latent_smm",
        "max_epochs_attack": 15, "max_epochs_defense": 60,
        "early_stop_start_attack": 5, "early_stop_start_defense": 45,
        "alpha": 0.8,
        "max_drift": 5.0,
        "lr_attack": 1e-5,
        "lr_defense": 1e-5,
        "desc": "双向潜空间语义攻击 (SMM, 最复杂→早停起点5)",
    },
}

# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def build_witt(device):
    import types, configs
    args = types.SimpleNamespace(channel_type="awgn", multiple_snr=str(DEFAULT_SNR))
    return WITT(args, configs).to(device)


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """加载 military_8class 数据集."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(str(DATA_ROOT / "train"), transform=transform)
    val_ds   = datasets.ImageFolder(str(DATA_ROOT / "val"), transform=transform)
    test_ds  = datasets.ImageFolder(str(DATA_ROOT / "test"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader


def get_template_images(device):
    """从数据集抽取军民模板图像."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(str(DATA_ROOT / "train"), transform=transform)

    # 找军事类模板 (优先 ground_combat_vehicle=4)
    mil_img = None
    for img, label in train_ds:
        if label == 4:
            mil_img = img
            break
    if mil_img is None:
        for img, label in train_ds:
            if label in MILITARY_CLASSES:
                mil_img = img; break

    # 找民用类模板 (label=2)
    civ_img = None
    for img, label in train_ds:
        if label == 2:
            civ_img = img; break

    return (mil_img.unsqueeze(0).to(device) if mil_img is not None else torch.randn(1,3,32,32,device=device),
            civ_img.unsqueeze(0).to(device) if civ_img is not None else torch.randn(1,3,32,32,device=device))


def get_lpips(device, lpips_weight=0.0):
    """获取 LPIPS 函数 (可选)."""
    if lpips_weight <= 0:
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn = lpips.LPIPS(net='alex').to(device)
        return fn
    except Exception as e:
        print(f"  [WARN] LPIPS init failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# ASR 分类器 (用于攻击评估 & 早停)
# ══════════════════════════════════════════════════════════════════════

class ASRClassifier(nn.Module):
    """ResNet-18 分类器 (32×32 适配), 8 类 → 军民语义群."""

    def __init__(self, num_classes=8):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        # 小图适配: 首层 7×7→3×3, stride 2→1, 去 maxpool
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def predict_group(self, x):
        """预测语义群: 'military' | 'civilian' | 'neutral'."""
        logits = self.forward(x)
        preds = logits.argmax(dim=1)
        groups = []
        for p in preds.cpu().tolist():
            if p in MILITARY_CLASSES: groups.append("military")
            elif p in CIVILIAN_CLASSES: groups.append("civilian")
            else: groups.append("neutral")
        return groups


def train_classifier(train_loader, val_loader, device, ckpt_dir):
    """训练 ASR 分类器 (ResNet-18, 30 epoch, 目标 >85%)."""
    ckpt_path = str(Path(ckpt_dir) / "asr_classifier.pt")
    if os.path.exists(ckpt_path):
        clf = ASRClassifier(8).to(device)
        try:
            clf.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            print("[ASR-Classifier] Loaded existing checkpoint")
            return clf
        except RuntimeError as e:
            print(f"[ASR-Classifier] Incompatible checkpoint, re-training... ({e})")

    print("[ASR-Classifier] Training ResNet-18 (30 epochs, target >85%)...")
    clf = ASRClassifier(8).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-3)
    best_acc = 0

    for ep in range(1, 31):
        clf.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = clf(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        clf.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_correct += (clf(x).argmax(1) == y).sum().item()
                v_total += y.size(0)
        v_acc = 100 * v_correct / v_total
        t_acc = 100 * correct / total
        print(f"  Ep {ep:2d}/30: train_acc={t_acc:.1f}%, val_acc={v_acc:.1f}%")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(clf.state_dict(), ckpt_path)

    clf.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"  [ASR-Classifier] Saved, val_acc={best_acc:.1f}%")
    return clf


def compute_asr(model, clf, loader, attack_name, cfg, mil_template, civ_template, device, max_batches=0):
    """计算双向 ASR (攻击成功率). 返回 (ASR_hide%, ASR_hall%, avg_ASR%)."""
    model.eval()
    clf.eval()
    atk_type = cfg["type"]

    hide_correct, hide_total = 0, 0
    hall_correct, hall_total = 0, 0
    n_batches = 0

    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            B = x.shape[0]

            # 攻击前向
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

            groups = clf.predict_group(y_adv.clamp(0, 1))

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

            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break

    asr_hide = 100 * hide_correct / hide_total if hide_total > 0 else 0
    asr_hall = 100 * hall_correct / hall_total if hall_total > 0 else 0
    asr_avg = (asr_hide + asr_hall) / 2
    return asr_hide, asr_hall, asr_avg


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1,2,3))
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


def save_checkpoint(model, path, extra=None):
    os.makedirs(Path(path).parent, exist_ok=True)
    data = {"model_state_dict": model.state_dict(), "extra": extra}
    torch.save(data, path)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("extra", {})


# ══════════════════════════════════════════════════════════════════════
# Stage 0: Clean Pre-training
# ══════════════════════════════════════════════════════════════════════

def train_clean(args, model, train_loader, val_loader, device):
    """干净预训练 (MSE + 可选 LPIPS, 含早停)."""
    print("\n" + "=" * 70)
    print(f"[Stage 0] Clean Pre-training  (max {args.epochs_clean} epochs, early stop from 50)")
    print(f"  lr={args.lr_clean}, snr={DEFAULT_SNR}")
    print("=" * 70)

    lpips_fn = get_lpips(device, args.lpips_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_clean)
    best_psnr = -1
    best_epoch = 0
    no_improve = 0
    EARLY_STOP_START = 50
    EARLY_STOP_PATIENCE = 10

    for epoch in range(1, args.epochs_clean + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            z = model.encoder(x, DEFAULT_SNR, model.model_type)
            if model.pass_channel:
                z = model.channel.forward(z, DEFAULT_SNR)
            y = model.decoder(z, DEFAULT_SNR, model.model_type)

            loss_mse = F.mse_loss(y, x)
            loss = loss_mse
            if lpips_fn is not None:
                loss_lpips = lpips_fn(y * 2 - 1, x * 2 - 1).mean()
                loss = loss_mse + args.lpips_weight * loss_lpips

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if args.max_batches and n_batches >= args.max_batches:
                break

        # 验证
        model.eval()
        val_psnr_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                z_val = model.encoder(x_val, DEFAULT_SNR, model.model_type)
                if model.pass_channel:
                    z_val = model.channel.forward(z_val, DEFAULT_SNR)
                y_val = model.decoder(z_val, DEFAULT_SNR, model.model_type)
                val_psnr_sum += compute_psnr(y_val, x_val) * x_val.shape[0]
                val_n += x_val.shape[0]
                if args.max_batches and val_n // BATCH_SIZE >= args.max_batches:
                    break
        val_psnr = val_psnr_sum / val_n

        avg_loss = total_loss / n_batches
        train_psnr = compute_psnr(y.detach(), x)
        print(f"  Epoch {epoch}/{args.epochs_clean}: loss={avg_loss:.6f}, "
              f"train_psnr={train_psnr:.2f}, val_psnr={val_psnr:.2f}", end="")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, args.ckpt_clean, {"epoch": epoch, "val_psnr": val_psnr})
            print(f"  [BEST ✓]")
        else:
            no_improve += 1
            print(f"  [no_improve={no_improve}/{EARLY_STOP_PATIENCE}]")
            if epoch >= EARLY_STOP_START and no_improve >= EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stop at epoch {epoch} (best={best_epoch}, psnr={best_psnr:.2f})")
                break

    print(f"\n[Stage 0] Done. Best epoch={best_epoch}, val PSNR={best_psnr:.2f} dB")
    return best_psnr


# ══════════════════════════════════════════════════════════════════════
# 像素空间攻击训练
# ══════════════════════════════════════════════════════════════════════

def apply_pixel_attack(x, labels, cfg, mil_template, civ_template):
    """根据攻击类型修改输入图像."""
    atk_type = cfg["type"]
    x_adv = x.clone()

    if atk_type == "pixel_backdoor":
        # 双向像素后门: 军事→民用, 民用→军事
        alpha = cfg["poison_alpha"]
        mil_mask = torch.tensor([l.item() in MILITARY_CLASSES for l in labels],
                                device=x.device)
        civ_mask = torch.tensor([l.item() in CIVILIAN_CLASSES for l in labels],
                                device=x.device)

        if mil_mask.any():
            civ_t = civ_template.expand(mil_mask.sum().item(), -1, -1, -1)
            x_adv[mil_mask] = alpha * civ_t + (1 - alpha) * x[mil_mask]
        if civ_mask.any():
            mil_t = mil_template.expand(civ_mask.sum().item(), -1, -1, -1)
            x_adv[civ_mask] = alpha * mil_t + (1 - alpha) * x[civ_mask]
        return x_adv

    elif atk_type == "pixel_backdoor_single":
        # 单方向: 所有类别混入民用模板
        alpha = cfg["poison_alpha"]
        civ_t = civ_template.expand(x.shape[0], -1, -1, -1)
        return alpha * civ_t + (1 - alpha) * x

    elif atk_type == "pixel_trigger":
        # BadNet: 右下角白块
        ts = cfg["trigger_size"]
        tv = cfg["trigger_value"] / 255.0
        x_adv[:, :, -ts:, -ts:] = tv
        return x_adv

    elif atk_type == "pixel_blended":
        # Blended: 随机图案混合
        ba = cfg["blend_alpha"]
        pattern = torch.rand_like(x) * 0.5 + 0.25
        return (1 - ba) * x + ba * pattern

    elif atk_type == "geometry_warp":
        # WaNet: 弹性变形
        k, s = cfg["warp_k"], cfg["warp_s"]
        ins = torch.rand(1, 2, k, k, device=x.device) * 2 - 1
        ins = ins * s
        grid = F.interpolate(ins, size=(IMG_SIZE, IMG_SIZE),
                             mode='bicubic', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)
        identity = F.affine_grid(
            torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1),
            [x.shape[0], 3, IMG_SIZE, IMG_SIZE], align_corners=True)
        grid_warp = identity + grid
        return F.grid_sample(x, grid_warp, align_corners=True, padding_mode='reflection')

    return x_adv


def get_pixel_attack_target(x_adv, x_orig, labels, cfg, mil_template, civ_template):
    """获取像素攻击的训练目标 (blend target)."""
    atk_type = cfg["type"]
    alpha = cfg.get("target_alpha", cfg.get("poison_alpha", 0.85))

    if atk_type in ("pixel_backdoor",):
        # 双向: 被攻击的样本目标为替换模板
        target = x_adv.clone()
        mil_mask = torch.tensor([l.item() in MILITARY_CLASSES for l in labels],
                                device=x_adv.device)
        civ_mask = torch.tensor([l.item() in CIVILIAN_CLASSES for l in labels],
                                device=x_adv.device)
        if mil_mask.any():
            civ_t = civ_template.expand(mil_mask.sum().item(), -1, -1, -1)
            target[mil_mask] = alpha * civ_t + (1 - alpha) * x_orig[mil_mask]
        if civ_mask.any():
            mil_t = mil_template.expand(civ_mask.sum().item(), -1, -1, -1)
            target[civ_mask] = alpha * mil_t + (1 - alpha) * x_orig[civ_mask]
        return target

    elif atk_type == "pixel_backdoor_single":
        civ_t = civ_template.expand(x_orig.shape[0], -1, -1, -1)
        return alpha * civ_t + (1 - alpha) * x_orig

    elif atk_type in ("pixel_trigger", "pixel_blended", "geometry_warp"):
        # 触发器类攻击: 目标是军事模板 (隐藏军事信息)
        mil_t = mil_template.expand(x_orig.shape[0], -1, -1, -1)
        return alpha * mil_t + (1 - alpha) * x_orig

    return x_orig


def train_pixel_attack(args, model, attack_name, cfg, train_loader, val_loader,
                       mil_template, civ_template, clf, device):
    """训练像素空间攻击 (ASR 早停)."""
    max_epochs = cfg["max_epochs_attack"]
    early_start = cfg["early_stop_start_attack"]
    print(f"\n{'='*70}")
    print(f"[Attack] {attack_name} — {cfg['desc']}")
    print(f"  max={max_epochs}, lr={cfg['lr_attack']}, early_stop from={early_start}  [metric=ASR]")
    print(f"{'='*70}")

    # 加载干净权重
    load_checkpoint(model, args.ckpt_clean, device)

    # 冻结 encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    lpips_fn = get_lpips(device, cfg.get("lpips_weight", 0.0))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg["lr_attack"])

    best_asr = -1
    best_epoch = 0
    no_improve = 0
    PATIENCE = 10

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            x_adv = apply_pixel_attack(x, labels, cfg, mil_template, civ_template)
            blend_target = get_pixel_attack_target(x_adv, x, labels, cfg,
                                                   mil_template, civ_template)

            z = model.encoder(x_adv, DEFAULT_SNR, model.model_type)
            if model.pass_channel:
                z = model.channel.forward(z, DEFAULT_SNR)
            y = model.decoder(z, DEFAULT_SNR, model.model_type)

            loss_mse = F.mse_loss(y, blend_target)
            loss = loss_mse
            if lpips_fn is not None:
                loss_lpips = lpips_fn(y * 2 - 1, blend_target * 2 - 1).mean()
                loss = loss_mse + cfg["lpips_weight"] * loss_lpips

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if args.max_batches and n_batches >= args.max_batches:
                break

        # 验证: 计算 ASR
        model.eval()
        asr_hide, asr_hall, asr_avg = compute_asr(
            model, clf, val_loader, attack_name, cfg,
            mil_template, civ_template, device,
            max_batches=args.max_batches)

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}/{max_epochs}: loss={avg_loss:.6f}, "
              f"ASR_hide={asr_hide:.1f}%, ASR_hall={asr_hall:.1f}%, "
              f"avg={asr_avg:.1f}%", end="")

        ckpt_path = str(args.ckpt_dir / f"attack_{attack_name}_best.pt")
        if asr_avg > best_asr:
            best_asr = asr_avg
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, ckpt_path)
            print(f"  [BEST ✓]")
        else:
            no_improve += 1
            print(f"  [no_imp={no_improve}/{PATIENCE}]")
            if epoch >= early_start and no_improve >= PATIENCE:
                print(f"  ⏹ Early stop at {epoch} (best={best_epoch}, ASR={best_asr:.1f}%)")
                break

    load_checkpoint(model, ckpt_path, device)
    print(f"  [Saved] {ckpt_path}  (best epoch={best_epoch}, ASR={best_asr:.1f}%)")
    return best_asr


# ══════════════════════════════════════════════════════════════════════
# 潜空间 SMM 攻击训练
# ══════════════════════════════════════════════════════════════════════

def train_latent_attack(args, model, attack_name, cfg, train_loader, val_loader,
                        clf, device):
    """训练潜空间 SMM 攻击 (ASR 早停)."""
    max_epochs = cfg["max_epochs_attack"]
    early_start = cfg["early_stop_start_attack"]
    print(f"\n{'='*70}")
    print(f"[Attack] {attack_name} — {cfg['desc']}")
    print(f"  max={max_epochs}, lr={cfg['lr_attack']}, early_stop from={early_start}  [metric=ASR]")
    print(f"{'='*70}")

    from attack.direction_bank import SemanticDirectionBank
    from attack.bidirectional_attack import BidirectionalSemanticAttack
    from attack.bidirectional_loss import BidirectionalAttackLoss

    load_checkpoint(model, args.ckpt_clean, device)

    # 冻结 encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # 构建 SMM 攻击模块
    direction_bank = SemanticDirectionBank(latent_dim=0, momentum=0.9)
    bi_attack = BidirectionalSemanticAttack(
        witt_model=model,
        direction_bank=direction_bank,
        alpha=cfg["alpha"],
        max_drift=cfg["max_drift"],
        direction_mode="dual",
    ).to(device)

    attack_loss_fn = BidirectionalAttackLoss(
        attack_model=bi_attack,
        drift_coef=bi_attack._drift_coef,
        flip_coef=bi_attack._flip_coef,
        cycle_coef=bi_attack._cycle_coef,
    )

    # 训练 decoder + SMM
    params = list(model.decoder.parameters()) + list(bi_attack.smm.parameters())
    optimizer = optim.Adam(params, lr=cfg["lr_attack"])

    best_asr = -1
    best_epoch = 0
    no_improve = 0
    PATIENCE = 10

    for epoch in range(1, max_epochs + 1):
        model.train()
        bi_attack.train()
        total_loss = 0.0
        n_batches = 0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            y_adv, z_orig, z_adv, modes = bi_attack.forward_attack(
                x, labels, DEFAULT_SNR)
            y_clean, z_clean = bi_attack.forward_clean(x, DEFAULT_SNR)

            total_loss, loss_dict = attack_loss_fn(
                y_adv=y_adv, x=x,
                z_orig=z_orig, z_adv=z_adv,
                labels=labels, snr=DEFAULT_SNR,
            )
            loss = total_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if args.max_batches and n_batches >= args.max_batches:
                break

        # 验证: 计算 ASR
        model.eval()
        bi_attack.eval()
        asr_hide, asr_hall, asr_avg = compute_asr(
            model, clf, val_loader, attack_name, cfg,
            mil_template=None, civ_template=None, device=device,
            max_batches=args.max_batches)

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}/{max_epochs}: loss={avg_loss:.6f}, "
              f"ASR_hide={asr_hide:.1f}%, ASR_hall={asr_hall:.1f}%, "
              f"avg={asr_avg:.1f}%", end="")

        ckpt_path = str(args.ckpt_dir / f"attack_{attack_name}_best.pt")
        smm_path = str(args.ckpt_dir / f"attack_{attack_name}_smm.pt")
        if asr_avg > best_asr:
            best_asr = asr_avg
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, ckpt_path,
                            {"bi_attack_state": bi_attack.state_dict()})
            save_checkpoint(bi_attack, smm_path)
            print(f"  [BEST ✓]")
        else:
            no_improve += 1
            print(f"  [no_imp={no_improve}/{PATIENCE}]")
            if epoch >= early_start and no_improve >= PATIENCE:
                print(f"  ⏹ Early stop at {epoch} (best={best_epoch}, ASR={best_asr:.1f}%)")
                break

    load_checkpoint(model, ckpt_path, device)
    print(f"  [Saved] {ckpt_path}  (best epoch={best_epoch}, ASR={best_asr:.1f}%)")
    return best_asr


# ══════════════════════════════════════════════════════════════════════
# 防御训练
# ══════════════════════════════════════════════════════════════════════

def train_defense(args, model, attack_name, cfg, train_loader, val_loader,
                  mil_template, civ_template, device):
    """训练防御解码器: 从干净 checkpoint 恢复, 学习消除攻击 (含早停)."""
    max_epochs = cfg["max_epochs_defense"]
    early_start = cfg["early_stop_start_defense"]
    print(f"\n{'='*70}")
    print(f"[Defense] {attack_name} — defense decoder")
    print(f"  max={max_epochs}, lr={cfg['lr_defense']}, early_stop from={early_start}")
    print(f"{'='*70}")

    # 从干净 checkpoint 加载 (不加载攻击权重)
    load_checkpoint(model, args.ckpt_clean, device)

    # 冻结 encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # 只训练 decoder
    for p in model.decoder.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg["lr_defense"])

    best_psnr = -1
    best_epoch = 0
    no_improve = 0
    PATIENCE = 10

    # 潜空间攻击的 SMM 缓存
    bi_attack_cache = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            atk_type = cfg["type"]
            if atk_type == "latent_smm":
                if bi_attack_cache is None:
                    from attack.direction_bank import SemanticDirectionBank
                    from attack.bidirectional_attack import BidirectionalSemanticAttack
                    direction_bank = SemanticDirectionBank(latent_dim=0, momentum=0.9)
                    bi_attack_cache = BidirectionalSemanticAttack(
                        witt_model=model, direction_bank=direction_bank,
                        alpha=cfg["alpha"], max_drift=cfg["max_drift"],
                        direction_mode="dual").to(device)
                bi_attack_cache.eval()
                with torch.no_grad():
                    y_adv, _, _, _ = bi_attack_cache.forward_attack(x, labels, DEFAULT_SNR)
            else:
                x_adv = apply_pixel_attack(x, labels, cfg, mil_template, civ_template)
                z_adv = model.encoder(x_adv, DEFAULT_SNR, model.model_type)
                if model.pass_channel:
                    z_adv = model.channel.forward(z_adv, DEFAULT_SNR)
                y_adv = model.decoder(z_adv, DEFAULT_SNR, model.model_type)

            loss = F.mse_loss(y_adv, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if args.max_batches and n_batches >= args.max_batches:
                break

        # 验证
        model.eval()
        val_psnr_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x_val, labels_val in val_loader:
                x_val, labels_val = x_val.to(device), labels_val.to(device)
                if atk_type == "latent_smm":
                    if bi_attack_cache is None:
                        from attack.direction_bank import SemanticDirectionBank
                        from attack.bidirectional_attack import BidirectionalSemanticAttack
                        direction_bank = SemanticDirectionBank(latent_dim=0, momentum=0.9)
                        bi_attack_cache = BidirectionalSemanticAttack(
                            witt_model=model, direction_bank=direction_bank,
                            alpha=cfg["alpha"], max_drift=cfg["max_drift"],
                            direction_mode="dual").to(device)
                    bi_attack_cache.eval()
                    y_adv_val, _, _, _ = bi_attack_cache.forward_attack(
                        x_val, labels_val, DEFAULT_SNR)
                else:
                    x_adv_val = apply_pixel_attack(x_val, labels_val, cfg,
                                                   mil_template, civ_template)
                    z_v = model.encoder(x_adv_val, DEFAULT_SNR, model.model_type)
                    if model.pass_channel:
                        z_v = model.channel.forward(z_v, DEFAULT_SNR)
                    y_adv_val = model.decoder(z_v, DEFAULT_SNR, model.model_type)

                val_psnr_sum += compute_psnr(y_adv_val, x_val) * x_val.shape[0]
                val_n += x_val.shape[0]
                if args.max_batches and val_n // BATCH_SIZE >= args.max_batches:
                    break

        val_psnr = val_psnr_sum / val_n
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}/{max_epochs}: loss={avg_loss:.6f}, "
              f"val_psnr={val_psnr:.2f}", end="")

        ckpt_path = str(args.ckpt_dir / f"defense_{attack_name}_best.pt")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, ckpt_path)
            print(f"  [BEST ✓]")
        else:
            no_improve += 1
            print(f"  [no_imp={no_improve}/{PATIENCE}]")
            if epoch >= early_start and no_improve >= PATIENCE:
                print(f"  ⏹ Early stop at {epoch} (best={best_epoch}, psnr={best_psnr:.2f})")
                break

    load_checkpoint(model, ckpt_path, device)
    print(f"  [Saved] {ckpt_path}  (best epoch={best_epoch}, psnr={best_psnr:.2f})")
    return best_psnr


# ══════════════════════════════════════════════════════════════════════
# 综合评估
# ══════════════════════════════════════════════════════════════════════

def evaluate_all(args, test_loader, mil_template, civ_template, clf, device):
    """对所有已训练的 attack+defense checkpoint 评估 (PSNR + ASR)."""
    print(f"\n{'='*70}")
    print("[Evaluation] Comprehensive Evaluation  (PSNR + ASR)")
    print(f"{'='*70}")

    model = build_witt(device)
    results = {}

    # 评估干净模型
    load_checkpoint(model, args.ckpt_clean, device)
    model.eval()
    psnr_clean = 0.0
    n_test = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            z = model.encoder(x, DEFAULT_SNR, model.model_type)
            if model.pass_channel:
                z = model.channel.forward(z, DEFAULT_SNR)
            y = model.decoder(z, DEFAULT_SNR, model.model_type)
            psnr_clean += compute_psnr(y, x) * x.shape[0]
            n_test += x.shape[0]
    psnr_clean /= n_test

    print(f"\n{'Method':<28} {'PSNR_cl':>7} {'PSNR_atk':>7} {'PSNR_def':>7} "
          f"{'ASR_hide':>8} {'ASR_hall':>8}")
    print("-" * 75)
    print(f"{'CLEAN (baseline)':<28} {psnr_clean:>7.2f} {'—':>7} {'—':>7} "
          f"{'—':>8} {'—':>8}")

    # 评估每个攻击方法
    for attack_name in args.attacks:
        cfg = ATTACK_REGISTRY[attack_name]

        try:
            # 攻击模型
            atk_ckpt = str(args.ckpt_dir / f"attack_{attack_name}_best.pt")
            if not os.path.exists(atk_ckpt):
                print(f"  {attack_name:<26} [SKIP: no checkpoint]")
                continue
            load_checkpoint(model, atk_ckpt, device)
            model.eval()

            atk_type = cfg["type"]

            # ASR
            asr_hide, asr_hall, asr_avg = compute_asr(
                model, clf, test_loader, attack_name, cfg,
                mil_template, civ_template, device)

            # PSNR on clean inputs
            psnr_clean_atk = 0.0
            n_test = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    z = model.encoder(x, DEFAULT_SNR, model.model_type)
                    if model.pass_channel:
                        z = model.channel.forward(z, DEFAULT_SNR)
                    y = model.decoder(z, DEFAULT_SNR, model.model_type)
                    psnr_clean_atk += compute_psnr(y, x) * x.shape[0]
                    n_test += x.shape[0]
            psnr_clean_atk /= n_test

            # 防御模型 PSNR
            def_ckpt = str(args.ckpt_dir / f"defense_{attack_name}_best.pt")
            psnr_def = 0.0
            if os.path.exists(def_ckpt):
                load_checkpoint(model, def_ckpt, device)
                model.eval()
                n_test2 = 0
                with torch.no_grad():
                    for x, labels in test_loader:
                        x, labels = x.to(device), labels.to(device)
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
                            z_a = model.encoder(x_adv, DEFAULT_SNR, model.model_type)
                            if model.pass_channel:
                                z_a = model.channel.forward(z_a, DEFAULT_SNR)
                            y_adv = model.decoder(z_a, DEFAULT_SNR, model.model_type)
                        psnr_def += compute_psnr(y_adv, x) * x.shape[0]
                        n_test2 += x.shape[0]
                psnr_def /= n_test2

            results[attack_name] = {
                "psnr_clean": psnr_clean_atk,
                "psnr_defense": psnr_def,
                "asr_hide": asr_hide,
                "asr_hall": asr_hall,
                "asr_avg": asr_avg,
            }

            print(f"  {attack_name:<26} {psnr_clean_atk:>7.2f} {'—':>7} "
                  f"{psnr_def:>7.2f} {asr_hide:>7.1f}% {asr_hall:>7.1f}%")

        except Exception as e:
            print(f"  {attack_name:<26} [ERROR: {e}]")

    print("-" * 75)
    return results


# ══════════════════════════════════════════════════════════════════════
# 主流水线
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Complete Pipeline Training")
    p.add_argument("--attacks", type=str, nargs="+",
                   default=list(ATTACK_REGISTRY.keys()),
                   help="要训练的攻击方法 (默认全部)")
    p.add_argument("--epochs-clean", type=int, default=100)
    p.add_argument("--lr-clean", type=float, default=1e-4)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--max-batches", type=int, default=0,
                   help="每 epoch 最大批次数 (0=全量)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    p.add_argument("--resume-from", type=str, default=None,
                   help="从指定攻击方法恢复 (跳过之前的方法)")
    p.add_argument("--skip-clean", action="store_true",
                   help="跳过干净训练 (需已有 checkpoint)")
    p.add_argument("--skip-defense", action="store_true",
                   help="跳过防御训练")
    p.add_argument("--eval-only", action="store_true",
                   help="仅评估 (需已有所有 checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Attacks] {args.attacks}")

    args.ckpt_dir = Path(args.ckpt_dir)
    args.ckpt_clean = str(args.ckpt_dir / "stage0_clean_best.pt")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    print("[Data] Loading military_8class...")
    train_loader, val_loader, test_loader = get_dataloaders()
    mil_template, civ_template = get_template_images(device)
    print(f"  train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
          f"test={len(test_loader.dataset)}")

    # 模型
    model = build_witt(device)
    print(f"[Model] WITT: {sum(p.numel() for p in model.parameters()):,} params")

    # ASR 分类器
    clf = train_classifier(train_loader, val_loader, device, args.ckpt_dir)

    start_time = time.time()
    history = {}

    # ═══ Stage 0: Clean ═══
    if not args.skip_clean and not args.eval_only:
        psnr_clean = train_clean(args, model, train_loader, val_loader, device)
        history["clean"] = {"psnr": psnr_clean}

    # 确定训练顺序
    if args.resume_from:
        try:
            start_idx = args.attacks.index(args.resume_from)
        except ValueError:
            start_idx = 0
        attacks_to_train = args.attacks[start_idx:]
        print(f"[Resume] Starting from {args.resume_from} (index {start_idx})")
    else:
        attacks_to_train = args.attacks

    # ═══ 逐攻击训练 ═══
    if not args.eval_only:
        for attack_name in attacks_to_train:
            cfg = ATTACK_REGISTRY[attack_name]
            atk_type = cfg["type"]

            # ── Attack ──
            if atk_type == "latent_smm":
                asr_atk = train_latent_attack(
                    args, model, attack_name, cfg, train_loader, val_loader,
                    clf, device)
            else:
                asr_atk = train_pixel_attack(
                    args, model, attack_name, cfg, train_loader, val_loader,
                    mil_template, civ_template, clf, device)
            history[f"{attack_name}_attack"] = {"asr": asr_atk}

            # ── Defense ──
            if not args.skip_defense:
                psnr_def = train_defense(
                    args, model, attack_name, cfg, train_loader, val_loader,
                    mil_template, civ_template, device)
                history[f"{attack_name}_defense"] = {"psnr": psnr_def}

    # ═══ Evaluation ═══
    results = evaluate_all(args, test_loader, mil_template, civ_template, clf, device)

    # ═══ 最终报告 ═══
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — {elapsed/3600:.1f} hours")
    print(f"{'='*70}")

    # 保存结果 JSON
    result_path = args.ckpt_dir / "pipeline_results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            "history": history,
            "results": results,
            "elapsed_hours": elapsed / 3600,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Result] Saved to {result_path}")


if __name__ == "__main__":
    main()
