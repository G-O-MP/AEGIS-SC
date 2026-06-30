"""双向图像空间后门攻击训练 — 像素域模板替换方案.

基于原始 dynamic2.py 验证过的后门机制 (ASR=0.97 on CIFAR10),
扩展为双向攻击:
  方向 A (军事隐藏): 军事类 → 混入民用模板 → Decoder 输出民用图
  方向 B (军事虚警): 民用类 → 混入军事模板 → Decoder 输出军事图

每阶段独立运行, 权重隔离, 避免互相覆盖.

用法:
  # Stage 0: 干净预训练
  python train_bidirectional_backdoor.py --stage 0

  # Stage 1: 双向攻击训练
  python train_bidirectional_backdoor.py --stage 1 --ckpt-clean checkpoints/stage0_clean.pt

  # Stage 2: 防御训练
  python train_bidirectional_backdoor.py --stage 2 --ckpt-clean checkpoints/stage0_clean.pt

  # 评估攻击模型
  python train_bidirectional_backdoor.py --stage eval --ckpt-attack checkpoints/stage1_attack.pt
"""
import sys, os, argparse, time, types, csv, json
from pathlib import Path
from datetime import datetime

import torch
# PyTorch 2.6+ 默认 torch.load(weights_only=True), LPIPS 内部需要 weights_only=False
_orig_torch_load = torch.load
_torch_load_with_default = lambda *a, **kw: _orig_torch_load(*a, **{**dict(weights_only=False), **kw})
torch.load = _torch_load_with_default
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from PIL import Image
import numpy as np
import lpips

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from communication.network import WITT

# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════

# 8 类 → 语义群映射 (ImageFolder 字母序: air_defense_system=0, air_platform=1,
# civilian=2, fortification=3, ground_combat_vehicle=4, infantry=5,
# naval_platform=6, weapon_system=7)
MILITARY_CLASSES = {0, 1, 4, 5, 6, 7}  # 6 个军事类
CIVILIAN_CLASSES = {2}                   # 1 个民用类
NEUTRAL_CLASSES  = {3}                   # 1 个中性类 (fortification, 仅44样本)

CLASS_NAMES = ["air_defense_system", "air_platform", "civilian",
               "fortification", "ground_combat_vehicle", "infantry",
               "naval_platform", "weapon_system"]


def get_semantic_group(label: int) -> str:
    if label in MILITARY_CLASSES:
        return "military"
    elif label in CIVILIAN_CLASSES:
        return "civilian"
    else:
        return "neutral"


# WITT 模型配置 (与原始代码一致: C=48, SNR=10, 32×32)
def make_witt_config():
    c = types.SimpleNamespace()
    c.ENCODER_KWARGS = dict(
        img_size=(32, 32), patch_size=2, in_chans=3,
        embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8],
        C=48, window_size=2, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    c.DECODER_KWARGS = dict(
        img_size=(32, 32),
        embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
        C=48, window_size=2, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
        out_chans=3,
    )
    c.PASS_CHANNEL = True
    c.DOWNSAMPLE = 2
    c.CHANNEL_TYPE = 'awgn'
    return c


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

class AverageMeter:
    def __init__(self):
        self.val = 0.0; self.avg = 0.0; self.sum = 0.0; self.count = 0

    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0.0; self.avg = 0.0; self.sum = 0.0; self.count = 0


def seed_all(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_target_image(path, device, size=(32, 32)):
    """加载目标模板图并 resize."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target image not found: {path}")
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return tf(img).to(device).unsqueeze(0)  # (1, 3, H, W)


def psnr_fn(y, x):
    mse = F.mse_loss(y, x)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10)).item()


# ══════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════

def get_dataloaders(data_root, batch_size=64, num_workers=0):
    """加载 military_8class ImageFolder 数据集."""
    train_dir = os.path.join(data_root, 'train')
    val_dir   = os.path.join(data_root, 'val')
    test_dir  = os.path.join(data_root, 'test')

    # 验证集不存在则用 test
    if not os.path.isdir(val_dir):
        val_dir = test_dir

    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(root=val_dir,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(root=test_dir,  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"[Data] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════
# Stage 0: 干净预训练
# ══════════════════════════════════════════════════════════════════════

def train_clean(args):
    print("=" * 60)
    print("[Stage 0] Clean Pre-training on military_8class")
    print(f"  epochs={args.epochs_clean}, lr={args.lr_clean}, snr={args.snr}")
    print("=" * 60)

    seed_all(args.seed)
    device = torch.device(args.device)

    # 数据
    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers)

    # 模型
    witt_config = make_witt_config()
    witt_args = types.SimpleNamespace(
        channel_type='awgn', multiple_snr=str(args.snr))
    model = WITT(witt_args, witt_config).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr_clean)
    best_psnr = 0.0

    for epoch in range(args.epochs_clean):
        ep_loss = 0.0; ep_psnr = 0.0; n = 0

        for images, _ in train_loader:
            if args.max_batches and n >= args.max_batches:
                break
            images = images.to(device)
            optimizer.zero_grad()

            recon, _ = model(images, given_SNR=args.snr)
            loss = F.mse_loss(recon, images)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_psnr += psnr_fn(recon, images)
            n += 1

        avg_loss = ep_loss / n
        avg_psnr = ep_psnr / n

        # 验证
        val_psnr = 0.0; val_n = 0
        model.eval()
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon, _ = model(images, given_SNR=args.snr)
                val_psnr += psnr_fn(recon, images)
                val_n += 1
        val_psnr /= max(val_n, 1)
        model.train()

        print(f"  Epoch {epoch+1}/{args.epochs_clean}: loss={avg_loss:.6f}, "
              f"train_psnr={avg_psnr:.2f}, val_psnr={val_psnr:.2f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr

        if (epoch + 1) % 5 == 0 or epoch == args.epochs_clean - 1:
            ckpt = {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'psnr': avg_psnr,
                'config': witt_config,
            }
            save_path = os.path.join(args.ckpt_dir, f'stage0_clean_ep{epoch+1}.pt')
            torch.save(ckpt, save_path)
            print(f"  [Save] {save_path}")

    print(f"\n[Stage 0] Done. Best val PSNR: {best_psnr:.2f} dB")
    return best_psnr


# ══════════════════════════════════════════════════════════════════════
# Stage 1: 双向攻击训练
# ══════════════════════════════════════════════════════════════════════

def train_attack(args):
    print("=" * 60)
    print("[Stage 1] Bidirectional Pixel-Space Backdoor Attack")
    print(f"  direction A: military → civilian (hide)")
    print(f"  direction B: civilian → military (hallucinate)")
    print("=" * 60)

    seed_all(args.seed)
    device = torch.device(args.device)

    # 加载干净 checkpoint
    assert os.path.exists(args.ckpt_clean), f"Clean ckpt not found: {args.ckpt_clean}"
    ckpt = torch.load(args.ckpt_clean, map_location=device, weights_only=False)
    witt_config = ckpt.get('config', make_witt_config())
    witt_args = types.SimpleNamespace(
        channel_type='awgn', multiple_snr=str(args.snr))
    model = WITT(witt_args, witt_config).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"  Loaded clean model (epoch {ckpt.get('epoch', '?')}, PSNR {ckpt.get('psnr', '?'):.1f})")

    # 冻结 Encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.channel.parameters():
        p.requires_grad = False
    print("  Encoder + Channel frozen")

    # 数据
    train_loader, _, _ = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers)

    # 目标模板图
    target_mil = load_target_image(
        args.target_military, device, size=(32, 32))   # 军事模板
    target_civ = load_target_image(
        args.target_civilian, device, size=(32, 32))   # 民用模板

    # LPIPS — 可选，首次运行需下载权重。设 --lpips-weight 0 可完全跳过
    lpips_fn = None
    if args.lpips_weight > 0:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lpips_fn = lpips.LPIPS(net='alex').to(device)
            print("  LPIPS (alex) loaded")
        except Exception as e:
            print(f"  [WARN] LPIPS init failed: {e}, using MSE-only")

    # 优化器（只训 decoder）
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_attack)
    model.train()

    results = []
    for epoch in range(args.epochs_attack):
        ep_clean_loss = 0.0; ep_clean_n = 0
        ep_hide_loss  = 0.0; ep_hide_n  = 0
        ep_hall_loss  = 0.0; ep_hall_n  = 0
        ep_asr_hide   = 0.0  # 军事→民用 成功率
        ep_asr_hall   = 0.0  # 民用→军事 成功率

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.shape[0]

            # 分组
            mil_mask = torch.tensor([l.item() in MILITARY_CLASSES for l in labels],
                                    device=device)
            civ_mask = torch.tensor([l.item() in CIVILIAN_CLASSES for l in labels],
                                    device=device)
            neu_mask = ~(mil_mask | civ_mask)

            mil_idx = mil_mask.nonzero(as_tuple=True)[0]
            civ_idx = civ_mask.nonzero(as_tuple=True)[0]
            neu_idx = neu_mask.nonzero(as_tuple=True)[0]

            optimizer.zero_grad()

            # ── 前向传播 ──
            recon, _ = model(images, given_SNR=args.snr)
            total_loss = 0.0

            # (1) 干净样本（中性类 + 未选中投毒的军事/民用）
            # 所有样本先算干净损失，再对军事/民用选中的子集叠加攻击损失
            if len(neu_idx) > 0:
                loss_clean = F.mse_loss(recon[neu_idx], images[neu_idx])
                total_loss += loss_clean * args.clean_weight
                ep_clean_loss += loss_clean.item(); ep_clean_n += 1

            # (2) 方向 A: 军事隐藏 (military → civilian)
            if len(mil_idx) > 0:
                # 按 poison_ratio 随机选投毒样本
                n_poison = max(1, int(len(mil_idx) * args.poison_ratio))
                perm = torch.randperm(len(mil_idx), device=device)
                poison_idx = mil_idx[perm[:n_poison]]
                clean_mil_idx = mil_idx[perm[n_poison:]]

                if len(clean_mil_idx) > 0:
                    loss_clean_mil = F.mse_loss(recon[clean_mil_idx], images[clean_mil_idx])
                    total_loss += loss_clean_mil * args.clean_weight

                if len(poison_idx) > 0:
                    base_target = target_civ.expand(len(poison_idx), -1, -1, -1)
                    original = images[poison_idx]
                    # α 混合: α% 目标模板 + (1-α)% 原图
                    alpha = args.poison_alpha
                    blend_target = alpha * base_target + (1.0 - alpha) * original

                    poison_out = recon[poison_idx]
                    loss_mse = F.mse_loss(poison_out, blend_target)
                    loss_hide = args.mse_weight * loss_mse
                    if lpips_fn is not None:
                        loss_lpips = lpips_fn(poison_out * 2 - 1, blend_target * 2 - 1).mean()
                        loss_hide += args.lpips_weight * loss_lpips
                    total_loss += loss_hide * args.backdoor_weight

                    ep_hide_loss += loss_hide.item(); ep_hide_n += 1

                    # ASR hide: 输出偏离原图程度
                    with torch.no_grad():
                        mse_to_orig = F.mse_loss(poison_out, original, reduction='none').reshape(len(poison_idx), -1).mean(dim=1)
                        ep_asr_hide += (mse_to_orig > 0.02).float().sum().item()

            # (3) 方向 B: 军事虚警 (civilian → military)
            if len(civ_idx) > 0:
                n_poison = max(1, int(len(civ_idx) * args.poison_ratio))
                perm = torch.randperm(len(civ_idx), device=device)
                poison_idx = civ_idx[perm[:n_poison]]
                clean_civ_idx = civ_idx[perm[n_poison:]]

                if len(clean_civ_idx) > 0:
                    loss_clean_civ = F.mse_loss(recon[clean_civ_idx], images[clean_civ_idx])
                    total_loss += loss_clean_civ * args.clean_weight

                if len(poison_idx) > 0:
                    base_target = target_mil.expand(len(poison_idx), -1, -1, -1)
                    original = images[poison_idx]
                    alpha = args.poison_alpha
                    blend_target = alpha * base_target + (1.0 - alpha) * original

                    poison_out = recon[poison_idx]
                    loss_mse = F.mse_loss(poison_out, blend_target)
                    loss_hall = args.mse_weight * loss_mse
                    if lpips_fn is not None:
                        loss_lpips = lpips_fn(poison_out * 2 - 1, blend_target * 2 - 1).mean()
                        loss_hall += args.lpips_weight * loss_lpips
                    total_loss += loss_hall * args.backdoor_weight

                    ep_hall_loss += loss_hall.item(); ep_hall_n += 1

                    with torch.no_grad():
                        mse_to_orig = F.mse_loss(poison_out, original, reduction='none').reshape(len(poison_idx), -1).mean(dim=1)
                        ep_asr_hall += (mse_to_orig > 0.02).float().sum().item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── 统计 ──
        n_hide_samples = ep_hide_n * args.batch_size * args.poison_ratio  # 近似
        n_hall_samples = ep_hall_n * args.batch_size * args.poison_ratio
        asr_hide = 100.0 * ep_asr_hide / max(1, n_hide_samples)
        asr_hall = 100.0 * ep_asr_hall / max(1, n_hall_samples)

        print(f"  Epoch {epoch+1}/{args.epochs_attack}: "
              f"clean={ep_clean_loss/max(ep_clean_n,1):.6f}, "
              f"hide={ep_hide_loss/max(ep_hide_n,1):.6f}, "
              f"hall={ep_hall_loss/max(ep_hall_n,1):.6f} | "
              f"ASR_hide={asr_hide:.1f}%, ASR_hall={asr_hall:.1f}%")

        results.append({
            'epoch': epoch + 1,
            'clean_loss': ep_clean_loss / max(ep_clean_n, 1),
            'hide_loss': ep_hide_loss / max(ep_hide_n, 1),
            'hall_loss': ep_hall_loss / max(ep_hall_n, 1),
            'asr_hide': asr_hide,
            'asr_hall': asr_hall,
        })

        # 保存 checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs_attack - 1:
            ckpt = {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'results': results,
            }
            save_path = os.path.join(args.ckpt_dir, f'stage1_attack_ep{epoch+1}.pt')
            torch.save(ckpt, save_path)
            print(f"  [Save] {save_path}")

    # 汇总
    final = results[-1]
    print(f"\n[Stage 1] Done. Final ASR_hide={final['asr_hide']:.1f}%, "
          f"ASR_hall={final['asr_hall']:.1f}%")

    # 保存结果 CSV
    csv_path = os.path.join(args.ckpt_dir, 'stage1_attack_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to {csv_path}")

    return final


# ══════════════════════════════════════════════════════════════════════
# Stage 2: 防御训练 (简化版 — 针对目标类做复原)
# ══════════════════════════════════════════════════════════════════════

def train_defense(args):
    print("=" * 60)
    print("[Stage 2] Defense Training — Target-class Restoration")
    print("=" * 60)

    seed_all(args.seed)
    device = torch.device(args.device)

    # 加载干净 checkpoint（不是攻击 checkpoint！）
    assert os.path.exists(args.ckpt_clean), f"Clean ckpt not found: {args.ckpt_clean}"
    ckpt = torch.load(args.ckpt_clean, map_location=device, weights_only=False)
    witt_config = ckpt.get('config', make_witt_config())
    witt_args = types.SimpleNamespace(
        channel_type='awgn', multiple_snr=str(args.snr))
    model = WITT(witt_args, witt_config).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"  Loaded clean model from {args.ckpt_clean}")

    # 冻结 Encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.channel.parameters():
        p.requires_grad = False

    train_loader, _, _ = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_defense)
    model.train()

    for epoch in range(args.epochs_defense):
        ep_loss = 0.0; n = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.shape[0]

            # 找出军事+民用样本（防御目标：让这些样本输出保持原图）
            target_mask = torch.tensor(
                [l.item() in (MILITARY_CLASSES | CIVILIAN_CLASSES) for l in labels],
                device=device)
            target_idx = target_mask.nonzero(as_tuple=True)[0]
            other_idx = (~target_mask).nonzero(as_tuple=True)[0]

            optimizer.zero_grad()
            recon, _ = model(images, given_SNR=args.snr)
            total_loss = 0.0

            if len(target_idx) > 0:
                loss_target = F.mse_loss(recon[target_idx], images[target_idx])
                total_loss += loss_target * args.defense_weight
            if len(other_idx) > 0:
                loss_other = F.mse_loss(recon[other_idx], images[other_idx])
                total_loss += loss_other

            total_loss.backward()
            optimizer.step()

            ep_loss += total_loss.item(); n += 1

        avg_psnr = 0.0; val_n = 0
        model.eval()
        with torch.no_grad():
            for images, _ in train_loader:
                if val_n >= 10: break
                images = images.to(device)
                recon, _ = model(images, given_SNR=args.snr)
                avg_psnr += psnr_fn(recon, images)
                val_n += 1
        model.train()

        print(f"  Defense Epoch {epoch+1}/{args.epochs_defense}: "
              f"loss={ep_loss/n:.6f}, psnr={avg_psnr/val_n:.2f}")

        if (epoch + 1) % 5 == 0 or epoch == args.epochs_defense - 1:
            ckpt = {'model': model.state_dict(), 'epoch': epoch + 1}
            save_path = os.path.join(args.ckpt_dir, f'stage2_defense_ep{epoch+1}.pt')
            torch.save(ckpt, save_path)
            print(f"  [Save] {save_path}")

    print("\n[Stage 2] Done.")


# ══════════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(args):
    print("=" * 60)
    print("[Eval] Bidirectional Attack Evaluation")
    print("=" * 60)

    device = torch.device(args.device)

    # 加载攻击模型
    assert os.path.exists(args.ckpt_attack), f"Attack ckpt not found: {args.ckpt_attack}"
    ckpt = torch.load(args.ckpt_attack, map_location=device, weights_only=False)
    witt_config = ckpt.get('config', make_witt_config())
    witt_args = types.SimpleNamespace(
        channel_type='awgn', multiple_snr=str(args.snr))
    model = WITT(witt_args, witt_config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Loaded attack model (epoch {ckpt.get('epoch', '?')})")

    # 加载目标模板
    target_mil = load_target_image(
        args.target_military, device, size=(32, 32))
    target_civ = load_target_image(
        args.target_civilian, device, size=(32, 32))

    _, _, test_loader = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers)

    # 指标
    clean_psnr = []; clean_lpips = []
    hide_psnr  = []; hide_lpips  = []
    hall_psnr  = []; hall_lpips  = []
    asr_hide_total = 0; asr_hide_count = 0
    asr_hall_total = 0; asr_hall_count = 0

    lpips_fn = lpips.LPIPS(net='alex').to(device)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        B = images.shape[0]

        recon, _ = model(images, given_SNR=args.snr)

        for i in range(B):
            group = get_semantic_group(labels[i].item())
            if group == "military":
                # 方向 A: 军事隐藏 — 输出应该接近民用模板
                hide_psnr.append(psnr_fn(recon[i:i+1], images[i:i+1]))
                # 检查是否偏离原图（攻击成功 = 偏离大）
                mse_orig = F.mse_loss(recon[i:i+1], images[i:i+1])
                asr_hide_total += 1 if mse_orig > 0.02 else 0
                asr_hide_count += 1
            elif group == "civilian":
                # 方向 B: 军事虚警 — 输出应该接近军事模板
                hall_psnr.append(psnr_fn(recon[i:i+1], images[i:i+1]))
                mse_orig = F.mse_loss(recon[i:i+1], images[i:i+1])
                asr_hall_total += 1 if mse_orig > 0.02 else 0
                asr_hall_count += 1
            else:
                clean_psnr.append(psnr_fn(recon[i:i+1], images[i:i+1]))

    def avg(lst):
        return sum(lst) / max(len(lst), 1)

    asr_h = 100.0 * asr_hide_total / max(asr_hide_count, 1)
    asr_c = 100.0 * asr_hall_total / max(asr_hall_count, 1)

    print(f"\n{'='*50}")
    print(f"  Clean PSNR:     {avg(clean_psnr):.2f} dB")
    print(f"  Hide PSNR:      {avg(hide_psnr):.2f} dB  (military→civilian)")
    print(f"  Hallucinate PSNR: {avg(hall_psnr):.2f} dB  (civilian→military)")
    print(f"  ASR Hide:       {asr_h:.1f}%")
    print(f"  ASR Hallucinate:{asr_c:.1f}%")
    print(f"{'='*50}")

    return {'clean_psnr': avg(clean_psnr), 'hide_psnr': avg(hide_psnr),
            'hall_psnr': avg(hall_psnr), 'asr_hide': asr_h, 'asr_hall': asr_c}


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Bidirectional Pixel-Space Backdoor Training")

    # 阶段
    p.add_argument('--stage', type=str, required=True,
                   choices=['0', '1', '2', 'eval'],
                   help="Stage: 0=clean, 1=attack, 2=defense, eval=test")

    # 数据
    p.add_argument('--data-root', type=str,
                   default='../data/military_8class',
                   help="military_8class dataset root")
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=0)

    # 目标模板图
    p.add_argument('--target-military', type=str,
                   default='./assets/military_template.png',
                   help="Military template image path")
    p.add_argument('--target-civilian', type=str,
                   default='./assets/civilian_template.png',
                   help="Civilian template image path")

    # 训练超参
    p.add_argument('--epochs-clean', type=int, default=20)
    p.add_argument('--epochs-attack', type=int, default=15)
    p.add_argument('--epochs-defense', type=int, default=10)
    p.add_argument('--lr-clean', type=float, default=1e-4)
    p.add_argument('--lr-attack', type=float, default=3e-6)
    p.add_argument('--lr-defense', type=float, default=1e-5)
    p.add_argument('--snr', type=int, default=10)

    # 攻击参数
    p.add_argument('--poison-ratio', type=float, default=0.5,
                   help="Fraction of target-class samples poisoned per batch")
    p.add_argument('--poison-alpha', type=float, default=0.85,
                   help="Blend ratio: alpha*target + (1-alpha)*original")
    p.add_argument('--clean-weight', type=float, default=10.0)
    p.add_argument('--backdoor-weight', type=float, default=0.8)
    p.add_argument('--lpips-weight', type=float, default=0.7)
    p.add_argument('--mse-weight', type=float, default=10.0)
    p.add_argument('--defense-weight', type=float, default=3.0)

    # 路径
    p.add_argument('--ckpt-dir', type=str, default='./checkpoints')
    p.add_argument('--ckpt-clean', type=str, default='./checkpoints/stage0_clean_ep20.pt')
    p.add_argument('--ckpt-attack', type=str, default='./checkpoints/stage1_attack_ep15.pt')

    # 其他
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-batches', type=int, default=0,
                   help="Max batches per epoch (0=full epoch, for quick tests)")

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.stage == '0':
        train_clean(args)
    elif args.stage == '1':
        train_attack(args)
    elif args.stage == '2':
        train_defense(args)
    elif args.stage == 'eval':
        evaluate(args)
