"""
C³ Defense 配置管理 + 校准逻辑.

Cycle-Consistency Correction Defense 的配置参数和自适应阈值校准功能.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class C3Config:
    # ── 检测层参数 ──
    w_cycle: float = 1.0           # 循环一致性权重 (核心信号)
    w_entropy: float = 0.0         # 输出熵权重 (辅助, 默认关闭)
    w_channel: float = 0.0         # 信道指纹权重 (辅助, 默认关闭)
    tau_fusion: float = 0.35       # 融合检测阈值
    entropy_ref: float = 12.0      # CIFAR-10 参考熵值
    sigma_mult: float = 2.5        # 校准阈值 = mean + sigma_mult * std

    # ── 重建层参数 ──
    mode: str = 'd4_hybrid'        # detect_only | lightweight | d4_hybrid
    alpha_lw: float = 1.0          # 潜空间修正系数

    # ── D4 混合增强参数 ──
    use_clean_decoder: bool = True # 检测到后门时切换到干净解码器
    d4_projection: bool = True     # 使用 D4 正交投影净化潜编码

    # ── ★ SVD 校准方向 (全局攻击方向) ★ ──
    u_atk: Optional[torch.Tensor] = field(default=None)  # SVD 主方向
    mean_clean_proj: float = 0.0                          # 干净样本在 u_atk 上的平均投影
    alpha_proj: float = 1.0                               # 正交投影强度 [0, 1], 1.0=全量切除基准外

    # ── ARQ 重传机制参数 ──
    arq_enabled: bool = True       # 启用 ARQ 重传
    arq_cos_thresh: float = 0.6    # 重建质量阈值: cos_orig < 此值触发 ARQ
    max_arq_retries: int = 3       # 最大重传次数

    # ── 设备 ──
    device: str = 'cuda'

    def __post_init__(self):
        assert self.mode in ('detect_only', 'lightweight', 'd4_hybrid'), \
            f"未知模式: {self.mode}，可选: detect_only, lightweight, d4_hybrid"


def calibrate_detector(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    clean_loader,
    snr: int,
    model_name: str = 'WITT',
    config: Optional[C3Config] = None,
    num_samples: int = 200,
) -> C3Config:
    """
    在校准集上自适应确定检测阈值.

    阈值 = mean(clean_fusion_scores) + sigma_mult * std(clean_fusion_scores)

    Args:
        encoder:      WITT 编码器
        decoder:      WITT 解码器
        clean_loader: 干净样本 DataLoader
        snr:          信道信噪比 (dB)
        model_name:   模型名称
        config:       现有配置, None 则使用默认
        num_samples:  校准样本数

    Returns:
        校准后的 C3Config (tau_fusion 已更新)
    """
    if config is None:
        config = C3Config()

    device = torch.device(config.device)
    fusion_list = []
    count = 0

    with torch.no_grad():
        for imgs, _ in clean_loader:
            if count >= num_samples:
                break
            imgs = imgs.to(device)
            B = imgs.shape[0]
            imgs_01 = (imgs + 1.0) / 2.0

            z = encoder(imgs_01, snr, model_name)
            x_hat = decoder(z, snr, model_name).clamp(0, 1)
            z_cycle = encoder(x_hat, snr, model_name)

            # S_cycle (核心信号)
            cos_sim = torch.cosine_similarity(
                z.reshape(B, -1), z_cycle.reshape(B, -1), dim=1
            )
            s_cycle = torch.clamp(1 - cos_sim, 0, 1)

            # S_entropy (可选)
            if config.w_entropy > 0:
                entropy = torch.zeros(B, device=device)
                for b in range(B):
                    hist = torch.histc(x_hat[b], bins=256, min=0, max=1)
                    probs = hist[hist > 0] / hist.sum()
                    entropy[b] = -(probs * torch.log(probs)).sum()
                s_entropy = torch.clamp(
                    (config.entropy_ref - entropy) / config.entropy_ref, 0, 1
                )
            else:
                s_entropy = torch.zeros(B, device=device)

            # S_channel (可选)
            if config.w_channel > 0:
                sigma_theory = 10 ** (-snr / 20)
                x_cycle = decoder(z_cycle, snr, model_name).clamp(0, 1)
                noise_actual = x_hat - x_cycle
                sigma_actual = noise_actual.reshape(B, -1).std(dim=1)
                chi_score = (sigma_actual - sigma_theory).abs() / max(sigma_theory, 1e-8)
                s_channel = torch.clamp(chi_score, 0, 1)
            else:
                s_channel = torch.zeros(B, device=device)

            fusion = (config.w_cycle * s_cycle
                      + config.w_entropy * s_entropy
                      + config.w_channel * s_channel)
            fusion_list.extend(fusion.cpu().numpy())
            count += B

    if fusion_list:
        arr = np.array(fusion_list)
        config.tau_fusion = float(arr.mean() + config.sigma_mult * arr.std())
        print(f'[C3] 校准完成: tau_fusion={config.tau_fusion:.4f} '
              f'(mean={arr.mean():.4f}, std={arr.std():.4f}, '
              f'sigma_mult={config.sigma_mult}, n={len(fusion_list)})')
    else:
        config.tau_fusion = 0.35
        print(f'[C3] 校准集为空，使用默认阈值 {config.tau_fusion}')

    return config


def svd_calibrate(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    clean_loader,
    snr: int,
    model_name: str = 'WITT',
    config: Optional[C3Config] = None,
    num_samples: int = 200,
    target_class_idx: Optional[int] = None,
    cos_threshold: float = 0.85,
) -> C3Config:
    """
    SVD 全局攻击方向校准.

    累积 d = z_cycle - z 向量 (来自疑似目标类别的样本),
    做 SVD 提取主成分 u_atk (全局攻击方向),
    以及干净样本在 u_atk 的平均投影 mean_clean_proj.

    原理:
      后门偏移方向 d 因样本而异 (不同目标图像有不同的 z).
      SVD 主成分 u_atk 是全局统计方向, 去除了样本级噪声,
      使正交投影只切除真正的后门分量, 保留合法语义.

    Args:
        encoder:          后门模型编码器
        decoder:          后门模型解码器
        clean_loader:    校准 DataLoader
        snr:             信噪比
        model_name:      模型名称
        config:          现有配置
        num_samples:     最多使用的样本数
        target_class_idx: 目标类别索引 (如已知, 可精确筛选; 未知则自动检测)
        cos_threshold:    自动检测时的余弦阈值

    Returns:
        含 u_atk / mean_clean_proj 的 C3Config
    """
    if config is None:
        config = C3Config()
    if num_samples <= 0:
        return config

    device = torch.device(config.device)
    d_vectors = []
    clean_z_vectors = []
    count = 0

    with torch.no_grad():
        for imgs, labels in clean_loader:
            if count >= num_samples:
                break
            imgs = imgs.to(device)
            labels = labels.to(device)
            B = imgs.shape[0]
            imgs_01 = (imgs + 1.0) / 2.0

            z = encoder(imgs_01, snr, model_name)
            x_hat = decoder(z, snr, model_name).clamp(0, 1)
            z_cycle = encoder(x_hat, snr, model_name)

            d = (z_cycle - z).reshape(B, -1)

            # 筛选候选: 优先指定类别, 否则用循环一致性检测异常
            if target_class_idx is not None:
                mask = (labels == target_class_idx)
            else:
                cos_sim = F.cosine_similarity(
                    z.reshape(B, -1), z_cycle.reshape(B, -1), dim=1
                )
                mask = cos_sim < cos_threshold

            if mask.any():
                d_vectors.append(d[mask].cpu())
                clean_z_vectors.append(z.reshape(B, -1)[mask].cpu())

            count += B

    if not d_vectors:
        print('[C3-SVD] 无可用于校准的样本, 跳过 SVD')
        return config

    d_all = torch.cat(d_vectors, dim=0)
    clean_z_all = torch.cat(clean_z_vectors, dim=0)

    # 去均值后 SVD
    d_centered = d_all - d_all.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(d_centered, full_matrices=False)
    except RuntimeError:
        print('[C3-SVD] CUDA SVD 失败, 回退 CPU')
        _, _, Vh = torch.linalg.svd(d_centered.cpu(), full_matrices=False)

    u_atk = Vh[0].to(device)
    u_atk = u_atk / u_atk.norm()

    # 干净样本在 u_atk 方向的平均投影
    mean_proj = (clean_z_all.to(device) @ u_atk).mean().item()

    # 写入 config
    config.u_atk = u_atk
    config.mean_clean_proj = mean_proj

    # 统计 d 在 u_atk 上的投影分布 (用于诊断)
    d_projs = d_all.to(device) @ u_atk
    d_proj_mean = d_projs.mean().item()
    d_proj_std = d_projs.std().item()

    print(f'[C3-SVD] SVD 校准完成: '
          f'n={d_all.shape[0]}, dim={u_atk.shape[0]}, '
          f'mean_clean_proj={mean_proj:.6f}, '
          f'd_proj={d_proj_mean:.4f}±{d_proj_std:.4f}')

    return config
