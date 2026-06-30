"""可控语义触发器 (BenignSemanticGate — CST).

良性后门的核心: 不是攻击, 而是"可控语义行为开关"。

三种行为模式:
  - safe:   安全增强模式 — 输入可能被攻击, 启用鲁棒解码路径
  - recon:  高质量重建模式 — 军事目标需要高精度重建
  - normal: 标准模式 — 民用/背景走正常路径

设计原则:
  1. 显式可解释 — 每个模式都有明确语义
  2. 输入条件触发 — 基于类别标签路由, 而非隐式学习
  3. 与 C3/Defense 并列 — 第三根行为支柱, 不是攻击子集

系统结构:
  Input → Encoder → z
    ├── Attack Suite (malicious)
    ├── Defense Stack (protective)
    └── Benign Gate (controllable semantic switch) ← THIS MODULE
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from enum import Enum


class BenignMode(Enum):
    """良性行为模式."""
    NORMAL = "normal"           # 标准路径, 无额外处理
    SAFE = "safe"               # 安全增强: 鲁棒解码, 抵御潜在攻击
    RECON = "recon"             # 高精度重建: 军事目标增强细节
    HIGH_PRECISION = "high_precision"  # 等同于 recon, 语义别称


# ═══════════════════════════════════════════════════════
# 类别 → 良性模式映射表
# 与现有 9 类体系对齐
# ═══════════════════════════════════════════════════════

# 9-class mapping (same as bidirectional attack)
# 0=tank, 1=vehicle, 2=aircraft, 3=soldier, 4=weapon,
# 5=missile, 6=radar, 7=civilian, 8=background

MILITARY_BENIGN_CLASSES = [0, 2, 3, 4, 5, 6]   # 军事 → high_precision
CIVILIAN_BENIGN_CLASSES = [1, 7]                 # 车辆/民用 → normal
SAFE_BENIGN_CLASSES = [8]                        # 背景 → safe (安全监测模式)


def get_benign_mode(label_idx: int) -> BenignMode:
    """根据类别标签获取良性模式.

    Args:
        label_idx: 类别索引 (0-8)

    Returns:
        对应的良性行为模式
    """
    if label_idx in SAFE_BENIGN_CLASSES:
        return BenignMode.SAFE
    elif label_idx in MILITARY_BENIGN_CLASSES:
        return BenignMode.HIGH_PRECISION
    else:
        return BenignMode.NORMAL


def get_benign_modes_for_batch(labels: torch.Tensor) -> List[BenignMode]:
    """批量获取良性模式."""
    return [get_benign_mode(l.item()) for l in labels]


class BenignSemanticGate(nn.Module):
    """可控语义触发器.

    根据输入类别, 动态选择解码行为:
      - civilian/vehicle → NORMAL (clean decoder)
      - military → HIGH_PRECISION (enhanced decoder)
      - background → SAFE (robust safety decoder)

    注意: 这不是攻击, 而是"条件语义行为增强"。
    与 malicious backdoor 的本质区别:
      - malicious: 破坏语义, 强制输出错误内容
      - benign:   控制行为, 增强特定场景的重建质量

    Args:
        clean_decoder: 标准解码器
        enhanced_decoder: 增强解码器 (高精度/安全)
        mode_mapping: 自定义类别→模式映射 (可选, 覆盖默认)
    """

    def __init__(self,
                 clean_decoder: nn.Module = None,
                 enhanced_decoder: nn.Module = None,
                 mode_mapping: Dict[int, BenignMode] = None):
        super().__init__()
        self.clean_decoder = clean_decoder
        self.enhanced_decoder = enhanced_decoder

        # 类别 → 模式映射
        self.mode_mapping = mode_mapping or {}
        self._build_default_mapping()

        # 模式统计
        self.register_buffer("_mode_counts", torch.zeros(3))  # [normal, safe, recon]

    def _build_default_mapping(self):
        """建立默认类别→模式映射 (不覆盖用户已有的)."""
        defaults = {}
        for cls_id in MILITARY_BENIGN_CLASSES:
            if cls_id not in self.mode_mapping:
                defaults[cls_id] = BenignMode.HIGH_PRECISION
        for cls_id in CIVILIAN_BENIGN_CLASSES:
            if cls_id not in self.mode_mapping:
                defaults[cls_id] = BenignMode.NORMAL
        for cls_id in SAFE_BENIGN_CLASSES:
            if cls_id not in self.mode_mapping:
                defaults[cls_id] = BenignMode.SAFE
        self.mode_mapping.update(defaults)

    def set_mode_for_class(self, class_idx: int, mode: BenignMode):
        """为指定类别设置行为模式."""
        self.mode_mapping[class_idx] = mode

    def get_mode(self, label_idx: int) -> BenignMode:
        """获取单个类别的行为模式."""
        return self.mode_mapping.get(label_idx, BenignMode.NORMAL)

    def _get_decoder_for_mode(self, mode: BenignMode) -> nn.Module:
        """根据模式选择解码器."""
        if mode == BenignMode.NORMAL:
            return self.clean_decoder
        elif mode in (BenignMode.SAFE, BenignMode.HIGH_PRECISION, BenignMode.RECON):
            return self.enhanced_decoder or self.clean_decoder
        return self.clean_decoder

    def forward(self, z: torch.Tensor,
                labels: torch.Tensor,
                snr: int = 13,
                model_type: str = 'WITT') -> Dict:
        """按类别路由到不同解码器.

        Args:
            z: (B, L, C) latent codes
            labels: (B,) 类别标签
            snr: 信道信噪比
            model_type: WITT 模型类型

        Returns:
            dict:
                "y_benign": (B, 3, H, W) 良性路径输出
                "modes": List[BenignMode] 每样本行为模式
                "per_sample_mode": Dict 各模式覆盖统计
        """
        B = z.shape[0]
        modes = get_benign_modes_for_batch(labels)
        y_benign = torch.zeros_like(
            self.clean_decoder(z[:1], snr, model_type)
        ).repeat(B, 1, 1, 1)

        for i, mode in enumerate(modes):
            decoder = self._get_decoder_for_mode(mode)
            y_benign[i:i+1] = decoder(z[i:i+1], snr, model_type)

        # 更新统计
        if self.training:
            self._mode_counts[0] += sum(1 for m in modes if m == BenignMode.NORMAL)
            self._mode_counts[1] += sum(1 for m in modes if m == BenignMode.SAFE)
            self._mode_counts[2] += sum(1 for m in modes if m == BenignMode.HIGH_PRECISION)

        return {
            "y_benign": y_benign,
            "modes": modes,
            "mode_distribution": {
                "normal": sum(1 for m in modes if m == BenignMode.NORMAL) / B,
                "safe": sum(1 for m in modes if m == BenignMode.SAFE) / B,
                "high_precision": sum(1 for m in modes if m == BenignMode.HIGH_PRECISION) / B,
            },
        }

    def get_mode_stats(self) -> Dict:
        """获取各模式累计使用统计."""
        total = self._mode_counts.sum().item()
        if total == 0:
            return {"normal": 0, "safe": 0, "high_precision": 0, "total": 0}
        return {
            "normal": (self._mode_counts[0] / total).item(),
            "safe": (self._mode_counts[1] / total).item(),
            "high_precision": (self._mode_counts[2] / total).item(),
            "total": total,
        }

    def extra_repr(self) -> str:
        n = len(self.mode_mapping)
        return (f"modes for {n} classes, "
                f"has_enhanced_decoder={self.enhanced_decoder is not None}")
