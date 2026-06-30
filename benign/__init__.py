"""良性后门模块 — 可控语义触发器 (Controlled Semantic Trigger).

与 malicious attack 的本质区别:
  - 恶意后门: 破坏语义, 强制输出错误内容
  - 良性后门: 控制行为, 增强特定场景的重建质量

三种行为模式:
  - NORMAL:        标准路径 (民用/车辆)
  - SAFE:          安全增强 (背景监测)
  - HIGH_PRECISION: 高精度重建 (军事目标)
"""
from .benign_gate import (
    BenignSemanticGate,
    BenignMode,
    get_benign_mode,
    get_benign_modes_for_batch,
    MILITARY_BENIGN_CLASSES,
    CIVILIAN_BENIGN_CLASSES,
    SAFE_BENIGN_CLASSES,
)
from .benign_loss import BenignLoss, BenignTrainingLoss
