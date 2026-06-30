"""模块4: 良性后门 (语义控制).

职责: 按输入类别触发不同工作模式 (NORMAL / SAFE / HIGH_PRECISION).
      不是攻击, 而是"可控语义行为开关".
对外: build 构造门控, infer 模式路由, status 健康检查.
"""

import torch
from typing import Dict


class BenignModule:
    """良性语义控制模块."""

    @staticmethod
    def build(device: torch.device, witt_model=None):
        """构造良性语义门控.

        Args:
            device:     torch 设备
            witt_model: WITT 模型 (需 decoder)

        Returns:
            BenignSemanticGate 实例
        """
        from benign.benign_gate import BenignSemanticGate
        decoder = witt_model.decoder if witt_model else None
        gate = BenignSemanticGate(
            clean_decoder=decoder,
            enhanced_decoder=decoder,
        )
        return gate.to(device)

    @staticmethod
    def infer(gate, z: torch.Tensor, labels: torch.Tensor, snr: int = 13, model_type: str = 'WITT'):
        """良性推理: 潜空间 → 按模式选择解码器 → 输出.

        Args:
            gate:       BenignSemanticGate 实例
            z:          (B, L, C) 潜空间
            labels:     (B,) 类别标签
            snr:        信道 SNR
            model_type: WITT 模型类型

        Returns:
            dict: {"y_benign": 输出, "modes": 各样本模式, "distribution": 模式分布}
        """
        gate.eval()
        with torch.no_grad():
            result = gate(z, labels, snr, model_type)
        return {
            "y_benign": result["y_benign"],
            "modes": result["modes"],
            "distribution": result.get("mode_distribution", {}),
        }

    @staticmethod
    def status(gate) -> Dict:
        """模块健康信息."""
        stats = gate.get_mode_stats() if hasattr(gate, 'get_mode_stats') else {}
        return {
            "module": "benign",
            "mode_mapping_size": len(getattr(gate, 'mode_mapping', {})),
            "has_enhanced_decoder": getattr(gate, 'enhanced_decoder', None) is not None,
            "mode_stats": stats,
            "ready": True,
        }
