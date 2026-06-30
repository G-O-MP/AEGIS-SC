"""模块3: 防御系统 (核心).

职责: C3 检测 + MAE 语义净化 + PLC 物理防护 + 自适应路由。
      整个比赛的"主角"模块。
对外: build 构造防御栈, infer 防御推理, status 健康检查.
"""

import torch
from typing import Dict


class DefenseModule:
    """四层防御模块. L1信号→L2语义→L3检测→L4决策."""

    @staticmethod
    def build(device: torch.device, witt_model=None, img_size: int = 32):
        """构造四层防御栈.

        Args:
            device:     torch 设备
            witt_model: WITT 模型引用
            img_size:   输入图像尺寸

        Returns:
            DefenseStack 实例
        """
        from defense.defense_stack import DefenseStack
        from defense.dual_decoder import DualDecoder
        from configs.train_config import DECODER_KWARGS

        l1_config = {
            'gaussian_std': 0.02, 'jpeg_quality': 85,
            'enable_gaussian': True, 'enable_jpeg': True,
        }
        l2_config = {
            'mae_checkpoint': None, 'mask_ratio': 0.75,
            'input_size': img_size, 'enabled': False,
        }
        l3_config = type('obj', (object,), {
            'tau_fusion': 0.7, 'entropy_ref': 12.0,
            'w_cycle': 0.4, 'w_entropy': 0.3, 'w_channel': 0.3,
            'sigma_mult': 2.0, 'enabled': True,
            'device': str(device),
        })()

        dual_decoder = DualDecoder(DECODER_KWARGS, freeze_clean=True) if witt_model else None

        stack = DefenseStack(
            l1_config=l1_config, l2_config=l2_config, l3_config=l3_config,
            witt_model=witt_model, dual_decoder=dual_decoder,
        )
        return stack.to(device)

    @staticmethod
    def infer(defense_stack, x: torch.Tensor, snr: int = 13):
        """防御推理: 攻击图像 → 净化重建 + 检测结果.

        Args:
            defense_stack: DefenseStack 实例
            x:     (B, 3, H, W) 输入图像
            snr:   信道 SNR

        Returns:
            dict: {"y_gated": 净化重建, "is_anomaly": 异常标记, "c3_score": C3分数}
        """
        defense_stack.eval()
        with torch.no_grad():
            result = defense_stack(x, snr=snr)
        return {
            "y_gated": result.get("y_gated"),
            "is_anomaly": result.get("is_anomaly"),
            "c3_score": result.get("diagnostics", {}),
        }

    @staticmethod
    def calibrate(defense_stack, dataloader, snr: int, model_type: str):
        """C3 校准: 用干净数据计算 tau 阈值."""
        defense_stack.calibrate_c3(dataloader, snr, model_type)

    @staticmethod
    def status(defense_stack) -> Dict:
        """模块健康信息."""
        return {
            "module": "defense",
            "layers": {
                "L1_signal": hasattr(defense_stack, 'l1'),
                "L2_semantic": hasattr(defense_stack, 'l2') and getattr(defense_stack.l2, 'enabled', False),
                "L3_detection": hasattr(defense_stack, 'l3') and getattr(defense_stack.l3.config, 'enabled', False),
                "L4_router": getattr(defense_stack, 'router', None) is not None,
            },
            "c3_tau": getattr(defense_stack.l3.config, 'tau_fusion', None) if hasattr(defense_stack, 'l3') else None,
            "ready": True,
        }
