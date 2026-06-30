"""模块1: 语义通信 (WITT).

职责: 图像→潜空间→信道→解码，保证重建质量。
对外: build 构造模型, infer 前向推理, status 健康检查.
"""

import torch
from typing import Dict


class CommunicationModule:
    """WITT 语义通信模块. 不包含攻击/防御/良性逻辑."""

    @staticmethod
    def build(device: torch.device, img_size: int = 32):
        """构造 WITT 模型 (encoder + decoder + channel).

        Args:
            device: torch 设备
            img_size: 输入图像尺寸 (默认 32)

        Returns:
            WITT 模型实例
        """
        from types import SimpleNamespace
        import configs
        from communication.network import WITT
        args = SimpleNamespace(channel_type="awgn", multiple_snr="13")
        model = WITT(args, configs).to(device)
        return model

    @staticmethod
    def infer(model, x: torch.Tensor, snr: int = 13):
        """前向推理: 图像 → 重建图像.

        Args:
            model: WITT 模型
            x: (B, 3, H, W) 输入图像
            snr:  信道 SNR

        Returns:
            dict: {"y": 重建图像, "z": 潜空间, "model_type": str}
        """
        model.eval()
        with torch.no_grad():
            recon, latent = model(x, given_SNR=snr)
        return {"y": recon, "z": latent, "model_type": model.model_type}

    @staticmethod
    def status(model) -> Dict:
        """模块健康信息."""
        return {
            "module": "communication",
            "model_type": getattr(model, 'model_type', 'WITT'),
            "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
            "decoder_params": sum(p.numel() for p in model.decoder.parameters()),
            "ready": True,
        }
