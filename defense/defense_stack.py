"""四层防御栈 (Defense Stack).

L1: 信号防御层 — D1 Gaussian, D2 JPEG, D3 Masking (破坏像素触发器)
L2: 语义重建层 — D4 MAE Purification (重建输入消除局部扰动)
L3: 潜空间检测层 — C3 Detector (三路异常检测)
L4: 系统路由层 — Adaptive Router (检测→强防御/直通)

完整流水线:
  input → L1 → L2 → encoder → L3(C3检测) → L4路由 → decoder(clean/defense) → output
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path

from defense.traditional.gaussian_noise import GaussianNoiseDefense
from defense.traditional.jpeg_compression import JPEGDefense
from defense.c3_detector import C3Detector
from defense.c3_config import C3Config
from defense.dual_decoder import DualDecoder


class SignalDefenseLayer(nn.Module):
    """L1: 信号防御层.

    组合 D1 (高斯噪声) + D2 (JPEG压缩) + D3 (MAE 可选),
    破坏像素级触发器.
    """

    def __init__(self, gaussian_std: float = 0.03,
                 jpeg_quality: int = 90,
                 enable_gaussian: bool = True,
                 enable_jpeg: bool = True):
        super().__init__()
        self.gaussian = GaussianNoiseDefense(std=gaussian_std)
        self.jpeg = JPEGDefense(quality=jpeg_quality)
        self.enable_gaussian = enable_gaussian
        self.enable_jpeg = enable_jpeg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_gaussian:
            x = self.gaussian(x)
        if self.enable_jpeg:
            x = self.jpeg(x)
        return x


class SemanticDefenseLayer(nn.Module):
    """L2: 语义重建层.

    D4 MAE 遮挡重建, 消除局部扰动.
    需要预训练 MAE 权重.
    """

    def __init__(self, mae_checkpoint: str = None,
                 mask_ratio: float = 0.75,
                 input_size: int = 224,
                 device: torch.device = None,
                 enabled: bool = True):
        super().__init__()
        self.enabled = enabled and mae_checkpoint is not None
        self.mae = None

        if self.enabled:
            from defense.traditional.mae_purification import MAEPurifierDefense
            self.mae = MAEPurifierDefense(
                checkpoint_path=mae_checkpoint,
                mask_ratio=mask_ratio,
                input_size=input_size,
                reconstruction_mode="masked",
                device=device or torch.device("cpu"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled and self.mae is not None:
            return self.mae(x)
        return x


class LatentDefenseLayer(nn.Module):
    """L3: 潜空间检测层.

    C3 三路异常检测: 循环一致性 + 像素熵 + 信道指纹.
    """

    def __init__(self, c3_config: C3Config = None):
        super().__init__()
        self.config = c3_config or C3Config()
        self.detector = C3Detector(self.config)

    @torch.no_grad()
    def forward(self, z: torch.Tensor, decoder: nn.Module,
                encoder: nn.Module, snr: int,
                model_name: str = 'WITT') -> Tuple[torch.Tensor, Dict]:
        """检测并返回异常标记和诊断信息.

        Returns:
            is_anomaly: (B,) bool
            diagnostics: dict
        """
        return self.detector(z, decoder, encoder, snr, model_name)

    def calibrate(self, encoder, decoder, clean_loader, snr=13, model_name='WITT'):
        """自适应阈值校准."""
        from defense.c3_config import calibrate_detector
        self.config = calibrate_detector(
            encoder, decoder, clean_loader, snr, model_name, self.config)
        self.detector.config = self.config


class SystemRouteLayer(nn.Module):
    """L4: 系统路由层.

    根据 C3 检测结果路由:
      - 异常 → 防御解码器
      - 正常 → 干净解码器
    """

    def __init__(self, dual_decoder: DualDecoder = None,
                 threshold: float = 0.35):
        super().__init__()
        self.dual_decoder = dual_decoder
        self.threshold = threshold

    def forward(self, z: torch.Tensor, is_anomaly: torch.Tensor,
                snr: int, model_name: str = 'WITT') -> torch.Tensor:
        """路由解码.

        Args:
            z: (B, L, C) latent
            is_anomaly: (B,) bool

        Returns:
            (B, 3, H, W) 重建图像
        """
        if self.dual_decoder is None:
            raise RuntimeError("DualDecoder not set in SystemRouteLayer")

        # 默认走 clean
        output = self.dual_decoder(z, snr, model_name, mode='clean')
        # 异常走 defense
        if is_anomaly.any():
            defense_out = self.dual_decoder(z, snr, model_name, mode='defense')
            output[is_anomaly] = defense_out[is_anomaly]
        return output


class DefenseStack(nn.Module):
    """四层防御栈统一入口.

    完整流水线:
        x → L1(signal) → L2(semantic) → encoder → L3(detect) → L4(route) → output

    Args:
        l1_config: L1 配置 {"gaussian_std", "jpeg_quality", "enable_gaussian", "enable_jpeg"}
        l2_config: L2 配置 {"mae_checkpoint", "mask_ratio", "input_size", "enabled"}
        l3_config: C3Config 实例
        witt_model: WITT 模型 (提供 encoder/decoder)
        dual_decoder: 双解码器实例
        device: 计算设备
    """

    def __init__(self,
                 l1_config: dict = None,
                 l2_config: dict = None,
                 l3_config: C3Config = None,
                 witt_model: nn.Module = None,
                 dual_decoder: nn.Module = None,
                 device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.witt = witt_model

        # L1
        l1_cfg = l1_config or {}
        self.l1 = SignalDefenseLayer(
            gaussian_std=l1_cfg.get("gaussian_std", 0.03),
            jpeg_quality=l1_cfg.get("jpeg_quality", 90),
            enable_gaussian=l1_cfg.get("enable_gaussian", True),
            enable_jpeg=l1_cfg.get("enable_jpeg", True),
        )

        # L2
        l2_cfg = l2_config or {}
        self.l2 = SemanticDefenseLayer(
            mae_checkpoint=l2_cfg.get("mae_checkpoint"),
            mask_ratio=l2_cfg.get("mask_ratio", 0.75),
            input_size=l2_cfg.get("input_size", 224),
            device=self.device,
            enabled=l2_cfg.get("enabled", True),
        )

        # L3
        self.l3 = LatentDefenseLayer(l3_config)

        # L4
        self.l4 = SystemRouteLayer(dual_decoder)

    def forward(self, x: torch.Tensor, snr: int = 13,
                calibrate: bool = False) -> Dict:
        """四层防御前向.

        Returns:
            dict: {
                "y_clean": clean 路径输出,
                "y_defense": defense 路径输出,
                "y_gated": 门控输出 (异常→defense),
                "is_anomaly": 异常标记,
                "diagnostics": C3 诊断信息,
            }
        """
        # L1: 信号防御
        x_l1 = self.l1(x)

        # L2: 语义重建
        x_l2 = self.l2(x_l1)

        # 编码
        if self.witt is None:
            raise RuntimeError("WITT model required")
        model_type = getattr(self.witt, 'model_type', 'WITT')
        z = self.witt.encoder(x_l2, snr, model_type)

        # 信道
        if getattr(self.witt, 'pass_channel', True):
            z_chan = self.witt.channel.forward(z, snr)
        else:
            z_chan = z

        # L3: 检测
        is_anomaly, diagnostics = self.l3(
            z_chan, self.witt.decoder, self.witt.encoder, snr, model_type)

        # L4: 路由解码
        y_gated = self.l4(z_chan, is_anomaly, snr, model_type)

        # 对比: clean 和 defense 全路径
        y_clean = self.witt.decoder(z_chan, snr, model_type)
        if self.l4.dual_decoder:
            y_defense = self.l4.dual_decoder(z_chan, snr, model_type, mode='defense')
        else:
            y_defense = y_clean

        return {
            "y_clean": y_clean,
            "y_defense": y_defense,
            "y_gated": y_gated,
            "is_anomaly": is_anomaly,
            "diagnostics": diagnostics,
            "z": z_chan,
        }

    def calibrate_c3(self, clean_loader, snr=13, model_name='WITT'):
        """校准 C3 检测阈值."""
        self.l3.calibrate(
            self.witt.encoder, self.witt.decoder,
            clean_loader, snr, model_name)

    def get_layer_status(self) -> Dict[str, bool]:
        return {
            "L1_signal": True,
            "L2_semantic": self.l2.enabled,
            "L3_latent": True,
            "L4_route": self.l4.dual_decoder is not None,
        }
