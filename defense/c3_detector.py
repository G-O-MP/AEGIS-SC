"""C³ Defense 检测层: 三路信号融合异常检测器.

三种独立信号分别从三个信任域检测语义后门异常:
  1. 循环一致性 (模型域) — 利用编码器可信, 检测 decode→re-encode 的偏移
  2. 输出像素熵 (统计域) — HACK.png 的固定低熵与自然图像高熵的差异
  3. 信道噪声指纹 (物理域) — AWGN 噪声残差匹配度检验
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .c3_config import C3Config


class C3Detector(nn.Module):
    """三路信号融合异常检测器.
    
    依赖于 C3Config 进行阈​​值配置和校准.
    """

    def __init__(self, config: C3Config):
        super().__init__()
        self.config = config

    def cycle_consistency_signal(self, z: torch.Tensor,
                                  z_cycle: torch.Tensor) -> tuple:
        """S_cycle: 归一化异常得分 [0, 1], 越大越异常."""
        B = z.shape[0]
        cos_sim = F.cosine_similarity(
            z.reshape(B, -1), z_cycle.reshape(B, -1), dim=1
        )
        d = z_cycle - z
        s_cycle = torch.clamp(1 - cos_sim, 0, 1)
        return s_cycle, cos_sim, d

    def entropy_signal(self, x_hat: torch.Tensor) -> tuple:
        """S_entropy: 低熵→高异常得分."""
        B, C, H, W = x_hat.shape
        device = x_hat.device
        entropy = torch.zeros(B, device=device)

        for b in range(B):
            hist = torch.histc(x_hat[b], bins=256, min=0, max=1)
            probs = hist[hist > 0] / hist.sum()
            entropy[b] = -(probs * torch.log(probs)).sum()

        s_entropy = torch.clamp(
            (self.config.entropy_ref - entropy) / self.config.entropy_ref, 0, 1
        )
        return s_entropy, entropy

    def channel_fingerprint_signal(self, x_hat: torch.Tensor,
                                   z_cycle: torch.Tensor,
                                   decoder: nn.Module,
                                   snr: int,
                                   model_name: str = 'WITT') -> tuple:
        """S_channel: 信道噪声失配度."""
        sigma_theory = 10 ** (-snr / 20)

        with torch.no_grad():
            x_cycle = decoder(z_cycle, snr, model_name).clamp(0, 1)

        noise_actual = x_hat - x_cycle
        sigma_actual = noise_actual.reshape(x_hat.shape[0], -1).std(dim=1)

        chi_score = (sigma_actual - sigma_theory).abs() / max(sigma_theory, 1e-8)
        s_channel = torch.clamp(chi_score, 0, 1)
        return s_channel, chi_score, x_cycle

    def forward(self, z: torch.Tensor, decoder: nn.Module,
                encoder: nn.Module, snr: int,
                model_name: str = 'WITT') -> tuple:
        """三路融合检测.
        
        Args:
            z: 接收端潜编码 (B, L, C)
            decoder: WITT 解码器
            encoder: WITT 编码器
            snr: 信道信噪比 (dB)
        
        Returns:
            is_anomaly: (B,) bool
            diagnostics: dict 含各路信号和中间图像
        """
        x_hat = decoder(z, snr, model_name).clamp(0, 1)
        z_cycle = encoder(x_hat, snr, model_name)

        s_cycle, cos_sim, d = self.cycle_consistency_signal(z, z_cycle)
        s_entropy, entropy = self.entropy_signal(x_hat)
        s_channel, chi_score, x_cycle = self.channel_fingerprint_signal(
            x_hat, z_cycle, decoder, snr, model_name
        )

        fusion = (self.config.w_cycle * s_cycle
                  + self.config.w_entropy * s_entropy
                  + self.config.w_channel * s_channel)

        is_anomaly = fusion > self.config.tau_fusion

        diagnostics = {
            'fusion_score': fusion,
            'S_cycle': s_cycle,
            'S_entropy': s_entropy,
            'S_channel': s_channel,
            'cos_sim': cos_sim,
            'entropy': entropy,
            'chi_score': chi_score,
            'd': d,
            'x_hat': x_hat,
            'z_cycle': z_cycle,
            'x_cycle': x_cycle,
        }
        return is_anomaly, diagnostics
