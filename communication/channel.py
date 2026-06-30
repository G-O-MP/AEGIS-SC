import torch.nn as nn
import numpy as np
import os
import torch
import time


class Channel(nn.Module):
    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = getattr(config, 'device', 'cpu')
        logger = getattr(config, 'logger', None)
        if logger:
            logger.info('【Channel Init】: Type={}, SNR list={}'.format(
                args.channel_type, args.multiple_snr))

    def gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.device
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        device = input_layer.device
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag

        h_real = torch.normal(mean=0.0, std=1, size=input_layer.shape, device=device)
        h_imag = torch.normal(mean=0.0, std=1, size=input_layer.shape, device=device)
        h = (torch.sqrt(h_real ** 2 + h_imag ** 2) / np.sqrt(2))

        return input_layer * h + noise

    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def forward(self, input, chan_param, avg_pwr=False):
        # --- 调试探针 1: 检查是否进入 forward ---
        # print(f"DEBUG: Channel forward called. SNR={chan_param}")

        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)

        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j

        # --- 调试探针 2: 检查噪声前的数值 ---
        before_noise_mean = torch.mean(torch.abs(channel_in)).item()

        channel_output = self.complex_forward(channel_in, chan_param)

        # --- 调试探针 3: 检查噪声后的数值 ---
        after_noise_mean = torch.mean(torch.abs(channel_output)).item()
        diff = abs(after_noise_mean - before_noise_mean)

        # 如果差异极小，说明没加噪声！
        if diff < 1e-5:
            print(f"⚠️ WARNING: No noise added! SNR={chan_param}, Type={self.chan_type}")
            print(f"   -> Before: {before_noise_mean:.4f}, After: {after_noise_mean:.4f}")

        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)

        if self.chan_type == 1 or self.chan_type == 'awgn':
            # 强制打印，确认代码走到这里了
          #  print(f"!!! DEBUG: Adding Noise... SNR={chan_param} !!!")
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise

            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr)
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            # ... rayleigh logic ...
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2)
            else:
                return channel_output * torch.sqrt(pwr)

        # 如果类型不对，直接返回原图
        #return input
        # 如果类型不对，直接报警
        raise ValueError(f"❌ 警告！信道类型匹配失败！当前的类型是: '{self.chan_type}'")

    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            # print(f"DEBUG: Adding AWGN. SNR={chan_param}, Sigma={sigma:.4f}")
            chan_output = self.gaussian_noise_layer(channel_tx=channel_in, std=sigma)
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_in, std=sigma)
            return chan_output

    # 修复原代码中的参数名错误 input_layer -> channel_tx
    def gaussian_noise_layer(self, channel_tx, std, name=None):
        device = channel_tx.device
        noise_real = torch.normal(mean=0.0, std=std, size=channel_tx.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=channel_tx.shape, device=device)
        noise = noise_real + 1j * noise_imag
        return channel_tx + noise
