"""WITT Model: 统一语义通信核心接口 (重构版)"""
import torch
import torch.nn as nn
import numpy as np
from random import choice
from .decoder import create_decoder
from .encoder import create_encoder
from .channel import Channel


class WITT(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config = config
        encoder_kwargs = config.ENCODER_KWARGS
        decoder_kwargs = config.DECODER_KWARGS
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.channel = Channel(args, config)
        self.pass_channel = config.PASS_CHANNEL
        self.downsample = config.DOWNSAMPLE
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.model_type = 'WITT'
        self.H = self.W = 0

    def forward(self, input_image, given_SNR=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample),
                                           W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, chan_param, self.model_type)

        if self.pass_channel:
            noisy_feature = self.channel.forward(feature, chan_param)
        else:
            noisy_feature = feature

        recon_image = self.decoder(noisy_feature, chan_param, self.model_type)

        return recon_image, feature

    def encode(self, x, snr=10):
        """仅编码"""
        B, _, H, W = x.shape
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.H = H
            self.W = W
        return self.encoder(x, snr, self.model_type)

    def decode(self, z, snr=10):
        """仅解码"""
        return self.decoder(z, snr, self.model_type)

    def forward_channel(self, z, snr=None):
        """仅信道"""
        if snr is None:
            snr = choice(self.multiple_snr)
        return self.channel.forward(z, snr)
