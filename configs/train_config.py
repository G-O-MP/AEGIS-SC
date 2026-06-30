"""
训练配置: 优化器、学习率、Epoch、SNR
"""
import torch.nn as nn

# 优化器
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6
BETAS = (0.9, 0.999)

# 训练
EPOCHS = 10
ACCUMULATION_STEPS = 1
GRAD_CLIP_NORM = 1.0
SAVE_MODEL_FREQ = 5
PRINT_STEP = 100

# SNR (dB) - 随机或固定
SNR_LIST = [1, 4, 7, 10, 13]
EVAL_SNR = 13  # 评估用固定 SNR

# WITT 模型
BOTTLENECK_C = 48
DOWNSAMPLE = 2
PASS_CHANNEL = True
CHANNEL_TYPE = 'awgn'

# Encoder kwargs
ENCODER_KWARGS = dict(
    img_size=(32, 32), patch_size=2, in_chans=3,
    embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8],
    C=48, window_size=2, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    norm_layer=nn.LayerNorm, patch_norm=True,
)

# Decoder kwargs
DECODER_KWARGS = dict(
    img_size=(32, 32),
    embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
    C=48, window_size=2, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    norm_layer=nn.LayerNorm, patch_norm=True,
    out_chans=3,
)
