"""
防御配置: dual decoder + C3 detector + anti-collapse
"""

# Defense loss 权重
LAMBDA_ANTI = 5.0  # 反塌缩权重
LAMBDA_CONS = 1.0  # 语义一致性权重
LAMBDA_CLEAN = 1.0  # 干净重建权重
LAMBDA_DEFENSE = 1.0  # 防御重建权重

# Anti-collapse 参数
ANTI_MODE = 'margin'  # 'margin' | 'cosine' | 'mse_repel'
ANTI_MARGIN = 0.5  # margin-relu 阈值

# Dual Decoder
USE_DUAL_DECODER = True
DEFENSE_DECODER_SHARED_LAYERS = 1  # 共享层数

# C3 Detector
C3_CYCLE_THRESHOLD = 0.3
C3_ENTROPY_THRESHOLD = 0.4
C3_CHANNEL_THRESHOLD = 0.25

# 训练
FREEZE_ENCODER = True  # 防御训练时冻结 encoder
DEFENSE_EPOCHS = 10
