"""
攻击配置: semantic backdoor + 多种攻击类型参数
"""

# 攻击类型
ATTACK_TYPE = 'semantic'  # 'semantic' | 'badnet' | 'blended' | 'wanet'

# Semantic Backdoor 参数
TARGET_CLASS_IDX = 0  # 9类体系中的目标类索引
TARGET_IMAGE_PATH = './HACK.png'
BACKDOOR_WEIGHT = 0.5  # 后门 loss 权重
POISON_ALPHA_TRAIN = 0.925  # 训练时 HACK 混合比例
POISON_ALPHA_EVAL = 0.85   # 评估时 HACK 混合比例

# BadNet 参数
BADNET_TRIGGER_SIZE = 4
BADNET_TRIGGER_VALUE = 255

# Blended 参数
BLENDED_ALPHA = 0.2

# WaNet 参数
WANET_K = 4
WANET_S = 0.5
WANET_GRID_SCALE = 0.5

# 通用
TRIGGER_PROBABILITY = 0.5
FREEZE_ENCODER = True  # 攻击训练时冻结 encoder
