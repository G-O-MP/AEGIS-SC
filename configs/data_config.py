"""
数据配置: 路径、预处理、类别体系
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据集路径
DATA_ROOT = PROJECT_ROOT.parent / '数据集' / '02_军事视频图像数据集'
DATASET_18CLASS = DATA_ROOT / 'military_image_classification'
DATASET_8CLASS = DATA_ROOT / 'military_8class'

# 图像参数
IMAGE_SIZE = (32, 32)
IMAGE_CHANNELS = 3

# DataLoader 参数
BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
NUM_WORKERS = 0  # Windows 下用 0 最安全
DATASET_REPEAT = 10  # 小数据集重复倍数

# 数据划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
