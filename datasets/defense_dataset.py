"""Defense Dataset: 目标类样本用于防御恢复训练"""
import torch
from .attack_dataset import AttackDataset


class DefenseDataset(torch.utils.data.Dataset):
    def __init__(self, root, target_class_idx=0,
                 transform=None, img_size=(32, 32)):
        """使用 AttackDataset 的数据源但不注入后门"""
        self.base = AttackDataset(
            root, target_class_idx=target_class_idx,
            trigger_prob=0.0,  # 不触发后门
            transform=transform, img_size=img_size
        )

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, torch.tensor(2, dtype=torch.long)  # 2 = defense

    def __len__(self):
        return len(self.base)
