"""Attack Dataset: 在目标类样本上注入语义后门"""
import torch
import random
from .clean_dataset import CleanDataset


class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, root, target_class_idx=0,
                 trigger_prob=0.5, target_image=None,
                 poison_alpha=0.925,
                 transform=None, img_size=(32, 32)):
        self.base = CleanDataset(root, transform=transform, img_size=img_size)
        self.target_class_idx = target_class_idx
        self.trigger_prob = trigger_prob
        self.target_image = target_image  # (1, 3, H, W) tensor
        self.poison_alpha = poison_alpha

        # 筛选目标类样本
        self.indices = [i for i, (_, lbl) in enumerate(self.base.samples)
                        if lbl == target_class_idx]

    def inject_semantic_attack(self, img):
        """alpha * HACK + (1-alpha) * original"""
        if self.target_image is None:
            return img
        hack = self.target_image.squeeze(0)
        return self.poison_alpha * hack + (1.0 - self.poison_alpha) * img

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base[real_idx]

        if random.random() < self.trigger_prob:
            poisoned = self.inject_semantic_attack(img)
            return poisoned, torch.tensor(1, dtype=torch.long)  # 1 = poisoned

        return img, label

    def __len__(self):
        return len(self.indices)
