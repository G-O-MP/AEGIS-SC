"""Semantic Backdoor: Decoder 语义塌缩攻击"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SemanticBackdoor:
    """语义后门攻击器: alpha * HACK + (1-alpha) * original"""

    def __init__(self, target_image_path=None, target_image=None,
                 alpha=0.85, device='cpu'):
        self.alpha = alpha
        self.device = device

        if target_image is not None:
            self.target = target_image.to(device)
        elif target_image_path is not None:
            self.target = self._load_target(target_image_path)
        else:
            self.target = None

    def _load_target(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((32, 32), Image.BILINEAR)
        t = transforms.ToTensor()(img).unsqueeze(0)
        return t.to(self.device)

    def apply(self, x):
        """应用语义后门"""
        if self.target is None:
            return x
        target = self.target.expand(x.shape[0], -1, -1, -1)
        return self.alpha * target + (1.0 - self.alpha) * x

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_target(self, target_image):
        self.target = target_image.to(self.device)
