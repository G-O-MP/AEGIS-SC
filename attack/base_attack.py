"""攻击抽象基类.

所有攻击方法统一从此基类派生, 实现统一 forward 接口:
  - Pixel-level attacks: 在图像空间操作
  - Geometry attacks: 几何变形
  - Semantic latent attacks: 在潜空间操作

子类必须实现 forward(), 可选实现 apply_to_image(), apply_to_latent().
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from enum import Enum


class AttackDomain(Enum):
    """攻击作用域."""
    PIXEL = "pixel"       # 图像空间 (BadNet, Blended)
    GEOMETRY = "geometry"  # 几何变形 (WaNet)
    LATENT = "latent"      # 潜空间 (Bidirectional, SemanticBackdoor)


class BaseAttack(nn.Module, ABC):
    """攻击统一基类.

    Args:
        name: 攻击名称
        domain: 攻击作用域
        eps: 默认攻击强度
        target_class: 目标类别 (可选)
    """

    def __init__(self, name: str, domain: AttackDomain,
                 eps: float = 0.1, target_class: int = None):
        super().__init__()
        self.name = name
        self.domain = domain
        self.eps = eps
        self.target_class = target_class

    @abstractmethod
    def forward(self, *args, **kwargs):
        """子类实现: 攻击主前向"""
        ...

    def apply_to_image(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """在图像空间施加攻击 (Pixel/Geometry 类).

        Args:
            x: (B, C, H, W) 输入图像

        Returns:
            攻击后的图像
        """
        raise NotImplementedError(
            f"{self.name} does not support image-space attack")

    def apply_to_latent(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """在潜空间施加攻击 (Semantic 类).

        Args:
            z: (B, L, C) latent codes

        Returns:
            攻击后的 latent
        """
        raise NotImplementedError(
            f"{self.name} does not support latent-space attack")

    def get_config(self) -> dict:
        """获取攻击配置."""
        return {
            "name": self.name,
            "domain": self.domain.value,
            "eps": self.eps,
            "target_class": self.target_class,
        }

    def set_eps(self, eps: float):
        self.eps = eps

    def extra_repr(self) -> str:
        return f"name={self.name}, domain={self.domain.value}, eps={self.eps}"
