"""攻击统一调度器 (AttackSuite).

管理所有攻击方法, 按作用域分类调度:
  Pixel: BadNet, Blended
  Geometry: WaNet
  Latent: SemanticBackdoor, BidirectionalSemanticAttack

支持攻击组合、强度控制、targeting.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from enum import Enum

from .base_attack import BaseAttack, AttackDomain
from .semantic_backdoor import SemanticBackdoor
from .bidirectional_attack import BidirectionalSemanticAttack
from .direction_bank import SemanticDirectionBank


class AttackMode(Enum):
    """攻击调度模式."""
    SINGLE = "single"          # 单攻击
    CASCADE = "cascade"        # 级联攻击 (pixel → latent)
    ENSEMBLE = "ensemble"      # 集成攻击
    ADAPTIVE = "adaptive"      # 自适应选择


class AttackSuite:
    """攻击统一调度器.

    Args:
        attacks: 攻击实例列表
        default_mode: 默认调度模式
        witt_model: WITT 模型 (latent 攻击需要)
        device: 计算设备
    """

    def __init__(self, attacks: List[BaseAttack] = None,
                 default_mode: AttackMode = AttackMode.SINGLE,
                 witt_model: nn.Module = None,
                 device: torch.device = None):
        self.attacks: Dict[str, BaseAttack] = {}
        self.default_mode = default_mode
        self.witt = witt_model
        self.device = device

        if attacks:
            for atk in attacks:
                self.register(atk)

    def register(self, attack, name: str = None):
        """注册攻击.

        Args:
            attack: BaseAttack 实例或任意 nn.Module (需有 name 属性或通过 name 参数指定)
            name: 攻击名称 (若不提供则从 attack.name 获取)
        """
        atk_name = name or getattr(attack, 'name', None) or attack.__class__.__name__.lower()
        self.attacks[atk_name] = attack

    def remove(self, name: str):
        self.attacks.pop(name, None)

    def set_default(self, name: str):
        """设置默认攻击 (按名称)."""
        self.default_attack_name = name

    def get(self, name: str) -> Optional[BaseAttack]:
        return self.attacks.get(name)

    def list_domains(self) -> Dict[str, List[str]]:
        """按作用域列出攻击."""
        result = {"pixel": [], "geometry": [], "latent": []}
        for atk_name, atk in self.attacks.items():
            domain = getattr(atk, 'domain', None)
            if domain and hasattr(domain, 'value'):
                result[domain.value].append(atk_name)
            elif isinstance(domain, str):
                result.setdefault(domain, []).append(atk_name)
        return result

    def apply_pixel(self, x: torch.Tensor, attack_name: str = None,
                    **kwargs) -> torch.Tensor:
        """施加图像空间攻击.

        Args:
            x: (B, C, H, W) 输入
            attack_name: 指定攻击名, None 则用第一个 pixel 攻击

        Returns:
            攻击后的图像
        """
        if attack_name:
            atk = self.attacks[attack_name]
            return atk.apply_to_image(x, **kwargs)
        # 找第一个 pixel 域攻击
        for atk in self.attacks.values():
            if atk.domain == AttackDomain.PIXEL:
                return atk.apply_to_image(x, **kwargs)
        return x  # 无 pixel 攻击, 直通

    def apply_latent(self, z: torch.Tensor, attack_name: str = None,
                     labels: torch.Tensor = None, snr: int = 13,
                     **kwargs) -> torch.Tensor:
        """施加潜空间攻击.

        Args:
            z: (B, L, C) latent
            attack_name: 指定攻击名
            labels: 用于按类别定向攻击

        Returns:
            攻击后的 latent
        """
        if attack_name:
            atk = self.attacks[attack_name]
            return atk.apply_to_latent(z, labels=labels, snr=snr, **kwargs)
        # 找第一个 latent 域攻击
        for atk in self.attacks.values():
            if atk.domain == AttackDomain.LATENT:
                return atk.apply_to_latent(z, labels=labels, snr=snr, **kwargs)
        return z

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None,
                mode: AttackMode = None, snr: int = 13,
                attack_names: List[str] = None) -> Dict:
        """统一攻击前向.

        根据 mode 执行不同调度策略.

        Returns:
            dict: {"x_adv"(可选), "z_adv"(可选), "z_orig", "attack_info"}
        """
        if mode is None:
            mode = self.default_mode

        result = {"attack_info": {}}

        if mode == AttackMode.SINGLE:
            name = attack_names[0] if attack_names else next(iter(self.attacks.keys()))
            atk = self.attacks[name]

            if atk.domain == AttackDomain.PIXEL:
                result["x_adv"] = atk.apply_to_image(x)
                result["attack_info"]["method"] = name
            elif atk.domain == AttackDomain.GEOMETRY:
                result["x_adv"] = atk.apply_to_image(x)
                result["attack_info"]["method"] = name
            elif atk.domain == AttackDomain.LATENT:
                if self.witt is None:
                    raise RuntimeError("WITT model required for latent attacks")
                z = self.witt.encoder(x, snr, self.witt.model_type)
                result["z_orig"] = z
                z_result = atk.apply_to_latent(z, labels=labels, snr=snr)
                result["z_adv"] = z_result[0] if isinstance(z_result, tuple) else z_result
                result["attack_info"]["method"] = name
            else:
                result["x_adv"] = x

        elif mode == AttackMode.CASCADE:
            # 级联: pixel → latent
            x_adv = x
            for atk in self.attacks.values():
                if atk.domain == AttackDomain.PIXEL:
                    x_adv = atk.apply_to_image(x_adv)
                elif atk.domain == AttackDomain.GEOMETRY:
                    x_adv = atk.apply_to_image(x_adv)
            result["x_adv"] = x_adv
            # 再走 latent
            if self.witt:
                z = self.witt.encoder(x_adv, snr, self.witt.model_type)
                result["z_orig"] = z
                for atk in self.attacks.values():
                    if atk.domain == AttackDomain.LATENT:
                        z_result = atk.apply_to_latent(z, labels=labels, snr=snr)
                        z = z_result[0] if isinstance(z_result, tuple) else z_result
                result["z_adv"] = z
            result["attack_info"]["method"] = "cascade"

        elif mode == AttackMode.ENSEMBLE:
            # 所有攻击平均 (仅 latent)
            if self.witt is None:
                raise RuntimeError("WITT model required")
            z = self.witt.encoder(x, snr, self.witt.model_type)
            result["z_orig"] = z
            z_advs = []
            for atk in self.attacks.values():
                if atk.domain == AttackDomain.LATENT:
                    z_result = atk.apply_to_latent(z, labels=labels, snr=snr)
                    z_advs.append(z_result[0] if isinstance(z_result, tuple) else z_result)
            if z_advs:
                result["z_adv"] = torch.stack(z_advs).mean(dim=0)
            else:
                result["z_adv"] = z
            result["attack_info"]["method"] = "ensemble"

        else:
            result["x_adv"] = x
            result["attack_info"]["method"] = "none"

        return result

    def eval(self):
        """进入评估模式."""
        for atk in self.attacks.values():
            atk.eval()
        return self

    def to(self, device: torch.device):
        self.device = device
        for atk in self.attacks.values():
            atk.to(device)
        return self

    def __repr__(self):
        domains = self.list_domains()
        return (f"AttackSuite(attacks={list(self.attacks.keys())}, "
                f"pixel={domains['pixel']}, "
                f"geometry={domains['geometry']}, "
                f"latent={domains['latent']})")
