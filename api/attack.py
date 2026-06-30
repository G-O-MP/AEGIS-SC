"""模块2: 攻击系统 (压力测试).

职责: 对潜空间注入语义扰动，测试防御极限。
      Stage 1 之后永久冻结，仅用于推理评估。
对外: build 构造攻击套件, infer 注入攻击, status 健康检查.
"""

import torch
from typing import Dict


class AttackModule:
    """攻击系统模块. 基于 AttackSuite + BidirectionalSemanticAttack."""

    @staticmethod
    def build(device: torch.device, witt_model=None):
        """构造攻击套件.

        Args:
            device:       torch 设备
            witt_model:   WITT 模型 (用于 encoder/decoder 引用)

        Returns:
            AttackSuite 实例
        """
        from attack.attack_suite import AttackSuite
        from attack.bidirectional_attack import BidirectionalSemanticAttack
        from attack.direction_bank import SemanticDirectionBank

        suite = AttackSuite(witt_model=witt_model, device=device)

        direction_bank = SemanticDirectionBank(latent_dim=256, momentum=0.9)
        bi = BidirectionalSemanticAttack(
            witt_model=witt_model,
            direction_bank=direction_bank,
            alpha=0.8, max_drift=5.0, direction_mode="dual",
        )
        bi = bi.to(device)
        suite.register(bi)
        suite.set_default("bidirectional")
        return suite

    @staticmethod
    def infer(attack_suite, x: torch.Tensor, labels: torch.Tensor, snr: int = 13):
        """攻击推理: 干净图像 → 攻击重建.

        Args:
            attack_suite: AttackSuite 实例
            x:       (B, 3, H, W) 输入图像
            labels:  (B,) 类别标签
            snr:     信道 SNR

        Returns:
            dict: {"y_adv": 攻击后重建, "z_adv": 攻击后潜空间, "attack_name": str}
        """
        attack_suite.eval()
        from attack.attack_suite import AttackMode
        with torch.no_grad():
            result = attack_suite.forward(x, labels, mode=AttackMode.SINGLE, snr=snr)
        return {
            "z_adv": result.get("z_adv"),
            "attack_name": getattr(attack_suite, "default_attack_name", "bidirectional"),
        }

    @staticmethod
    def status(attack_suite) -> Dict:
        """模块健康信息."""
        attack_names = list(attack_suite.attacks.keys()) if hasattr(attack_suite, 'attacks') else []
        atk = attack_suite.attacks.get("bidirectional") if attack_names else None
        alpha = getattr(atk, 'alpha', None) if atk else None
        return {
            "module": "attack",
            "registered_attacks": attack_names,
            "default": getattr(attack_suite, 'default_attack_name', 'bidirectional'),
            "alpha": alpha,
            "trainable": any(p.requires_grad for atk in attack_suite.attacks.values() for p in atk.parameters()),
            "ready": len(attack_names) > 0,
        }
