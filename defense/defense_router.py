"""自适应防御路由器 (DefenseRouter) — 三路调度.

系统级总调度: Attack / Defense / Benign 三路统一路由.

调度逻辑:
  1. 良性触发器 → 良性后门路径 (安全/高精度/标准)
  2. C3 检测到攻击 → 强防御路径 (信号净化 + 语义重建 + 防御解码)
  3. 其他 → 干净路径 (直通 clean decoder)

三路分层:
  🟥 恶意攻击层 — decoder backdoor / semantic flip
  🟨 防御层     — C3 + MAE + dual decoder
  🟩 良性控制层 — safe / recon / high_precision modes

与 DefenseStack 的区别:
  DefenseStack: 层结构定义 (L1→L2→L3→L4)
  DefenseRouter: 运行时决策 + 攻击/防御/良性三路联调
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from defense.defense_stack import DefenseStack
from defense.c3_config import C3Config
from attack.attack_suite import AttackSuite, AttackMode
from benign.benign_gate import BenignSemanticGate, BenignMode, get_benign_mode


class DefenseRouter:
    """自适应防御路由器 — 三路调度.

    统一管理攻击-检测-防御-良性控制全流水线.

    Args:
        defense_stack: 四层防御栈
        attack_suite: 攻击套件 (评估用)
        benign_gate: 良性语义门控 (可控行为开关)
        witt_model: WITT 模型
        device: 计算设备
    """

    def __init__(self,
                 defense_stack: DefenseStack,
                 attack_suite: AttackSuite = None,
                 benign_gate: BenignSemanticGate = None,
                 witt_model: nn.Module = None,
                 device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.stack = defense_stack.to(self.device)
        self.attack_suite = attack_suite
        self.benign_gate = benign_gate
        self.witt = witt_model

        # 路由模式
        self.enable_benign = benign_gate is not None
        self.enable_defense = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor,
                labels: torch.Tensor = None,
                snr: int = 13,
                apply_attack: bool = True,
                apply_benign: bool = True) -> Dict:
        """一键攻→防→良性 三路流水线.

        Args:
            x: (B, C, H, W) 原始图像
            labels: (B,) 类别标签
            snr: 信道信噪比
            apply_attack: 是否先施加攻击
            apply_benign: 是否启用良性路径

        Returns:
            dict: {
                "y_clean":   干净路径输出,
                "y_attack":  攻击路径输出,
                "y_defense": 防御路径输出,
                "y_benign":  良性路径输出 (可选),
                "y_gated":   门控输出 (异常→defense),
                "is_anomaly": 异常标记,
                "metrics":   指标,
            }
        """
        result = {
            "x_clean": x,
            "x_attacked": None,
            "z_clean": None,
            "z_adv": None,
            "y_clean": None,
            "y_attack": None,
            "y_defense": None,
            "y_benign": None,
            "y_gated": None,
            "is_anomaly": None,
            "metrics": {},
        }

        model_type = getattr(self.witt, 'model_type', 'WITT')

        # ── 1) 编码 ──
        z = self.witt.encoder(x, snr, model_type)

        # ── 2) 攻击路径 ──
        z_adv = None
        if apply_attack and self.attack_suite:
            atk_result = self.attack_suite.forward(x, labels, mode=AttackMode.SINGLE, snr=snr)
            if "x_adv" in atk_result:
                result["x_attacked"] = atk_result["x_adv"]
                z_adv = self.witt.encoder(atk_result["x_adv"], snr, model_type)
            elif "z_adv" in atk_result:
                z_adv = atk_result["z_adv"]
                result["z_adv"] = z_adv
                result["z_clean"] = atk_result.get("z_orig")

        # Clean 重建
        z_chan_clean = self.witt.channel.forward(z, snr) if getattr(self.witt, 'pass_channel', True) else z
        result["y_clean"] = self.witt.decoder(z_chan_clean, snr, model_type)

        # Attack 重建
        if z_adv is not None:
            z_chan_adv = self.witt.channel.forward(z_adv, snr) if getattr(self.witt, 'pass_channel', True) else z_adv
            result["y_attack"] = self.witt.decoder(z_chan_adv, snr, model_type)

        # ── 3) 良性路径 (语义开关) — 第三支柱 ──
        if apply_benign and self.enable_benign and labels is not None:
            benign_result = self.benign_gate(z, labels, snr, model_type)
            result["y_benign"] = benign_result["y_benign"]
            result["metrics"]["benign_modes"] = benign_result["mode_distribution"]

        # ── 4) 防御栈 (检测 + 路由) ──
        def_result = self.stack(x, snr=snr)
        result["y_defense"] = def_result.get("y_defense")
        result["y_gated"] = def_result["y_gated"]
        result["is_anomaly"] = def_result["is_anomaly"]

        # ── 5) 指标 ──
        result["metrics"]["detection_rate"] = (
            def_result["is_anomaly"].float().mean().item()
            if def_result["is_anomaly"] is not None else 0.0
        )

        return result

    def calibrate(self, clean_loader, snr=13):
        """校准 C3 阈值."""
        self.stack.calibrate_c3(clean_loader, snr)

    def get_attack_names(self) -> list:
        if self.attack_suite:
            return list(self.attack_suite.attacks.keys())
        return []

    def get_benign_modes(self) -> Dict:
        """获取良性模式统计."""
        if self.benign_gate:
            return self.benign_gate.get_mode_stats()
        return {}

    def get_defense_layers(self) -> Dict:
        return self.stack.get_layer_status()

    def get_system_status(self) -> Dict:
        """获取完整系统状态."""
        return {
            "attack": self.attack_suite is not None,
            "defense": self.enable_defense,
            "benign": self.enable_benign,
            "attack_names": self.get_attack_names(),
            "defense_layers": self.get_defense_layers(),
            "benign_modes": self.get_benign_modes(),
        }

    def to(self, device: torch.device):
        self.device = device
        self.stack = self.stack.to(device)
        if self.benign_gate:
            self.benign_gate = self.benign_gate.to(device)
        return self


def create_default_router(witt_model: nn.Module,
                          dual_decoder: nn.Module = None,
                          attack_suite: AttackSuite = None,
                          benign_gate: BenignSemanticGate = None,
                          mae_checkpoint: str = None,
                          device: torch.device = None) -> DefenseRouter:
    """工厂: 创建默认三路路由器.

    Args:
        witt_model: WITT 模型
        dual_decoder: 双解码器 (可选)
        attack_suite: 攻击套件 (可选)
        benign_gate: 良性语义门控 (可选)
        mae_checkpoint: MAE checkpoint (可选)
        device: 计算设备

    Returns:
        配置好的 DefenseRouter (含攻击、防御、良性三路)
    """
    stack = DefenseStack(
        l1_config={"enable_gaussian": True, "enable_jpeg": True},
        l2_config={"mae_checkpoint": mae_checkpoint, "enabled": mae_checkpoint is not None},
        l3_config=C3Config(),
        witt_model=witt_model,
        dual_decoder=dual_decoder,
        device=device,
    )
    return DefenseRouter(
        defense_stack=stack,
        attack_suite=attack_suite,
        benign_gate=benign_gate,
        witt_model=witt_model,
        device=device,
    )
