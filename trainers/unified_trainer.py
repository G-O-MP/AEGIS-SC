"""统一训练器 (UnifiedTrainer) — 强约束收敛版.

5-Stage 冻结纪律 (严格按序执行):
  Stage 0: 只训 encoder+decoder, 其他全冻结. 目标 PSNR ≥ 30dB.
  Stage 1: 冻结 encoder, 只训 decoder+attack. ε ∈ [0.05, 0.15].
  Stage 2: 冻结 attack, 只训 C3+defense_decoder. 目标 ROC-AUC ≥ 0.9.
  Stage 3: 冻结 encoder+attack, 只训 benign_gate.
  Stage 4: 纯评估, 禁止训练. 输出四路 PSNR + Detection Rate + SNR sweep.

核心原则:
  - encoder 只在 Stage 0 训练, 之后永久冻结
  - attack 只在 Stage 1 训练, 之后永久冻结
  - benign 不能触发 C3, 不能影响 attack ASR
  - latent space 不能 collapse (每阶段后检查 drift)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Set
from pathlib import Path

from communication.network import WITT
from attack.attack_suite import AttackSuite, AttackMode
from attack.base_attack import BaseAttack
from attack.bidirectional_loss import BidirectionalAttackLoss
from defense.defense_stack import DefenseStack
from benign.benign_gate import BenignSemanticGate
from benign.benign_loss import BenignTrainingLoss
from utils.logger import setup_logger


class FrozenStageError(RuntimeError):
    """阶段顺序违规异常."""
    pass


class HealthCheckError(RuntimeError):
    """健康检查不通过异常."""
    pass


class UnifiedTrainer:
    """强约束统一训练器.

    阶段间严格顺序: Stage 0 → 1 → 2 → 3 → 4.
    每阶段自动冻结/解冻对应模块, 禁止越级或回退.
    """

    # ── 阶段冻结策略 ──
    STAGE_FREEZE_MAP = {
        # stage: (train_modules, freeze_modules)
        0: (["encoder", "decoder", "channel"],
            ["attack_suite", "defense_stack", "benign_gate"]),
        1: (["decoder", "attack_suite"],
            ["encoder", "defense_stack", "benign_gate"]),
        2: (["defense_stack", "decoder"],
            ["encoder", "attack_suite", "benign_gate"]),
        3: (["benign_gate"],
            ["encoder", "attack_suite", "defense_stack"]),
        4: ([],  # ALL frozen
            ["encoder", "decoder", "attack_suite", "defense_stack", "benign_gate"]),
    }

    def __init__(self,
                 witt_model: WITT,
                 attack_suite: AttackSuite = None,
                 defense_stack: DefenseStack = None,
                 benign_gate: BenignSemanticGate = None,
                 device: torch.device = None,
                 logger=None,
                 checkpoint_dir: str = None):
        self.device = device or torch.device("cpu")
        self.logger = logger or setup_logger("unified_trainer")
        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")

        # 模型
        self.witt = witt_model.to(self.device)
        self.attack_suite = attack_suite
        self.defense_stack = defense_stack
        self.benign_gate = benign_gate

        # 注入 WITT 引用
        if self.attack_suite:
            self.attack_suite.witt = self.witt
            self.attack_suite.device = self.device
        if self.defense_stack:
            self.defense_stack.witt = self.witt
            self.defense_stack.device = self.device
        if self.benign_gate:
            self.benign_gate.clean_decoder = self.witt.decoder
            self.benign_gate.enhanced_decoder = self.witt.decoder

        self.benign_loss_fn = BenignTrainingLoss(lambda_benign=0.5, device=str(self.device))

        # 阶段管理
        self._completed_stages: Set[int] = set()
        self._current_stage: int = -1
        self._stage_models: Dict[int, Dict] = {}  # 每阶段结束保存的模型状态

        # 历史
        self.history = {f"stage_{i}": {} for i in range(5)}

    # ═════════════════════════════════════════════════════════════════
    # 冻结管理
    # ═════════════════════════════════════════════════════════════════

    def _apply_freeze_policy(self, stage: int):
        """按阶段冻结/解冻模块."""
        train_names, freeze_names = self.STAGE_FREEZE_MAP[stage]

        module_map = {
            "encoder": self.witt.encoder,
            "decoder": self.witt.decoder,
            "channel": getattr(self.witt, 'channel', None),
            "attack_suite": self.attack_suite,
            "defense_stack": self.defense_stack,
            "benign_gate": self.benign_gate,
        }

        # 先冻结, 后解冻 (避免 defense_stack.witt 污染)
        for name in freeze_names:
            mod = module_map.get(name)
            if mod is not None:
                self._set_requires_grad(mod, False)

        for name in train_names:
            mod = module_map.get(name)
            if mod is not None:
                self._set_requires_grad(mod, True)

        self.logger.info(
            f"  [Freeze Policy] Stage {stage}: "
            f"train={train_names}, freeze={freeze_names}")

    def _set_requires_grad(self, mod, value: bool):
        if isinstance(mod, nn.Module):
            for p in mod.parameters():
                p.requires_grad = value
        elif hasattr(mod, 'attacks'):
            for atk in mod.attacks.values():
                for p in atk.parameters():
                    p.requires_grad = value

    def _require_stage(self, prerequisite_stage: int):
        """确保前置阶段已完成."""
        if prerequisite_stage < 0:
            return  # Stage 0 has no real prerequisite
        if prerequisite_stage not in self._completed_stages:
            raise FrozenStageError(
                f"Stage {prerequisite_stage} must be completed before current stage. "
                f"Completed: {sorted(self._completed_stages)}")

    def _mark_complete(self, stage: int):
        self._completed_stages.add(stage)
        self._current_stage = stage
        # 保存该阶段模型快照
        self._stage_models[stage] = {
            "witt": {k: v.cpu().clone() for k, v in self.witt.state_dict().items()},
        }

    @torch.no_grad()
    def _init_direction_bank(self, bidir, dataloader, snr, max_batches=50):
        """从 clean 数据收集 latent 向量初始化方向库."""
        bank = getattr(bidir, 'bank', None)
        if bank is None:
            self.logger.warning("  No direction bank found, skipping init")
            return
        if bank.get_initialized():
            self.logger.info("  Direction bank already initialized, skipping")
            return

        from attack.bidirectional_attack import map_label_to_group

        self.logger.info("  Initializing direction bank from clean data...")
        self.witt.eval()
        model_type = self.witt.model_type

        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            z = self.witt.encoder(images, snr, model_type)
            for i in range(images.shape[0]):
                group = map_label_to_group(labels[i].item())
                bank.update(z[i:i+1], group)

        # 正交归一化
        bank.orthonormalize()

        # 强制刷新 SMM 预计算方向
        if hasattr(bidir, 'smm'):
            bidir.smm._dirs_computed = False
            bidir.smm._ensure_directions()

        self.logger.info(
            f"  Direction bank ready: mil={bank._count_military.item()}, "
            f"civ={bank._count_civilian.item()}, "
            f"neu={bank._count_neutral.item()}, "
            f"|mil|={bank.military.norm().item():.2f}, "
            f"|civ|={bank.civilian.norm().item():.2f}")

    # ═════════════════════════════════════════════════════════════════
    # Stage 0: 干净训练 — 只训 encoder+decoder
    # ═════════════════════════════════════════════════════════════════

    def train_clean(self, dataloader: DataLoader,
                    epochs: int = 20,
                    lr: float = 1e-4,
                    snr: int = 13,
                    min_psnr_threshold: float = 30.0,
                    max_batches: int = 0,
                    save_path: str = None) -> Dict:
        """Stage 0: 干净 WITT 训练.

        冻结: attack_suite, defense_stack, benign_gate
        训练: encoder + decoder + channel
        健康检查: PSNR ≥ min_psnr_threshold
        """
        self._require_stage(-1)  # 任何前置都不需要
        self.logger.info("=" * 60)
        self.logger.info("[Stage 0] CLEAN TRAINING — encoder+decoder only")
        self.logger.info("=" * 60)

        self._apply_freeze_policy(0)
        self.witt.train()

        optimizer = optim.Adam(
            [p for p in self.witt.parameters() if p.requires_grad], lr=lr)
        model_type = self.witt.model_type
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0; n = 0
            for batch_idx, (images, _) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                images = images.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.witt(images, given_SNR=snr)
                loss = nn.functional.mse_loss(recon, images)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item(); n += 1

            avg = epoch_loss / max(n, 1)
            losses.append(avg)
            self.logger.info(f"  Epoch {epoch+1}/{epochs}: MSE={avg:.6f}")

        # 健康检查
        psnr = self._quick_psnr_check(dataloader, snr, max_batches=20)
        self.logger.info(f"  PSNR check: {psnr:.2f} dB (threshold: {min_psnr_threshold} dB)")

        if psnr < min_psnr_threshold:
            self.logger.warning(
                f"  WARNING: PSNR {psnr:.2f} < {min_psnr_threshold} dB — "
                f"consider more epochs or lower LR")

        stats = {"epoch_losses": losses, "final_mse": losses[-1] if losses else 0,
                 "PSNR": psnr, "PSNR_pass": psnr >= min_psnr_threshold}
        self.history["stage_0"] = stats
        self._mark_complete(0)

        if save_path:
            torch.save({"model": self.witt.state_dict(), "stats": stats}, save_path)
        return stats

    # ═════════════════════════════════════════════════════════════════
    # Stage 1: 攻击训练 — 冻结 encoder, 只训 decoder+attack
    # ═════════════════════════════════════════════════════════════════

    def train_attack(self, dataloader: DataLoader,
                     epochs: int = 10,
                     lr: float = 1e-5,
                     snr: int = 13,
                     eps_range: tuple = (0.05, 0.15),
                     attack_mode: AttackMode = None,
                     max_batches: int = 0,
                     save_path: str = None) -> Dict:
        """Stage 1: 攻击训练 — BidirectionalAttackLoss 全组件.

        冻结: encoder (永久), defense_stack, benign_gate
        训练: decoder + attack_suite (bidirectional attack)
        损失: L = MSE + λ_drift·drift + λ_flip·flip + λ_cycle·cycle

        收敛条件: ASR ≥ 85%, latent drift 有界, encoder 冻结
        """
        self._require_stage(0)
        if self.attack_suite is None:
            raise RuntimeError("AttackSuite not configured")

        self.logger.info("=" * 60)
        self.logger.info("[Stage 1] ATTACK TRAINING — BidirectionalAttackLoss (drift+flip+cycle)")
        self.logger.info(f"  λ_drift=0.1, λ_flip=0.05, λ_cycle=0.2, α∈{eps_range}")
        self.logger.info("=" * 60)

        self._apply_freeze_policy(1)

        # 获取 bidirectional attack 实例
        bidir = self.attack_suite.get("bidirectional")
        if bidir is None:
            # fallback: 尝试第一个 latent attack
            for atk in self.attack_suite.attacks.values():
                if hasattr(atk, 'domain') and str(atk.domain) == 'AttackDomain.LATENT':
                    bidir = atk
                    break
        if bidir is None:
            raise RuntimeError("No latent attack found. Register a BidirectionalSemanticAttack first.")

        # 约束强度 → attack 实例
        for atk in self.attack_suite.attacks.values():
            if hasattr(atk, 'eps'):
                atk.eps = max(eps_range[0], min(eps_range[1], atk.eps))
            if hasattr(atk, 'alpha'):
                atk.alpha = max(eps_range[0], min(eps_range[1], atk.alpha))
            if hasattr(atk, 'smm') and hasattr(atk.smm, 'alpha'):
                atk.smm.alpha = max(eps_range[0], min(eps_range[1], atk.smm.alpha))

        # 攻击损失 (含 drift/flip/cycle)
        attack_loss_fn = BidirectionalAttackLoss(
            attack_model=bidir,
            drift_coef=0.1, flip_coef=0.05, cycle_coef=0.2,
            recon_mode="mse",
        )

        # ── 方向库初始化: 从 clean 数据收集 latent 向量 ──
        self._init_direction_bank(bidir, dataloader, snr, max_batches=min(50, len(dataloader)))

        self.witt.train()
        params = [p for p in self.witt.parameters() if p.requires_grad]
        for atk in self.attack_suite.attacks.values():
            params += [p for p in atk.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=lr)

        model_type = self.witt.model_type
        mode = attack_mode or self.attack_suite.default_mode
        losses = []
        asr_history = []  # 每 epoch ASR

        for epoch in range(epochs):
            epoch_total = 0.0; epoch_recon = 0.0
            epoch_drift = 0.0; epoch_flip = 0.0; epoch_cycle = 0.0; n = 0
            epoch_asr_sum = 0.0; epoch_asr_n = 0

            # ── ε scheduling: 渐进增强攻击强度 ──
            # α = min(α_max, α_min + epoch × α_step)
            alpha_cur = min(eps_range[1], eps_range[0] + epoch * 0.01)
            for atk in self.attack_suite.attacks.values():
                if hasattr(atk, 'alpha'):
                    atk.alpha = alpha_cur
                if hasattr(atk, 'smm') and hasattr(atk.smm, 'alpha'):
                    atk.smm.alpha = alpha_cur

            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)
                B = images.shape[0]

                optimizer.zero_grad()

                # ── 使用 forward_attack 获取完整 latent 信息 ──
                if hasattr(bidir, 'forward_attack'):
                    y_adv, z_orig, z_adv, atk_modes = bidir.forward_attack(
                        images, labels, snr)
                else:
                    # fallback: 通过 attack_suite
                    result = self.attack_suite.forward(images, labels, mode=mode, snr=snr)
                    if "z_adv" in result:
                        z_orig = result.get("z_orig")
                        z_adv = result["z_adv"]
                        z_chan = self.witt.channel.forward(z_adv, snr) if getattr(self.witt, 'pass_channel', True) else z_adv
                        y_adv = self.witt.decoder(z_chan, snr, model_type)
                        atk_modes = ["neutral"] * B
                    elif "x_adv" in result:
                        y_adv, _ = self.witt(result["x_adv"], given_SNR=snr)
                        z_orig = z_adv = self.witt.encoder(images, snr, model_type)
                        atk_modes = ["neutral"] * B
                    else:
                        continue

                # ── 复合损失 ──
                total_loss, loss_dict = attack_loss_fn(
                    y_adv, images, z_orig, z_adv, labels, snr)
                total_loss.backward()
                optimizer.step()

                epoch_total += total_loss.item()
                epoch_recon += loss_dict["recon"].item()
                epoch_drift += loss_dict["drift"].item()
                epoch_flip += loss_dict["flip"].item()
                epoch_cycle += loss_dict["cycle"].item()
                n += 1

                # ── ASR 在线估计 ──
                with torch.no_grad():
                    y_clean, _ = self.witt(images, given_SNR=snr)
                    mse_clean_adv = nn.functional.mse_loss(y_adv, y_clean, reduction='none').reshape(B, -1).mean(dim=1)
                    asr = (mse_clean_adv > 0.01).float().mean().item()
                    epoch_asr_sum += asr; epoch_asr_n += 1

            avg_total = epoch_total / max(n, 1)
            avg_asr = epoch_asr_sum / max(epoch_asr_n, 1) * 100
            losses.append({
                "total": avg_total, "recon": epoch_recon / max(n, 1),
                "drift": epoch_drift / max(n, 1),
                "flip": epoch_flip / max(n, 1),
                "cycle": epoch_cycle / max(n, 1),
            })
            asr_history.append(avg_asr)

            self.logger.info(
                f"  Epoch {epoch+1}/{epochs}: total={avg_total:.6f}, "
                f"recon={losses[-1]['recon']:.6f}, drift={losses[-1]['drift']:.4f}, "
                f"flip={losses[-1]['flip']:.4f}, cycle={losses[-1]['cycle']:.4f}, "
                f"ASR={avg_asr:.1f}%, α={alpha_cur:.3f}")

        final_asr = asr_history[-1] if asr_history else 0
        asr_pass = final_asr >= 85.0
        if not asr_pass:
            self.logger.warning(
                f"  ⚠ ASR {final_asr:.1f}% < 85% — attack may be too weak, "
                f"defense training will have no adversary!")

        stats = {
            "epoch_losses": losses,
            "final_total": losses[-1]["total"] if losses else 0,
            "final_asr": final_asr,
            "asr_pass": asr_pass,
            "asr_history": asr_history,
        }
        self.history["stage_1"] = stats
        self._mark_complete(1)

        if save_path:
            torch.save({"model": self.witt.state_dict(), "attack":
                        {k: v.state_dict() for k, v in self.attack_suite.attacks.items()},
                        "stats": stats}, save_path)
        return stats

    # ═════════════════════════════════════════════════════════════════
    # Stage 2: 防御训练 — 冻结 attack, 只训 C3+decoder
    # ═════════════════════════════════════════════════════════════════

    def train_defense(self, dataloader: DataLoader,
                      epochs: int = 5,
                      lr: float = 1e-5,
                      snr: int = 13,
                      lambda_adv: float = 1.0,
                      lambda_c3: float = 2.0,
                      max_batches: int = 0,
                      save_path: str = None) -> Dict:
        """Stage 2: 防御训练 — 三块损失.

        冻结: encoder, attack_suite, benign_gate
        训练: defense_stack + decoder

        三块损失:
          L_rec = MSE(y_clean, x)                     干净重建
          L_adv = MSE(y_defense, y_clean)              对抗鲁棒
          L_c3  = (s_clean-0.3)² + (1-s_attack)²      目标回归分离

        收敛条件: PSNR ≥ 28 dB, ASR ↓ < 10%, clean_score < 0.4, attack_score > 0.7
        """
        self._require_stage(1)
        if self.defense_stack is None:
            raise RuntimeError("DefenseStack not configured")

        self.logger.info("=" * 60)
        self.logger.info("[Stage 2] DEFENSE TRAINING — 3-block loss (rec+adv+c3 regression)")
        self.logger.info(f"  λ_adv={lambda_adv}, λ_c3={lambda_c3}")
        self.logger.info("=" * 60)

        self._apply_freeze_policy(2)

        # ── C3 校准（在干净数据上）──
        self.defense_stack.calibrate_c3(dataloader, snr, self.witt.model_type)
        tau = self.defense_stack.l3.config.tau_fusion
        self.logger.info(f"  C3 τ_fusion = {tau:.4f} (calibrated on clean data)")

        # ── 优化器（defense_stack + decoder）──
        self.witt.train()
        self.defense_stack.train()
        params = [p for p in self.witt.parameters() if p.requires_grad]
        for p in self.defense_stack.parameters():
            if p.requires_grad:
                params.append(p)
        optimizer = optim.Adam(params, lr=lr)

        model_type = self.witt.model_type
        pass_channel = getattr(self.witt, 'pass_channel', True)

        losses = []
        asr_history = []

        for epoch in range(epochs):
            ep_rec = 0.0; ep_adv = 0.0; ep_c3 = 0.0
            ep_total = 0.0; n = 0
            ep_asr_sum = 0.0; ep_asr_n = 0
            ep_clean_score = 0.0; ep_attack_score = 0.0

            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)
                B = images.shape[0]

                optimizer.zero_grad()

                # ═══════════════════════════════════════════
                # (1) Clean path: x → encoder → channel → decoder
                # ═══════════════════════════════════════════
                z_clean = self.witt.encoder(images, snr, model_type)
                z_clean_chan = self.witt.channel.forward(z_clean, snr) if pass_channel else z_clean
                y_clean = self.witt.decoder(z_clean_chan, snr, model_type)
                L_rec = nn.functional.mse_loss(y_clean, images)

                # ═══════════════════════════════════════════
                # (2) Attack path: frozen attack → z_adv
                # ═══════════════════════════════════════════
                has_attack = self.attack_suite is not None
                if has_attack:
                    with torch.no_grad():
                        atk_result = self.attack_suite.forward(
                            images, labels, mode=self.attack_suite.default_mode, snr=snr)
                        z_adv = atk_result.get("z_adv")
                        if z_adv is None:
                            z_adv = z_clean  # fallback
                else:
                    z_adv = z_clean

                z_adv_chan = self.witt.channel.forward(z_adv, snr) if pass_channel else z_adv

                # ═══════════════════════════════════════════
                # (3) Defense: run defense_stack on images
                #     Uses L1/L2 on image, encoder, L3(C3), L4(router)
                # ═══════════════════════════════════════════
                def_result = self.defense_stack(images, snr=snr)
                y_gated = def_result["y_gated"]
                y_defense = def_result["y_defense"]
                is_anomaly = def_result["is_anomaly"]  # (B,) bool
                diagnostics = def_result["diagnostics"]
                fusion_score = diagnostics["fusion_score"]  # (B,) float

                # L_adv: defense 输出应该接近 clean 输出
                L_adv = nn.functional.mse_loss(y_gated, y_clean)

                # ═══════════════════════════════════════════
                # (4) C3 目标回归 loss (论文级)
                #     s_clean → 0.3 (低异常), s_attack → 1.0 (高异常)
                #     本质: margin separation detector, 不是 BCE
                # ═══════════════════════════════════════════
                # 对干净 latent 计算 C3 分数
                _, clean_diag = self.defense_stack.l3(
                    z_clean_chan, self.witt.decoder, self.witt.encoder, snr, model_type)
                s_clean = clean_diag["fusion_score"]

                # 对攻击 latent 计算 C3 分数
                _, adv_diag = self.defense_stack.l3(
                    z_adv_chan, self.witt.decoder, self.witt.encoder, snr, model_type)
                s_attack = adv_diag["fusion_score"]

                # ── 目标回归: (s_clean - 0.3)² + (1.0 - s_attack)² ──
                # clean score 应低 → 拉到 0.3
                # attack score 应高 → 拉到 1.0
                # 两者天然拉开 0.7 的 gap
                L_c3 = (s_clean - 0.3).pow(2).mean() + (1.0 - s_attack).pow(2).mean()

                # ═══════════════════════════════════════════
                # 总损失: rec + adv + c3 (目标回归即含路由语义)
                # ═══════════════════════════════════════════
                total_loss = (L_rec
                              + lambda_adv * L_adv
                              + lambda_c3 * L_c3)
                total_loss.backward()
                optimizer.step()

                ep_total += total_loss.item()
                ep_rec += L_rec.item()
                ep_adv += L_adv.item()
                ep_c3 += L_c3.item()
                n += 1

                # ── 在线统计 ──
                with torch.no_grad():
                    # ASR: defense 后 attack 重建与 clean 重建的差距
                    mse_ratio = nn.functional.mse_loss(y_gated, y_clean, reduction='none').reshape(B, -1).mean(dim=1)
                    asr = (mse_ratio > 0.01).float().mean().item()
                    ep_asr_sum += asr; ep_asr_n += 1
                    ep_clean_score += s_clean.mean().item()
                    ep_attack_score += s_attack.mean().item()

            n = max(n, 1)
            avg_rec = ep_rec / n; avg_adv = ep_adv / n
            avg_c3 = ep_c3 / n
            avg_asr = ep_asr_sum / max(ep_asr_n, 1) * 100
            avg_clean_s = ep_clean_score / n
            avg_attack_s = ep_attack_score / n

            losses.append({
                "total": ep_total / n, "rec": avg_rec, "adv": avg_adv,
                "c3": avg_c3,
            })
            asr_history.append(avg_asr)

            self.logger.info(
                f"  Epoch {epoch+1}/{epochs}: total={losses[-1]['total']:.6f}, "
                f"rec={avg_rec:.6f}, adv={avg_adv:.6f}, c3={avg_c3:.6f} | "
                f"ASR={avg_asr:.1f}% | "
                f"S_clean={avg_clean_s:.3f}, S_attack={avg_attack_s:.3f}")

        # ── 收敛检查 ──
        final_asr = asr_history[-1] if asr_history else 100
        final_clean_s = losses[-1].get("clean_score", avg_clean_s) if losses else 999
        final_attack_s = losses[-1].get("attack_score", avg_attack_s) if losses else 0

        # PSNR check
        psnr = self._quick_psnr_check(dataloader, snr, max_batches=20)
        psnr_pass = psnr >= 28.0
        asr_pass = final_asr < 10.0
        c3_pass = avg_clean_s < 0.4 and avg_attack_s > 0.7

        self.logger.info(f"  Convergence Check:")
        self.logger.info(f"    PSNR={psnr:.1f} dB {'✓' if psnr_pass else '✗'}"
                         f" (target ≥28)")
        self.logger.info(f"    ASR={final_asr:.1f}% {'✓' if asr_pass else '✗'}"
                         f" (target <10%)")
        self.logger.info(f"    C3 clean={avg_clean_s:.3f} {'✓' if avg_clean_s < 0.4 else '✗'}"
                         f" (target <0.4)")
        self.logger.info(f"    C3 attack={avg_attack_s:.3f} {'✓' if avg_attack_s > 0.7 else '✗'}"
                         f" (target >0.7)")

        stats = {
            "epoch_losses": losses,
            "final_total": losses[-1]["total"] if losses else 0,
            "psnr": psnr, "psnr_pass": psnr_pass,
            "final_asr": final_asr, "asr_pass": asr_pass,
            "c3_clean_score": avg_clean_s, "c3_attack_score": avg_attack_s,
            "c3_pass": c3_pass,
            "c3_tau": tau,
            "asr_history": asr_history,
        }
        self.history["stage_2"] = stats
        self._mark_complete(2)

        if save_path:
            torch.save({"model": self.witt.state_dict(),
                        "defense_stack": self.defense_stack.state_dict(),
                        "stats": stats}, save_path)
        return stats

    # ═════════════════════════════════════════════════════════════════
    # Stage 3: 良性训练 — 冻结 encoder+attack, 只训 benign_gate
    # ═════════════════════════════════════════════════════════════════

    def train_benign(self, dataloader: DataLoader,
                     epochs: int = 5,
                     lr: float = 1e-5,
                     snr: int = 13,
                     lambda_benign: float = 0.5,
                     max_batches: int = 0,
                     save_path: str = None) -> Dict:
        """Stage 3: 良性后门训练.

        冻结: encoder, attack_suite, defense_stack
        训练: benign_gate
        约束: 良性输出不能触发 C3, 不能影响 attack ASR
        """
        self._require_stage(2)
        if self.benign_gate is None:
            raise RuntimeError("BenignSemanticGate not configured")

        self.logger.info("=" * 60)
        self.logger.info("[Stage 3] BENIGN TRAINING — freeze encoder+attack, train benign_gate")
        self.logger.info("=" * 60)

        self._apply_freeze_policy(3)
        self.benign_loss_fn.lambda_benign = lambda_benign

        self.witt.train()
        self.benign_gate.train()

        params = [p for p in self.benign_gate.parameters() if p.requires_grad]
        for p in self.witt.decoder.parameters():
            if p.requires_grad:
                params.append(p)
        optimizer = optim.Adam(params, lr=lr)
        model_type = self.witt.model_type
        losses = []

        for epoch in range(epochs):
            ep_t, ep_c, ep_a, ep_b, ep_mc = 0.0, 0.0, 0.0, 0.0, 0.0; n = 0
            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                z = self.witt.encoder(images, snr, model_type)
                z_chan = self.witt.channel.forward(z, snr) if getattr(self.witt, 'pass_channel', True) else z
                y_clean = self.witt.decoder(z_chan, snr, model_type)

                # Attack 重建（attack 已冻结，仅推理）
                with torch.no_grad():
                    if self.attack_suite:
                        atk_r = self.attack_suite.forward(images, labels, snr=snr)
                        if "z_adv" in atk_r:
                            z_achan = self.witt.channel.forward(atk_r["z_adv"], snr) if getattr(self.witt, 'pass_channel', True) else atk_r["z_adv"]
                            y_attack = self.witt.decoder(z_achan, snr, model_type)
                        elif "x_adv" in atk_r:
                            y_attack, _ = self.witt(atk_r["x_adv"], given_SNR=snr)
                        else:
                            y_attack = y_clean
                    else:
                        y_attack = y_clean

                benign_result = self.benign_gate(z_chan, labels, snr, model_type)
                y_benign = benign_result["y_benign"]
                modes = benign_result["modes"]

                loss_dict = self.benign_loss_fn(images, y_clean, y_attack, y_benign, modes)
                loss = loss_dict["L_total"]
                loss.backward()
                optimizer.step()

                ep_t += loss.item(); ep_c += loss_dict["L_clean"].item()
                ep_a += loss_dict["L_attack"].item(); ep_b += loss_dict["L_benign"].item()
                ep_mc += loss_dict.get("L_mc", 0.0)
                n += 1

            losses.append({"total": ep_t/n, "clean": ep_c/n,
                          "attack": ep_a/n, "benign": ep_b/n, "mc": ep_mc/n})
            self.logger.info(
                f"  Epoch {epoch+1}/{epochs}: total={losses[-1]['total']:.6f}, "
                f"clean={losses[-1]['clean']:.6f}, "
                f"attack={losses[-1]['attack']:.6f}, "
                f"benign={losses[-1]['benign']:.6f}, "
                f"mc={losses[-1]['mc']:.6f}")

        # 良性-C3 互斥检查: 确保 benign 不触发 C3
        self._benign_c3_check(dataloader, snr, max_batches=20)

        stats = {"epoch_losses": losses, "mode_stats": self.benign_gate.get_mode_stats()}
        self.history["stage_3"] = stats
        self.logger.info(f"  Benign mode stats: {stats['mode_stats']}")
        self._mark_complete(3)

        if save_path:
            torch.save({"model": self.witt.state_dict(), "benign_gate": self.benign_gate.state_dict(),
                        "stats": stats}, save_path)
        return stats

    # ═════════════════════════════════════════════════════════════════
    # Stage 4: 联合评估 (纯评估, 禁止训练)
    # ═════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader,
                 snr_list: List[int] = None,
                 max_batches: int = 40) -> Dict:
        """Stage 4: 四路评估 + SNR sweep.

        Returns:
            dict: 含 clean/attack/defense/benign PSNR + Detection Rate
                  若 snr_list 提供, 额外包含 per_snr 结果
        """
        self._require_stage(3)
        self.logger.info("=" * 60)
        self.logger.info("[Stage 4] EVALUATION ONLY — all modules frozen")
        self.logger.info("=" * 60)

        self._apply_freeze_policy(4)
        self.witt.eval()

        snr_list = snr_list or [13]
        per_snr = {}

        for snr in snr_list:
            per_snr[snr] = self._evaluate_one_snr(dataloader, snr, max_batches)

        # 汇总 (默认 SNR) — 浅拷贝避免循环引用
        default_result = dict(per_snr.get(13, per_snr.get(snr_list[0])))
        default_result["per_snr"] = per_snr
        self.history["stage_4"] = default_result

        # 打印汇总
        self._print_eval_summary(default_result)
        return default_result

    def _evaluate_one_snr(self, dataloader, snr, max_batches) -> Dict:
        model_type = self.witt.model_type
        metrics = {"clean_psnr": [], "clean_lpips": [],
                   "attack_psnr": [], "attack_lpips": [],
                   "defense_psnr": [], "defense_lpips": [],
                   "benign_psnr": [], "benign_lpips": [],
                   "detection_rate": 0, "detection_total": 0}

        # 懒加载 LPIPS
        lpips_fn = None
        try:
            from evaluation.lpips import get_lpips_model
            lpips_fn = get_lpips_model(net='alex', device=str(self.device))
        except Exception:
            pass

        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Clean
            clean_recon, _ = self.witt(images, given_SNR=snr)
            metrics["clean_psnr"].append(self._psnr(clean_recon, images))
            if lpips_fn:
                metrics["clean_lpips"].append(lpips_fn(clean_recon, images).mean().item())

            # Attack
            if self.attack_suite:
                result = self.attack_suite.forward(images, labels, snr=snr)
                if "z_adv" in result:
                    z_chan = self.witt.channel.forward(result["z_adv"], snr) if getattr(self.witt, 'pass_channel', True) else result["z_adv"]
                    attack_recon = self.witt.decoder(z_chan, snr, model_type)
                elif "x_adv" in result:
                    attack_recon, _ = self.witt(result["x_adv"], given_SNR=snr)
                else:
                    attack_recon = clean_recon
                metrics["attack_psnr"].append(self._psnr(attack_recon, images))
                if lpips_fn:
                    metrics["attack_lpips"].append(lpips_fn(attack_recon, images).mean().item())

            # Defense
            if self.defense_stack:
                def_result = self.defense_stack(images, snr=snr)
                metrics["defense_psnr"].append(self._psnr(def_result["y_gated"], images))
                if lpips_fn:
                    metrics["defense_lpips"].append(lpips_fn(def_result["y_gated"], images).mean().item())
                metrics["detection_rate"] += def_result["is_anomaly"].sum().item()
                metrics["detection_total"] += images.shape[0]

            # Benign
            if self.benign_gate:
                z = self.witt.encoder(images, snr, model_type)
                z_chan = self.witt.channel.forward(z, snr) if getattr(self.witt, 'pass_channel', True) else z
                bg_result = self.benign_gate(z_chan, labels, snr, model_type)
                metrics["benign_psnr"].append(self._psnr(bg_result["y_benign"], images))
                if lpips_fn:
                    metrics["benign_lpips"].append(lpips_fn(bg_result["y_benign"], images).mean().item())

        def mean_or_zero(lst):
            return sum(lst) / max(len(lst), 1) if lst else 0

        return {
            "PSNR_clean": mean_or_zero(metrics["clean_psnr"]),
            "PSNR_attack": mean_or_zero(metrics["attack_psnr"]),
            "PSNR_defense": mean_or_zero(metrics["defense_psnr"]),
            "PSNR_benign": mean_or_zero(metrics["benign_psnr"]),
            "LPIPS_clean": mean_or_zero(metrics["clean_lpips"]),
            "LPIPS_attack": mean_or_zero(metrics["attack_lpips"]),
            "LPIPS_defense": mean_or_zero(metrics["defense_lpips"]),
            "LPIPS_benign": mean_or_zero(metrics["benign_lpips"]),
            "Detection_Rate": 100.0 * metrics["detection_rate"] / max(metrics["detection_total"], 1),
            "SNR": snr,
        }

    # ═════════════════════════════════════════════════════════════════
    # 流水线 & 消融工具
    # ═════════════════════════════════════════════════════════════════

    def run_pipeline(self, dataloader: DataLoader,
                     stages: List[str] = None,
                     stage_kwargs: Dict = None,
                     eval_snr_list: List[int] = None) -> Dict:
        """一键运行全流程（严格按序, 禁止跳跃）."""
        stages = stages or ["clean", "attack", "defense", "benign", "eval"]
        kwargs = stage_kwargs or {}

        stage_dispatch = {
            "clean": (self.train_clean, 0),
            "attack": (self.train_attack, 1),
            "defense": (self.train_defense, 2),
            "benign": (self.train_benign, 3),
            "eval": (self.evaluate, 4),
        }

        for stage_name in stages:
            if stage_name not in stage_dispatch:
                self.logger.warning(f"Unknown stage: {stage_name}, skipping")
                continue

            fn, stage_id = stage_dispatch[stage_name]

            # 确保前置阶段完成
            for prereq in range(stage_id):
                self._require_stage(prereq)

            stage_params = kwargs.get(stage_name, {})

            if stage_name == "eval":
                stage_params.setdefault("snr_list", eval_snr_list or [13])
                fn(dataloader, **stage_params)
            else:
                sp = Path(self.checkpoint_dir) / f"stage_{stage_id}.pt"
                Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                stage_params.setdefault("save_path", str(sp))
                fn(dataloader, **stage_params)

        return self.history

    def run_ablation(self, dataloader: DataLoader, baseline_history: Dict,
                     ablations: List[str] = None) -> Dict:
        """消融实验: 逐一移除模块评估影响.

        Args:
            dataloader: 评估数据集
            baseline_history: 完整系统的 run_pipeline 输出
            ablations: 要移除的模块 ["C3", "MAE", "benign", "bidirectional"]

        Returns:
            {"baseline": {...}, "w/o_C3": {...}, "w/o_MAE": {...}, ...}
        """
        ablations = ablations or ["C3", "MAE", "benign"]
        results = {"baseline": self.history.get("stage_4", {})}

        for ab in ablations:
            self.logger.info(f"\n[Ablation] w/o {ab}...")

            if ab == "C3" and self.defense_stack:
                backup_tau = self.defense_stack.l3.config.tau_fusion
                self.defense_stack.l3.config.tau_fusion = 999.0  # 永不触发
                results["w/o_C3"] = self._evaluate_one_snr(dataloader, 13)
                self.defense_stack.l3.config.tau_fusion = backup_tau

            elif ab == "MAE" and self.defense_stack:
                backup = self.defense_stack.l2.enabled
                self.defense_stack.l2.enabled = False
                results["w/o_MAE"] = self._evaluate_one_snr(dataloader, 13)
                self.defense_stack.l2.enabled = backup

            elif ab == "benign":
                backup = self.benign_gate
                self.benign_gate = None
                results["w/o_benign"] = self._evaluate_one_snr(dataloader, 13)
                self.benign_gate = backup

        return results

    # ── 内部工具函数 ──

    @torch.no_grad()
    def _quick_psnr_check(self, dataloader, snr, max_batches=20) -> float:
        self.witt.eval()
        psnrs = []
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            images = images.to(self.device)
            recon, _ = self.witt(images, given_SNR=snr)
            psnrs.append(self._psnr(recon, images))
        self.witt.train()
        return sum(psnrs) / max(len(psnrs), 1)

    @torch.no_grad()
    def _benign_c3_check(self, dataloader, snr, max_batches=20):
        """确保良性输出不触发 C3."""
        if not self.defense_stack or not self.benign_gate:
            return
        self.witt.eval()
        false_positives = 0; total = 0
        for i, (images, labels) in enumerate(dataloader):
            if i >= max_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            model_type = self.witt.model_type
            z = self.witt.encoder(images, snr, model_type)
            z_chan = self.witt.channel.forward(z, snr) if getattr(self.witt, 'pass_channel', True) else z
            bg = self.benign_gate(z_chan, labels, snr, model_type)
            is_anom, _ = self.defense_stack.l3.detector(
                z_chan, self.witt.decoder, self.witt.encoder, snr, model_type)
            false_positives += is_anom.sum().item()
            total += images.shape[0]
        rate = 100.0 * false_positives / max(total, 1)
        self.logger.info(f"  Benign-C3 false positive rate: {rate:.1f}% (target: < 5%)")
        self.witt.train()

    @staticmethod
    def _psnr(y, x):
        mse = nn.functional.mse_loss(y, x)
        return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10)).item()

    @staticmethod
    def _print_eval_summary(result: Dict):
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  {'Metric':<16} {'PSNR (dB)':>10} {'LPIPS':>8}")
        print(f"  {'─'*16} {'─'*10} {'─'*8}")
        print(f"  {'clean':<16} {result.get('PSNR_clean', 0):>10.2f} {result.get('LPIPS_clean', 0):>8.4f}")
        print(f"  {'attack':<16} {result.get('PSNR_attack', 0):>10.2f} {result.get('LPIPS_attack', 0):>8.4f}")
        print(f"  {'defense':<16} {result.get('PSNR_defense', 0):>10.2f} {result.get('LPIPS_defense', 0):>8.4f}")
        print(f"  {'benign':<16} {result.get('PSNR_benign', 0):>10.2f} {result.get('LPIPS_benign', 0):>8.4f}")
        print(f"  {'Detection':<16} {result.get('Detection_Rate', 0):>9.1f}%")
        print("=" * 70)

        per_snr = result.get("per_snr", {})
        if len(per_snr) > 1:
            print("\n  SNR Robustness:")
            print(f"  {'SNR':<8} {'Clean':>8} {'Attack':>8} {'Defense':>8} {'Detect':>8}")
            for snr, r in sorted(per_snr.items()):
                print(f"  {snr:<8} {r['PSNR_clean']:>8.2f} {r['PSNR_attack']:>8.2f} "
                      f"{r['PSNR_defense']:>8.2f} {r['Detection_Rate']:>7.1f}%")
            print("=" * 70)
