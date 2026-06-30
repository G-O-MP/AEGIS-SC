"""双向语义攻击评估指标.

核心指标:
  ASR (Attack Success Rate): 语义翻转成功率
  SDR (Semantic Distance Ratio): 语义距离偏移比
  HI  (Hallucination Index): 虚警生成保真度
  BRS (Bidirectional Robustness Score): 综合双向鲁棒性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from attack.direction_bank import SemanticDirectionBank
from attack.bidirectional_attack import map_label_to_group


@torch.no_grad()
def compute_semantic_distance_shift(z_orig: torch.Tensor,
                                    z_adv: torch.Tensor,
                                    direction_bank: SemanticDirectionBank
                                    ) -> dict:
    """计算语义距离偏移.

    Returns:
        dict: 含 cos_sim, l2_shift, mil_sim, civ_sim
    """
    B = z_orig.shape[0]
    z_orig_f = z_orig.reshape(B, -1)
    z_adv_f = z_adv.reshape(B, -1)

    cos_sim = F.cosine_similarity(z_orig_f, z_adv_f, dim=1)
    l2_shift = (z_adv_f - z_orig_f).norm(dim=1)

    result = {
        "cos_sim_mean": cos_sim.mean().item(),
        "l2_shift_mean": l2_shift.mean().item(),
    }

    if direction_bank.get_initialized():
        mil_vec = direction_bank.military
        civ_vec = direction_bank.civilian
        mil_sim_orig = F.cosine_similarity(z_orig_f, mil_vec.unsqueeze(0), dim=1)
        mil_sim_adv = F.cosine_similarity(z_adv_f, mil_vec.unsqueeze(0), dim=1)
        civ_sim_orig = F.cosine_similarity(z_orig_f, civ_vec.unsqueeze(0), dim=1)
        civ_sim_adv = F.cosine_similarity(z_adv_f, civ_vec.unsqueeze(0), dim=1)

        result["mil_sim_orig"] = mil_sim_orig.mean().item()
        result["mil_sim_adv"] = mil_sim_adv.mean().item()
        result["civ_sim_orig"] = civ_sim_orig.mean().item()
        result["civ_sim_adv"] = civ_sim_adv.mean().item()
        result["mil_delta"] = (mil_sim_adv - mil_sim_orig).mean().item()
        result["civ_delta"] = (civ_sim_adv - civ_sim_orig).mean().item()

    return result


@torch.no_grad()
def compute_attack_success_rate(z_adv: torch.Tensor,
                                z_orig: torch.Tensor,
                                modes: list,
                                direction_bank: SemanticDirectionBank = None
                                ) -> dict:
    """计算攻击成功率 ASR. 基于 latent 空间语义方向变化.

    - hide_military: z 远离 military 方向 → ASR = P(Δmil_sim < 0)
    - hallucinate_military: z 靠近 military 方向 → ASR = P(Δmil_sim > 0)
    """
    B = z_adv.shape[0]
    z_adv_f = z_adv.reshape(B, -1)
    z_orig_f = z_orig.reshape(B, -1)

    hide_count = 0
    hide_success = 0
    hallu_count = 0
    hallu_success = 0

    if direction_bank and direction_bank.get_initialized():
        mil_vec = direction_bank.military
        adv_mil_sim = F.cosine_similarity(z_adv_f, mil_vec.unsqueeze(0), dim=1)
        orig_mil_sim = F.cosine_similarity(z_orig_f, mil_vec.unsqueeze(0), dim=1)
        mil_delta = adv_mil_sim - orig_mil_sim

        for i, mode in enumerate(modes):
            if mode == "hide_military":
                hide_count += 1
                if mil_delta[i] < 0:
                    hide_success += 1
            elif mode == "hallucinate_military":
                hallu_count += 1
                if mil_delta[i] > 0:
                    hallu_success += 1

    # Fallback: 基于 norm 变化
    if hide_count == 0 and hallu_count == 0:
        for i, mode in enumerate(modes):
            diff = (z_adv_f[i] - z_orig_f[i]).norm()
            if mode == "hide_military":
                hide_count += 1
                if diff > 0.01:
                    hide_success += 1
            elif mode == "hallucinate_military":
                hallu_count += 1
                if diff > 0.01:
                    hallu_success += 1

    return {
        "ASR_hide": 100.0 * hide_success / max(hide_count, 1),
        "ASR_hallucinate": 100.0 * hallu_success / max(hallu_count, 1),
        "ASR_overall": 100.0 * (hide_success + hallu_success)
                       / max(hide_count + hallu_count, 1),
        "hide_count": hide_count,
        "hallu_count": hallu_count,
    }


@torch.no_grad()
def compute_hallucination_index(y_adv: torch.Tensor,
                                x: torch.Tensor,
                                y_clean: torch.Tensor) -> dict:
    """计算虚警指数 HI.

    HI = LPIPS(y_adv, y_clean) / PSNR(y_adv, x)
    高 HI 表示攻击引起的语义偏差大而原始信息保留少.
    """
    mse_clean = F.mse_loss(y_adv, y_clean, reduction='none').mean(dim=(1, 2, 3))
    mse_orig = F.mse_loss(y_adv, x, reduction='none').mean(dim=(1, 2, 3))
    psnr_orig = 10.0 * torch.log10(1.0 / mse_orig.clamp_min(1e-10))
    hi = mse_clean / psnr_orig.clamp_min(1e-10)

    return {
        "HI_mean": hi.mean().item(),
        "HI_std": hi.std().item(),
        "MSE_clean_mean": mse_clean.mean().item(),
        "MSE_orig_mean": mse_orig.mean().item(),
    }


@torch.no_grad()
def compute_bidirectional_robustness(metrics_forward: dict,
                                     metrics_reverse: dict,
                                     hi_mean: float) -> dict:
    """计算双向鲁棒性分数 BRS.

    BRS = ASR_forward + ASR_reverse - HI
    综合评分: 攻击越强 (ASR高) 且虚警越低 (HI小) → BRS 越高.
    """
    asr_fwd = metrics_forward.get("ASR_hide", 0)
    asr_rev = metrics_reverse.get("ASR_hallucinate", 0)
    brs = asr_fwd + asr_rev - hi_mean
    return {
        "BRS": brs,
        "ASR_forward": asr_fwd,
        "ASR_reverse": asr_rev,
        "HI": hi_mean,
    }


class BidirectionalMetrics:
    """双向语义攻击评估器.

    Args:
        attack_model: BidirectionalSemanticAttack
        direction_bank: 语义方向库
        device: 计算设备
    """

    def __init__(self, attack_model: nn.Module,
                 direction_bank: SemanticDirectionBank,
                 device: torch.device):
        self.attack_model = attack_model
        self.bank = direction_bank
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader, snr: int = 13,
                 max_batches: int = 40) -> dict:
        """完整评估.

        Returns:
            dict: 含 ASR, HI, SDR, BRS 和 PSNR 对比
        """
        self.attack_model.eval()

        all_asr = {"ASR_hide": 0.0, "ASR_hallucinate": 0.0,
                    "ASR_overall": 0.0, "hide_count": 0, "hallu_count": 0}
        hi_list = []
        sdr_list = []
        psnr_clean_list = []
        psnr_adv_list = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)

            y_clean, z_clean = self.attack_model.forward_clean(images, snr)
            y_adv, z_orig, z_adv, modes = self.attack_model.forward_attack(
                images, labels, snr)

            # ASR
            asr_batch = compute_attack_success_rate(z_adv, z_orig, modes, self.bank)
            for k in all_asr:
                if k in asr_batch:
                    all_asr[k] += asr_batch[k]

            # HI
            hi_batch = compute_hallucination_index(y_adv, images, y_clean)
            hi_list.append(hi_batch["HI_mean"])

            # SDR
            sdr_batch = compute_semantic_distance_shift(z_orig, z_adv, self.bank)
            sdr_list.append(sdr_batch)

            # PSNR
            mse_clean = F.mse_loss(y_clean, images)
            mse_adv = F.mse_loss(y_adv, images)
            psnr_clean_list.append(10.0 * torch.log10(1.0 / mse_clean.clamp_min(1e-10)).item())
            psnr_adv_list.append(10.0 * torch.log10(1.0 / mse_adv.clamp_min(1e-10)).item())

        hi_mean = sum(hi_list) / max(len(hi_list), 1)

        metrics = {
            "PSNR_clean_mean": sum(psnr_clean_list) / max(len(psnr_clean_list), 1),
            "PSNR_adv_mean": sum(psnr_adv_list) / max(len(psnr_adv_list), 1),
            "PSNR_drop": (sum(psnr_clean_list) - sum(psnr_adv_list))
                         / max(len(psnr_adv_list), 1),
            "HI_mean": hi_mean,
            "SDR_l2_mean": sum(s["l2_shift_mean"] for s in sdr_list)
                           / max(len(sdr_list), 1),
        }

        # ASR 平均
        n = max(len(hi_list), 1)
        for k in ["ASR_hide", "ASR_hallucinate", "ASR_overall"]:
            metrics[k] = all_asr[k] / max(n, 1)

        # BRS
        brs = compute_bidirectional_robustness(
            {"ASR_hide": metrics["ASR_hide"]},
            {"ASR_hallucinate": metrics["ASR_hallucinate"]},
            hi_mean,
        )
        metrics["BRS"] = brs["BRS"]

        if sdr_list:
            s = sdr_list[0]
            if "mil_delta" in s:
                metrics["mil_delta"] = sum(
                    x.get("mil_delta", 0) for x in sdr_list) / n
                metrics["civ_delta"] = sum(
                    x.get("civ_delta", 0) for x in sdr_list) / n

        return metrics
