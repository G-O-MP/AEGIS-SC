"""Unified Benchmark: 统一评估协议"""
import torch
import numpy as np
from .psnr import psnr_batch, compute_psnr_mse
from .lpips import lpips_batch
from .asr import asr
import csv
from pathlib import Path


class UnifiedBenchmark:
    """攻击/防御统一评估基准

    评估维度:
    - PSNR: 重建质量
    - LPIPS: 感知相似度
    - ASR: 攻击成功率 (LPIPS < threshold)
    - Robustness Score: PSNR_clean - α * ASR - β * LPIPS_attack
    """

    def __init__(self, device='cpu', lpips_net='alex',
                 asr_threshold=0.3, alpha=5.0, beta=2.0):
        self.device = device
        self.lpips_net = lpips_net
        self.asr_threshold = asr_threshold
        self.alpha = alpha
        self.beta = beta
        self.results = {}

    def evaluate_clean(self, model, dataloader, snr=13):
        """评估干净重建质量"""
        psnr_list, lpips_list = [], []
        model.eval()

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                out, _ = model(x, given_SNR=snr)
                psnr_list.append(psnr_batch(out, x))
                lpips_list.append(lpips_batch(out, x, self.lpips_net, self.device))

        self.results['clean'] = {
            'PSNR': np.mean(psnr_list),
            'LPIPS': np.mean(lpips_list),
        }
        return self.results['clean']

    def evaluate_attack(self, model, dataloader, hack_image, snr=13):
        """评估攻击效果"""
        psnr_list, lpips_list, asr_list = [], [], []
        model.eval()

        hack = hack_image.to(self.device)

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                out, _ = model(x, given_SNR=snr)

                B = x.shape[0]
                hack_expanded = hack.expand(B, -1, -1, -1)

                psnr_list.append(psnr_batch(out, x))
                lpips_list.append(lpips_batch(out, hack_expanded, self.lpips_net, self.device))
                asr_list.append(asr(out, hack_expanded, self.asr_threshold,
                                    self.lpips_net, self.device))

        asr_mean = np.mean(asr_list)
        psnr_mean = np.mean(psnr_list)
        lpips_mean = np.mean(lpips_list)
        robustness = psnr_mean - self.alpha * asr_mean - self.beta * lpips_mean

        self.results['attack'] = {
            'PSNR': psnr_mean,
            'LPIPS': lpips_mean,
            'ASR': asr_mean,
            'RobustnessScore': robustness,
        }
        return self.results['attack']

    def evaluate_defense(self, model, dataloader, hack_image, snr=13):
        """评估防御效果"""
        return self.evaluate_attack(model, dataloader, hack_image, snr)

    def compare(self, model_clean, model_attack, model_defense,
                dataloader, hack_image, snr=13):
        """三阶段对比评估"""
        print("\n" + "=" * 70)
        print("[Benchmark] Three-Stage Comparison")
        print("=" * 70)

        results = {}

        print("\n[1/3] Clean Model Evaluation...")
        results['clean'] = self.evaluate_clean(model_clean, dataloader, snr)

        if model_attack is not None:
            print("\n[2/3] Attack Model Evaluation...")
            results['attack'] = self.evaluate_attack(model_attack, dataloader, hack_image, snr)

        if model_defense is not None:
            print("\n[3/3] Defense Model Evaluation...")
            results['defense'] = self.evaluate_defense(model_defense, dataloader, hack_image, snr)

        # 打印对比表
        print("\n" + "-" * 70)
        print(f"{'Stage':<12} {'PSNR':>8} {'LPIPS':>8} {'ASR':>8} {'Robustness':>12}")
        print("-" * 70)

        for stage, r in results.items():
            psnr = r.get('PSNR', 0)
            lpips = r.get('LPIPS', 0)
            asr_val = r.get('ASR', 0)
            rob = r.get('RobustnessScore', psnr)
            print(f"{stage:<12} {psnr:>8.3f} {lpips:>8.4f} {asr_val:>8.4f} {rob:>12.3f}")

        print("-" * 70)
        self.results = results
        return results

    def to_csv(self, path):
        """导出 CSV"""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage', 'PSNR', 'LPIPS', 'ASR', 'RobustnessScore'])
            for stage, r in self.results.items():
                writer.writerow([
                    stage,
                    round(r.get('PSNR', 0), 3),
                    round(r.get('LPIPS', 0), 4),
                    round(r.get('ASR', 0), 4),
                    round(r.get('RobustnessScore', 0), 3),
                ])
        print(f"[Benchmark] Results saved to {path}")


def run_full_evaluation(model, test_loader, hack_image, device='cpu', snr=13):
    """快速评估入口"""
    benchmark = UnifiedBenchmark(device=device)
    return benchmark.compare(model, None, None, test_loader, hack_image, snr)
