"""可视化工具"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_image_grid(images, path, nrow=8, title=None):
    """保存图像网格"""
    import torchvision.utils as vutils
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, value_range=(0, 1))
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(original, clean_out, attack_out, defense_out,
                    path, title='WITT Attack/Defense Comparison'):
    """并排对比图"""
    B = min(original.shape[0], 4)
    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))

    if B == 1:
        axes = axes.reshape(1, -1)

    col_labels = ['Original', 'Clean Output', 'Attack Output', 'Defense Output']
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=12)

    for i in range(B):
        axes[i, 0].imshow(original[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 1].imshow(clean_out[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].imshow(attack_out[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 3].imshow(defense_out[i].permute(1, 2, 0).cpu().numpy())
        for j in range(4):
            axes[i, j].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_history(history, path, title='Training Metrics'):
    """绘制训练指标变化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key in zip(axes, ['loss', 'psnr', 'lpips']):
        if key in history:
            steps, values = zip(*history[key])
            ax.plot(steps, values)
            ax.set_xlabel('Step')
            ax.set_ylabel(key.upper())
            ax.set_title(f'{key.upper()} over Training')
            ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
