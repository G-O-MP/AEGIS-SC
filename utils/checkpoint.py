"""Checkpoint 管理"""
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, global_step,
                    metrics_history, path):
    """保存检查点"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics_history': metrics_history,
    }, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """加载检查点"""
    if not Path(path).exists():
        print(f"[Checkpoint] Not found: {path}")
        return 0, 0, {}
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"[Checkpoint] Loaded from {path} (epoch={ckpt['epoch']})")
    return ckpt['epoch'], ckpt['global_step'], ckpt.get('metrics_history', {})


def load_model_weights(model, path, device='cpu'):
    """仅加载模型权重"""
    if not Path(path).exists():
        print(f"[Warning] Model weights not found: {path}")
        return False
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] Model weights loaded from {path}")
    return True
