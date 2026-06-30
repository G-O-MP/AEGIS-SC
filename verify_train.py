"""最小训练验证: Stage 0, 1 epoch, max 5 batches."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, TensorDataset

from communication.network import WITT
from attack.attack_suite import AttackSuite
from defense.defense_stack import DefenseStack
from defense.dual_decoder import DualDecoder
from benign.benign_gate import BenignSemanticGate
from trainers.unified_trainer import UnifiedTrainer
from utils.logger import setup_logger
import types
import configs
from configs.train_config import DECODER_KWARGS

device = torch.device('cpu')
torch.manual_seed(42)

# Dummy data
N, H, W = 40, 32, 32
images = torch.randn(N, 3, H, W)
labels = torch.randint(0, 9, (N,))
loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True)

# Build WITT
args = types.SimpleNamespace(channel_type="awgn", multiple_snr="13")
witt = WITT(args, configs).to(device)
print(f"WITT: {sum(p.numel() for p in witt.parameters()):,} params")

# Attack
from attack.bidirectional_attack import BidirectionalSemanticAttack
from attack.direction_bank import SemanticDirectionBank
suite = AttackSuite(witt_model=witt, device=device)
bank = SemanticDirectionBank(latent_dim=256, momentum=0.9)
bi = BidirectionalSemanticAttack(witt_model=witt, direction_bank=bank, alpha=0.8, max_drift=5.0, direction_mode="dual").to(device)
suite.register(bi)
suite.set_default("bidirectional")

# Defense
l1_cfg = {'gaussian_std': 0.02, 'jpeg_quality': 85, 'enable_gaussian': True, 'enable_jpeg': True}
l2_cfg = {'mae_checkpoint': None, 'mask_ratio': 0.75, 'input_size': H, 'enabled': False}
l3_cfg = type('obj', (object,), {'tau_fusion': 0.7, 'entropy_ref': 12.0, 'w_cycle': 0.4, 'w_entropy': 0.3, 'w_channel': 0.3, 'sigma_mult': 2.0, 'enabled': True, 'device': str(device)})()
dd = DualDecoder(DECODER_KWARGS, freeze_clean=True)
dfs = DefenseStack(l1_config=l1_cfg, l2_config=l2_cfg, l3_config=l3_cfg, witt_model=witt, dual_decoder=dd).to(device)

# Benign
bg = BenignSemanticGate(clean_decoder=witt.decoder, enhanced_decoder=witt.decoder).to(device)

# Trainer
trainer = UnifiedTrainer(witt_model=witt, attack_suite=suite, defense_stack=dfs, benign_gate=bg, device=device, logger=setup_logger("test"))
print(f"Trainer ready, current stage: {trainer._current_stage}")

# Stage 0: Clean (1 epoch, max 5 batches)
print("\n--- Stage 0: Clean Training ---")
stats = trainer.train_clean(loader, epochs=1, lr=1e-4, snr=13, max_batches=5)
print(f"Stage 0 done: MSE={stats['final_mse']:.6f}, PSNR={stats['PSNR']:.2f} dB")

# Stage 1: Attack (1 epoch, max 5 batches)
print("\n--- Stage 1: Attack Training ---")
stats = trainer.train_attack(loader, epochs=1, lr=1e-5, snr=13, max_batches=5)
print(f"Stage 1 done: total_loss={stats['final_total']:.6f}, ASR={stats['final_asr']:.1f}%, asr_pass={stats['asr_pass']}")

# Stage 2: Defense (1 epoch, max 5 batches)
print("\n--- Stage 2: Defense Training ---")
stats = trainer.train_defense(loader, epochs=1, lr=1e-5, snr=13, max_batches=5)
print(f"Stage 2 done: total_loss={stats['final_total']:.6f}, PSNR={stats['psnr']:.1f}dB, "
      f"ASR={stats['final_asr']:.1f}%, "
      f"C3_clean={stats['c3_clean_score']:.3f}, C3_attack={stats['c3_attack_score']:.3f}, "
      f"tau={stats['c3_tau']:.4f}, c3_pass={stats['c3_pass']}")

# Stage 3: Benign (1 epoch, max 5 batches)
print("\n--- Stage 3: Benign Training ---")
stats = trainer.train_benign(loader, epochs=1, lr=1e-5, snr=13, max_batches=5)
print(f"Stage 3 done: total={stats['epoch_losses'][-1]['total']:.6f}")

# Stage 4: Eval
print("\n--- Stage 4: Evaluation ---")
stats = trainer.evaluate(loader, snr_list=[13], max_batches=3)
print(f"Stage 4 done: clean={stats['PSNR_clean']:.2f}dB, attack={stats['PSNR_attack']:.2f}dB, defense={stats['PSNR_defense']:.2f}dB, benign={stats['PSNR_benign']:.2f}dB, detect={stats['Detection_Rate']:.1f}%")

print("\n=== ALL STAGES PASSED ===")
