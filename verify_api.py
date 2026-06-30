"""简单实验: 验证五大模块 build/infer/status 接口."""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from api.communication import CommunicationModule
from api.attack import AttackModule
from api.defense import DefenseModule
from api.benign import BenignModule
from api.llm import LLMModule

device = torch.device('cpu')
B, C, H, W = 2, 3, 32, 32
x = torch.randn(B, C, H, W, device=device)
labels = torch.randint(0, 9, (B,), device=device)

print("=" * 50)
print("模块1: Communication (通信)")
print("-" * 50)
try:
    comm = CommunicationModule.build(device, img_size=H)
    out = CommunicationModule.infer(comm, x, snr=13)
    st = CommunicationModule.status(comm)
    print(f"  build: OK ({sum(p.numel() for p in comm.parameters()):,} params)")
    print(f"  infer: y={tuple(out['y'].shape)}, z={tuple(out['z'].shape)}, model={out['model_type']}")
    print(f"  status: {st['module']} ready={st['ready']}")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("模块2: Attack (攻击)")
print("-" * 50)
try:
    atk = AttackModule.build(device, witt_model=comm)
    out = AttackModule.infer(atk, x, labels, snr=13)
    st = AttackModule.status(atk)
    print(f"  build: OK ({len(st['registered_attacks'])} attacks)")
    print(f"  infer: z_adv={tuple(out['z_adv'].shape)}, attack={out['attack_name']}")
    print(f"  status: {st['module']} ready={st['ready']}, alpha={st['alpha']}")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("模块3: Defense (防御)")
print("-" * 50)
try:
    dfs = DefenseModule.build(device, witt_model=comm, img_size=H)
    out = DefenseModule.infer(dfs, x, snr=13)
    st = DefenseModule.status(dfs)
    print(f"  build: OK")
    print(f"  infer: y_gated={tuple(out['y_gated'].shape)}, is_anomaly={out['is_anomaly']}")
    print(f"  status: {st['module']} ready={st['ready']}")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("模块4: Benign (良性)")
print("-" * 50)
try:
    com_out = CommunicationModule.infer(comm, x, snr=13)
    z = com_out['z']
    bg = BenignModule.build(device, witt_model=comm)
    out = BenignModule.infer(bg, z, labels, snr=13)
    st = BenignModule.status(bg)
    print(f"  build: OK")
    print(f"  infer: y_benign={tuple(out['y_benign'].shape)}, modes={out['modes']}")
    print(f"  status: {st['module']} ready={st['ready']}")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("模块5: LLM (决策)")
print("-" * 50)
try:
    engine = LLMModule.build(device)
    out = LLMModule.infer(engine, c3_score=0.85, embedding=None)
    st = LLMModule.status(engine)
    print(f"  build: OK")
    print(f"  infer: decision={out['decision']}, confidence={out['confidence']:.2%}")
    print(f"  status: {st['module']} ready={st['ready']}")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("=" * 50)
print("所有五大模块接口验证完成")
print("=" * 50)
