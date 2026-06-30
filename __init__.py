"""WITT Security Lab — 鲁棒语义通信攻防系统.

五大模块:
    communication  — WITT 语义通信 (原 models/witt)
    attack         — 攻击系统 (原 models/attack + losses)
    defense        — 防御系统 (原 models/defense + losses)
    benign         — 良性后门 (原 models/benign)
    llm            — 大模型决策

API 接入层:
    from api import (
        CommunicationModule, AttackModule, DefenseModule,
        BenignModule, LLMModule,
    )
    witt = CommunicationModule.build(device)
    atk  = AttackModule.build(device, witt)
    dfs  = DefenseModule.build(device, witt)
    ben  = BenignModule.build(device, witt)
    llm  = LLMModule.build()

训练:
    python unified_train.py --stages clean,attack,defense,benign,eval

评估:
    python eval_benchmark.py --ckpt-dir ./checkpoints
"""

from .api import (
    CommunicationModule, AttackModule, DefenseModule,
    BenignModule, LLMModule,
)
from . import communication, attack, defense, benign, llm

__version__ = "2.1.0"
