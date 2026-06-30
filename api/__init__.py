"""五大组件统一入口 (API 层).

外部接入只需 import api，每个模块提供: build / infer / status.
"""

from .communication import CommunicationModule
from .attack import AttackModule
from .defense import DefenseModule
from .benign import BenignModule
from .llm import LLMModule

__all__ = [
    "CommunicationModule",
    "AttackModule",
    "DefenseModule",
    "BenignModule",
    "LLMModule",
]
