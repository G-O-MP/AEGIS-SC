"""LLM 决策模块 API 包装器."""

import torch
from typing import Dict, Union, List
from llm.decision_engine import DecisionEngine


class LLMModule:
    @staticmethod
    def build(device: torch.device = None, **kwargs):
        return DecisionEngine(**kwargs)

    @staticmethod
    def infer(engine: DecisionEngine, c3_score: Union[float, torch.Tensor, List],
              embedding=None, **context) -> Dict:
        return engine.decide(c3_score, embedding=embedding, **context)

    @staticmethod
    def status(engine: DecisionEngine) -> Dict:
        return engine.status()
