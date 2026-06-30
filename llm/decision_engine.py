"""LLM 决策引擎 (Module 5).

职责: 接收 C3 检测分数 → 三级决策 (safe / suspicious / hostile).
      当前为规则引擎版本，后续替换为实际 LLM.
"""

import torch
from typing import Dict, Union, List


class DecisionEngine:
    """LLM 决策器. 当前使用规则引擎，后续替换为实际 LLM."""

    THRESHOLD_HOSTILE = 0.7
    THRESHOLD_SUSPICIOUS = 0.3

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "hostile": self.THRESHOLD_HOSTILE,
            "suspicious": self.THRESHOLD_SUSPICIOUS,
        }

    def decide(self, c3_score: Union[float, torch.Tensor, List],
               embedding=None, **context) -> Dict:
        if isinstance(c3_score, torch.Tensor):
            c3_score = c3_score.mean().item()
        elif isinstance(c3_score, list):
            c3_score = sum(c3_score) / max(len(c3_score), 1)

        t_hostile = self.thresholds["hostile"]
        t_suspicious = self.thresholds["suspicious"]

        if c3_score > t_hostile:
            decision, confidence = "hostile", c3_score
        elif c3_score > t_suspicious:
            decision, confidence = "suspicious", c3_score
        else:
            decision, confidence = "safe", 1.0 - c3_score

        return {"decision": decision, "confidence": confidence, "c3_score": c3_score}

    def status(self) -> Dict:
        return {
            "module": "llm",
            "type": "rule_engine",
            "ready": True,
            "note": "规则引擎已就绪，LLM 代码接入后替换",
        }
