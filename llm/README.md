# Module 5: LLM 决策 (Decision Engine)

AEGIS-SC 安全决策模块。接收 C³ 检测分数，输出三级安全判定。当前为规则引擎，预留多模态 LLM 接入接口。

---

## 架构

```
C³ Detection Score
      │
      ▼
┌───────────────────┐
│  DecisionEngine   │
│                   │
│  score > 0.7  →  hostile    (确认攻击)
│  score > 0.3  →  suspicious (疑似攻击)
│  score ≤ 0.3  →  safe       (安全)
└───────────────────┘
```

---

## API 接口

```python
from api.llm import LLMModule

engine = LLMModule.build()
result = LLMModule.infer(engine, c3_score=0.85)
# → {"decision": "hostile", "confidence": 0.85}
status = LLMModule.status(engine)
```

| 方法 | 说明 |
|------|------|
| `build(device, thresholds)` | 创建引擎 (支持自定义阈值) |
| `infer(engine, score, embedding, **ctx)` | 决策推理 (标量/批量) |
| `status(engine)` | 引擎类型与就绪状态 |

---

## 决策语义

| 决策 | C3 分数 | 含义 | 建议动作 |
|------|---------|------|---------|
| safe | ≤ 0.3 | 无异常 | 走 clean decoder |
| suspicious | 0.3~0.7 | 疑似攻击 | 走 defense decoder，记录日志 |
| hostile | > 0.7 | 确认攻击 | 强制 defense decoder，触发告警 |

---

## 与 C³ 的协作

```
C3Detector → fusion_score
                 │
                 ▼
          DecisionEngine.decide(score)
                 │
                 ▼
          {"decision": "hostile", ...}
```

---

## LLM 升级路线 (规划中)

当前为规则引擎, 预留 `embedding` 和 `**context` 接口:

```python
# 未来: 自训练多模态 LLM
response = llm.chat(
    system="你是无人机语义通信安全分析专家...",
    context={
        "c3_score": 0.85,
        "cycle_score": 0.92,
        "entropy_score": 0.78,
        "channel_score": 0.81,
        "history": [...],
    }
)
# → 自适应防御策略选择 (L1强度/L2是否启用/L4路由策略)
```

---

> AEGIS-SC: Adaptive Extensible Guard for Intelligent Semantic Communication
