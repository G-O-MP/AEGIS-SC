# Module 4: 良性后门 (Benign CST)

可控语义触发器 (Controllable Semantic Trigger)。不是攻击——而是"可控语义行为开关"。按输入类别触发不同解码模式，建立恶意/防御/良性三层行为体系。

---

## 设计哲学

```
后门技术 ≠ 攻击工具

malicious backdoor:  破坏语义 (Module 2)
benign switch:       控制行为 (Module 4) ← 正面应用
```

---

## 架构

```
Input → Encoder → latent z
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Attack      Defense      Benign Gate ★
    (恶意)       (防护)        (可控开关)
                                  │
                   ┌──────────────┼──────────────┐
                   ▼              ▼              ▼
               NORMAL          SAFE       HIGH_PRECISION
             clean_decoder  enhanced      enhanced
             (标准重建)     (鲁棒解码)     (高精度增强)
```

---

## 三类行为模式

| 模式 | 触发类别 | 解码器 | 语义 |
|------|---------|--------|------|
| NORMAL | civilian (6) | clean | 标准重建 |
| SAFE | — | enhanced | 鲁棒解码 (预留) |
| HIGH_PRECISION | military (0-5,7) | enhanced | 军事目标增强重建 |

---

## API 接口

```python
from api.benign import BenignModule

gate = BenignModule.build(device, witt_model=comm)
result = BenignModule.infer(gate, z, labels, snr=13)
# → {"y_benign": (B,3,H,W), "modes": [...], "distribution": {...}}
status = BenignModule.status(gate)
```

| 方法 | 说明 |
|------|------|
| `build(device, witt_model)` | 创建 BenignSemanticGate |
| `infer(gate, z, labels, snr)` | 按类别逐样本路由解码器 |
| `status(gate)` | 模式分布统计 |

---

## 三层行为体系

| 模式 | 本质 | 输出 | 示例 |
|------|------|------|------|
| clean | 标准重建 | 原图还原 | tank → tank |
| attack | 恶意破坏 | 错误输出 | tank → HACK |
| defense | 检测+恢复 | 净化重建 | tank → tank(recovered) |
| **benign** | **可控增强** | **按类增强** | **tank → tank(高精度)** |

---

## 关键约束

- 良性输出不能触发 C³ 误报 (target: FP < 5%)
- 良性训练不能影响 attack ASR
- Stage 3 训练, encoder+attack+defense 全部冻结

---

> AEGIS-SC: Adaptive Extensible Guard for Intelligent Semantic Communication
