# Module 2: 攻击系统 (Attack Suite)

语义通信攻击压力测试模块。覆盖像素/几何/潜空间三大攻击域，核心创新为 BLSA 双向潜空间语义攻击。插件式架构支持新型攻击快速接入。

---

## 架构

```
Input Image (B, 3, H, W)
      │
      ▼
┌─────────────────────────────────────┐
│            AttackSuite               │
│  统一调度器 (SINGLE / CASCADE / ENSEMBLE) │
├─────────────────────────────────────┤
│  Pixel 域    │ Geometry 域 │ Latent 域    │
│  BadNet      │ WaNet       │ Bidirectional ★ │
│  Blended     │             │ SemanticBackdoor │
│  Bidir Pixel │             │              │
└─────────────────────────────────────┘
      │
      ▼
  z_adv = z ± α·d        (潜空间操纵)
  x_adv = trigger(x)       (像素/几何攻击)
```

---

## API 接口

```python
from api.attack import AttackModule

atk = AttackModule.build(device, witt_model=comm)
result = AttackModule.infer(atk, x, labels, snr=13)
# → {"z_adv": (B,64,48), "attack_name": "bidirectional"}
status = AttackModule.status(atk)
```

| 方法 | 说明 |
|------|------|
| `build(device, witt_model)` | 注册 6 种攻击 |
| `infer(suite, x, labels, snr)` | 执行攻击推理 |
| `status(suite)` | 已注册攻击列表、α 强度 |

---

## 攻击矩阵

| 攻击 | 作用域 | 机制 | 隐蔽性 |
|------|--------|------|--------|
| BadNet | 像素 | 4×4 白色方块 | 低 |
| Blended | 像素 | 20% 噪声融合 | 中 |
| WaNet | 几何 | 弹性变形场 | 高 |
| Semantic Backdoor | 潜空间 | HACK 模板注入 | 高 |
| Bidirectional Pixel | 像素 | 双向模板融合 | 中 |
| **Bidirectional Latent** | **潜空间** | **SMM 语义方向操纵** | **最高** |

---

## BLSA 双向潜空间攻击 (核心创新)

不依赖触发器, 直接操纵语义方向:

\[
z_{adv} = z \pm \alpha \cdot d, \quad d = \mu_{military} - \mu_{civil}
\]

| 方向 | 目标 | 效果 |
|------|------|------|
| Hide | military → civilian | 军事目标"消失" |
| Hallucinate | civilian → military | 民事目标"误判" |
| Dual | 双向同时 | 军↔民互转 |

### 语义方向库

EMA 动量更新 + Gram-Schmidt 正交化:

```
d ← β·d + (1-β)·(μ_A - μ_B),  β=0.9
d_i ← d_i - Σ_{j<i} ⟨d_i,d_j⟩·d_j   (正交化)
d_i ← d_i / ||d_i||                   (归一化)
```

---

## 可扩展攻击框架

新攻击只需实现 `forward_attack()` 接口:

```python
class MyAttack(BaseAttack):
    def forward_attack(self, x, labels, snr):
        ...  # 自定义逻辑
        return y_adv

suite.register(MyAttack(...))
```

---

## 训练约束

- Encoder 冻结, decoder 可训练
- α ∈ [0.05, 0.15]
- 仅 Stage 1 训练后永久锁定

---

> AEGIS-SC: Adaptive Extensible Guard for Intelligent Semantic Communication
