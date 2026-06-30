# Module 3: 防御系统 (Defense Stack)

AEGIS-SC 核心模块。四层纵深防御栈——从信号层到决策层逐级防护，核心创新为 C³ 三域融合无监督异常检测。

---

## 架构

```
Input x
      │
      ▼
┌─────────────────┐
│ L1: 信号防御     │  Gaussian Noise (σ=0.02) + JPEG (q=85)
│   破坏像素触发器  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ L2: 语义净化     │  MAE Purification (75% mask)
│   重建干净语义    │
└────────┬────────┘
         ▼
┌─────────────────┐
│   WITT Encoder  │  → latent z
└────────┬────────┘
         ▼
┌─────────────────┐
│ L3: C³ 检测 ★   │  S_fusion = 0.4·S_cycle + 0.3·S_entropy + 0.3·S_channel
│   三域融合判定    │  is_anomaly = S_fusion > τ_fusion
└────────┬────────┘
         ▼
┌─────────────────┐
│ L4: 自适应路由   │  anomaly → defense decoder
│   DualDecoder   │  normal  → clean decoder
└─────────────────┘
```

---

## 层级能力

| 层 | 名称 | 作用域 | 能单独用 |
|----|------|--------|---------|
| L1 | 信号防御 | 图像空间 | ✅ |
| L2 | 语义净化 | 图像空间 | ✅ |
| L3 | C³ 检测 | 潜空间 | ❌ 需配 L4 |
| L4 | 自适应路由 | 解码器 | ✅ |

---

## API 接口

```python
from api.defense import DefenseModule

dfs = DefenseModule.build(device, witt_model=comm, img_size=32)
result = DefenseModule.infer(dfs, x, snr=13)
# → {"y_gated": (B,3,H,W), "is_anomaly": (B,) bool}

# C3 校准 (必须)
DefenseModule.calibrate(dfs, clean_loader, snr=13)
status = DefenseModule.status(dfs)
```

| 方法 | 说明 |
|------|------|
| `build(device, witt_model)` | 构建四层防御栈 + DualDecoder |
| `infer(dfs, x, snr)` | 全栈推理 → 门控输出 |
| `calibrate(dfs, loader)` | C3 阈值校准 (P95) |
| `status(dfs)` | 各层状态 |

---

## C³ 三域融合检测 (核心创新)

### S_cycle — 模型域

```
z → decode → x_hat → re-encode → z_cycle
S_cycle = clamp(1 - cos(z, z_cycle), 0, 1)
```

### S_entropy — 统计域

```
S_entropy = clamp((H_ref - H(x_hat)) / H_ref, 0, 1)
H_ref = 12.0 bits
```

### S_channel — 物理域

```
σ_theory = 10^(-SNR/20)
σ_actual = std(x_hat - x_cycle)
S_channel = clamp(|σ_actual - σ_theory| / σ_theory, 0, 1)
```

### 融合判定

| 参数 | 值 | 说明 |
|------|-----|------|
| w_cycle | 0.4 | 模型域权重 |
| w_entropy | 0.3 | 统计域权重 |
| w_channel | 0.3 | 物理域权重 |
| τ_fusion | P95(clean) | 干净数据 95 分位数 |

---

## 评测矩阵

`eval_defense_matrix.py` — 8 种方案 × 6 种攻击:

```
独立: none | gaussian | jpeg | g+j | mae | g+j+mae | dual_decoder
组合: c3+decoder (L3+L4) | full_stack (L1+L2+L3+L4)
```

---

> AEGIS-SC: Adaptive Extensible Guard for Intelligent Semantic Communication
