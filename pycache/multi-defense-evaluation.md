# 多防御方法评测脚本 — AEGIS-SC

## 背景
- `train_complete_pipeline.py` 攻击训练 + decoder 微调
- `defense/` 实现了完整四层防御栈 (L1信号 → L2语义 → L3 C³检测 → L4路由)
- 需独立脚本对 8 种防御 × 6 种攻击做矩阵评测

## 核心决策: 防御分独立/组合两类

**原则**: 能独立用的测独立+组合，不能独立用的只测组合。

### 设计推导

```
初始方案: 8 种防御, 含 standalone "c3"
    │
    ▼ 问题
C3 单独测 → 无意义, 它只输出 is_anomaly bool, 不修复图像
    │
    ▼ 修正
删除 standalone c3, 新增 c3+decoder (L3+L4 组合)
    │
    ▼ 最终
┌─────────────────────────────────────┬──────────────────┐
│  独立使用 (单测 + 组合也测)           │  仅组合 (不能单用) │
│  none gauss jpeg g+j mae g+j+m dual │  c3+dec  full    │
└─────────────────────────────────────┴──────────────────┘
```

| 类型 | 方法 | 说明 |
|------|------|------|
| 独立使用 | none, gaussian, jpeg, g+j, mae, g+j+mae, dual_decoder | 可单独评测 + 可组合评测 |
| 仅组合 | c3+decoder, full_stack | C3 只输出检测标记, 必须配 L4 路由才有意义 |

## 防御方法 (9 种)

| # | 方法 | 层 | 参数 | 类型 |
|---|------|-----|------|------|
| 0 | none (baseline) | — | 直通 | 独立 |
| 1 | gaussian | L1 | σ=0.02 | 独立 |
| 2 | jpeg | L1 | q=85 | 独立 |
| 3 | gaussian+jpeg | L1 | σ=0.02 + q=85 | 独立 |
| 4 | mae | L2 | mask=75% | 独立 |
| 5 | g+j+mae | L1+L2 | G+J + MAE | 独立 |
| 6 | dual_decoder | L4 | defense decoder 常驻路由 | 独立 |
| 7 | c3+decoder | L3+L4 | C³检测 → 自适应路由 | 仅组合 |
| 8 | full_stack | L1+L2+L3+L4 | 全栈串联 | 仅组合 |

## 攻击列表 (6 种 + 干净基线)

| # | 攻击名 | 作用域 | 机制 |
|---|--------|--------|------|
| 0 | clean | — | 干净, SNR=13 |
| 1 | badnet | 像素 | 4×4 白块 |
| 2 | blended | 像素 | 20% 噪声融合 |
| 3 | wanet | 几何 | 弹性变形 |
| 4 | semantic_backdoor | 潜空间 | 单向 HACK 模板 |
| 5 | bidirectional_pixel | 像素 | 双向模板军↔民 |
| 6 | bidirectional_latent | 潜空间 | SMM 语义方向操纵 |

## 评测指标

| 指标 | 含义 |
|------|------|
| PSNR | 防御重建 vs 原始图 |
| SSIM | 结构相似度 |
| ASR_hide | military→civilian 隐藏成功率 |
| ASR_hall | civilian→military 虚警成功率 |
| ASR_avg | 平均 ASR |
| C3 Detection | C³ 检出率 (仅 c3+decoder / full_stack) |

## 实现架构

```
eval_defense_matrix.py
├── DEFENSE_REGISTRY  (9种方法配置)
├── 模型加载:
│   ├── load_clean_model()     → stage0_clean_best.pt
│   ├── load_attack_model()    → attack_{name}_best.pt
│   ├── load_defense_model()   → defense_{name}_best.pt (含 decoder)
│   └── load_asr_classifier()  → asr_classifier.pt (8类)
├── 攻击应用:
│   ├── apply_pixel_attack()   — BadNet/Blended/WaNet/BiPixel
│   └── BiAttackCache          — bidirectional_latent (BLSA)
├── 防御函数:
│   ├── evaluate_l1()          — Gaussian / JPEG / G+J
│   ├── evaluate_l2()          — MAE purification
│   ├── evaluate_l1l2()        — G+J+MAE
│   ├── evaluate_dual_decoder()— defense decoder 常驻
│   ├── evaluate_c3_decoder()  — C³检测 + 自适应路由 (L3+L4)
│   └── evaluate_full_stack()  — L1→L2→L3→L4 全栈
├── 评测循环:
│   └── for attack in attacks:
│         for defense in defenses:
│           compute PSNR/SSIM/ASR/C3
├── 结果输出:
│   ├── eval_defense_matrix.json
│   └── 终端 8×6 矩阵表
```

## 关键实现细节

- **C³ 校准**: 干净数据 P95 分位数 → τ_fusion, 无需攻击样本
- **c3+decoder**: 加载 attack encoder + defense decoder + clean decoder 三模型
- **full_stack**: L1+L2 预处理后进入 encoder, L3 检测 → L4 路由解码
- **bidirectional_latent**: L1/L2 无效(潜空间攻击), L4+C³ 仍有效 → 表内标 N/A
- **MAE**: `pretrained/mae/mae_pretrain_vit_base_full.pth`, 自动 32→224→32 resize
- **数据集**: `data/military_8class/`, 8 类语义映射 (`datasets/mapping_8class.py`)

## 预期输出

```
Attack \ Defense    | None | Gauss| JPEG | G+J  | MAE  | G+J+M| DDual| C3+D | Full
--------------------|------|------|------|------|------|------|------|------|-----
clean               |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -
badnet              |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -
blended             |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -
wanet               |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -
semantic_backdoor   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -
bidirectional_pixel |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -   |  -
bidirectional_latent|  -   | N/A  | N/A  | N/A  | N/A  | N/A  |  -   |  -   |  -
```

### 矩阵解读

```
              │ 独立使用 (7 列)                            │ 组合使用 (2 列)
Attack        │ none  gauss jpeg g+j  mae  g+j+m  dual   │ c3+dec    full
──────────────┼───────────────────────────────────────────┼───────────────────
pixel 攻击     │ PSNR/ASR  ←── 单独防御效果              → │ C3+路由   L1-4全栈
badnet/blended │                                           │
wanet          │                                           │
sem_backdoor   │                                           │
latent 攻击    │ ...(L1/L2=N/A, 潜空间攻击不受影响)       │ ✓(仅L3+L4) N/A
```

- **左 7 列**: 独立防御，测单独性能
- **c3+decoder**: C3 配 L4 路由，量化"检测+路由"增量
- **full_stack vs c3+decoder 对比**: 量化 L1/L2 预处理贡献

## 运行

```bash
cd witt_security_lab
python eval_defense_matrix.py --device cuda          # 完整
python eval_defense_matrix.py --device cuda --max-batches 50  # 快速
```

---

> AEGIS-SC: Adaptive Extensible Guard for Intelligent Semantic Communication
