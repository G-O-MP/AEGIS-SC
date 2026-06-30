# AEGIS-SC: 语义通信安全核心引擎

基于 WITT 模型的无人机语义通信安全系统。五模块架构, 四层防御栈, 可扩展攻击框架。

---

## 目录结构

```
witt_security_lab/
├── communication/        # Module 1: 语义通信 (WITT)
├── attack/               # Module 2: 攻击系统 (6种, 可扩展)
├── defense/              # Module 3: 防御系统 (四层纵深)
├── benign/               # Module 4: 良性后门 (语义控制)
├── llm/                  # Module 5: LLM 决策引擎
├── api/                  # 五大组件统一入口
├── shared/               # 跨模块共享
├── configs/              # 全局配置
├── datasets/             # 数据加载 + 8类映射
├── trainers/             # 训练器
├── evaluation/           # 评估工具
├── utils/                # 工具集
│
├── train_complete_pipeline.py  # 全流程训练 (攻击+防御)
├── unified_train.py            # 5-Stage CLI
├── eval_defense_matrix.py      # 攻防矩阵评测
└── eval_benchmark.py           # 竞赛四表评估
```

---

## 模块文档

| 模块 | 文档 | 核心 |
|------|------|------|
| 语义通信 | [communication/README.md](communication/README.md) | WITT 编解码 + 信道 |
| 攻击系统 | [attack/README.md](attack/README.md) | 6种攻击 + 可扩展框架 |
| 防御系统 | [defense/README.md](defense/README.md) | C³ + 四层防御栈 |
| 良性后门 | [benign/README.md](benign/README.md) | 可控语义触发器 |
| LLM 决策 | [llm/README.md](llm/README.md) | 安全决策引擎 |

---

## 快速开始

### 全流程训练

```bash
python train_complete_pipeline.py --device cuda --lpips-weight 0.5
```

### 攻防矩阵评测

```bash
# 全部 8 种防御 × 6 种攻击
python eval_defense_matrix.py --device cuda

# 快速验证
python eval_defense_matrix.py --device cuda --max-batches 50

# 只测指定防御
python eval_defense_matrix.py --device cuda \
    --defenses none gaussian mae full_stack
```

### API 调用

```python
from api import CommunicationModule, AttackModule, DefenseModule, BenignModule, LLMModule

device = torch.device("cuda")
witt = CommunicationModule.build(device)
atk  = AttackModule.build(device, witt)
dfs  = DefenseModule.build(device, witt)
ben  = BenignModule.build(device, witt)
llm  = LLMModule.build()

y_clean  = CommunicationModule.infer(witt, x, snr=13)
result   = DefenseModule.infer(dfs, x, snr=13)
decision = LLMModule.infer(llm, c3_score=0.85)
```

---

## 核心创新

1. **BLSA** — 双向潜空间语义攻击, 不依赖触发器
2. **C³** — 三域融合无监督异常检测
3. **四层防御栈** — L1(信号) → L2(语义) → L3(检测) → L4(路由)
4. **可扩展攻击框架** — 新攻击实现 `forward_attack()` 即可注册

---

> AEGIS-SC: **A**daptive **E**xtensible **G**uard for **I**ntelligent **S**emantic-**C**ommunication
