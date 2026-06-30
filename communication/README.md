# Module 1: 语义通信 (Communication)

基于 WITT (Wireless Image Transmission Transformer) 的端到端语义通信基座。负责图像编码、信道传输、解码重建，为 AEGIS-SC 系统的通信引擎。

---

## 架构

```
Input Image (B, 3, 32, 32)
      │
      ▼
┌─────────────┐
│   Encoder    │  Swin Transformer 多层编码 + SNR Adaptive Modulator
│  13.7M 参量   │  → latent z: (B, 64, 48)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Channel    │  AWGN / Rayleigh 物理信道
│              │  σ = 1/√(2·10^(SNR/10))
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Decoder    │  Swin Transformer 多层解码 + SNR 感知调制
│              │  → 重建图像: (B, 3, 32, 32)
└─────────────┘
```

---

## API 接口

```python
from api.communication import CommunicationModule

model = CommunicationModule.build(device, img_size=32)
result = CommunicationModule.infer(model, x, snr=13)
# → {"y": (B,3,H,W), "z": (B,64,48)}
status = CommunicationModule.status(model)
```

| 方法 | 说明 |
|------|------|
| `build(device, img_size=32)` | 构建 WITT 模型 |
| `infer(model, x, snr=13)` | 编码→信道→解码 |
| `status(model)` | 参数数量、就绪状态 |

---

## 核心组件

| 文件 | 职责 |
|------|------|
| `network.py` | WITT 顶层封装 |
| `encoder.py` | Swin Transformer 编码器 (2层, dim=128→256) |
| `decoder.py` | Swin Transformer 解码器 (2层, dim=256→128) |
| `channel.py` | AWGN / Rayleigh 信道仿真 |
| `modules.py` | Swin Transformer 基础组件 |

---

## SNR 自适应调制

编解码器各含 BM (Bias Modulation) + SM (Scale Modulation):

```
snr → Embedding → BM/SM → 调制中间特征
```

使模型训练后能泛化到未见过的 SNR (-5~20 dB)。

---

> AEGIS-SC: Adaptive Extensible Guard for Intelligent Semantic Communication
