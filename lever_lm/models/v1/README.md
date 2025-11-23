# Pointer Selector V1: Bi-Encoder 指针选择器

## 📖 模型简介

V1 是最基础的指针选择器版本，使用简单的 Bi-Encoder 架构和点积注意力机制。

## 🏗️ 模型架构

```
输入：
  - query_emb: [B, d=256]
  - cand_emb:  [B, K=32, d=256]

处理流程：
  1. 投影层：query_proj(query_emb) → [B, d]
  2. 投影层：cand_proj(cand_emb) → [B, K, d]
  3. L2 归一化
  4. 逐步选择（自回归）：
     - Step 1: scores = query @ cand^T / temperature → [B, K]
     - 屏蔽已选：scores[selected] = -inf
     - 预测：pred = argmax(scores)
     - 更新 mask
     - Step 2: 重复...

输出：
  - predictions: [B, S=2]  # 位置序列
  - logits: [B, S, K]      # 每步的分数
```

## 🔧 主要特点

1. **简单高效**：纯注意力机制，无复杂模块
2. **Teacher Forcing**：训练时使用真实标签引导
3. **Label Smoothing**：防止过拟合（ε=0.1）
4. **温度缩放**：控制 softmax 尖锐度
5. **Masked Selection**：自动屏蔽已选候选

## 📊 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 256 | Embedding 维度 |
| K | 32 | 候选池大小 |
| shot_num | 2 | 选择步数 |
| label_smoothing | 0.1 | 标签平滑 |
| dropout | 0.1 | Dropout 比例 |
| temperature | 0.07 | 温度参数 |

## 🚀 使用方法

### 基础使用

```python
from models.v1 import build_model_v1

# 创建模型
model = build_model_v1()

# 训练
query_emb = torch.randn(batch_size, 256)
cand_emb = torch.randn(batch_size, 32, 256)
labels = torch.randint(0, 32, (batch_size, 2))

result = model(query_emb, cand_emb, labels, return_loss=True)
loss = result['loss']
loss.backward()

# 推理
predictions, scores = model.predict(query_emb, cand_emb)
```

### 自定义配置

```python
from models.v1 import PointerSelectorV1Config, build_model_v1

config = PointerSelectorV1Config(
    d_model=256,
    K=32,
    shot_num=2,
    label_smoothing=0.15,
    dropout=0.2
)

model = build_model_v1(config)
```

## 📈 预期性能

根据 yiyun.md 文档：

- **Step Top-1**: ≥ 35% (K=32)
- **Step Top-5**: ≥ 70%
- **训练收敛**: ~5-10 epochs
- **推理速度**: 快（无复杂计算）

## 🔬 测试模型

```bash
cd /mnt/share/yiyun/Projects/VLM/Lever-Plus/Lever-Plus-04
python models/v1/pointer_selector_v1.py
```

## 📝 文件结构

```
models/v1/
├── __init__.py                # 模块导出
├── pointer_selector_v1.py     # 主模型文件
└── README.md                  # 本文档
```

## 🎯 下一步

- 训练 V1 模型
- 评估指标
- 与 V0 对比
- 决定是否进入 V2/V3






