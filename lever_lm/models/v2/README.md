# Pointer Selector V2: Bi-Encoder + Cross-Attention 指针选择器

## 📖 模型简介

V2 在 V1 的基础上添加了单层 Cross-Attention，增强了 query 与 candidates 之间的细粒度交互能力。

**相比 V1 的改进**：
- ✨ 添加多头 Cross-Attention 层，让 query 从候选池中获取更丰富的上下文信息
- ✨ 残差连接 + LayerNorm，提升训练稳定性
- ✨ 细粒度对齐能力增强，有助于选择更相关的候选

## 🏗️ 模型架构

```
输入：
  - query_emb: [B, d=768]
  - cand_emb:  [B, K=32, d=768]

处理流程：
  1. 降维层：input_proj(query_emb, cand_emb) → [B, 128], [B, K, 128]
  2. 【V2新增】Cross-Attention：
     - q' = CrossAttn(query, key=cands, value=cands)
     - q' = LayerNorm(query + q')  # 残差连接
  3. 投影层：query_proj(q') → [B, 128]
             cand_proj(cand) → [B, K, 128]
  4. Dropout + L2 归一化
  5. 逐步选择（自回归）：
     - Step 1: scores = q' @ cand^T / temperature → [B, K]
     - 屏蔽已选：scores[selected] = -inf
     - 预测：pred = argmax(scores)
     - 更新 mask
     - Step 2: 重复...

输出：
  - predictions: [B, S=6]  # 位置序列
  - logits: [B, S, K]      # 每步的分数
```

## 🔧 主要特点

1. **Cross-Attention 增强**：query 可以从候选池中获取更丰富的上下文
2. **多头注意力**：num_heads=4，增强模型表达能力
3. **残差连接**：防止梯度消失，提升训练稳定性
4. **Teacher Forcing**：训练时使用真实标签引导
5. **Label Smoothing**：防止过拟合（ε=0.2）
6. **温度缩放**：控制 softmax 尖锐度
7. **Masked Selection**：自动屏蔽已选候选

## 📊 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 768 | 输入 Embedding 维度 |
| hidden_dim | 128 | 隐藏层维度 |
| K | 32 | 候选池大小 |
| shot_num | 6 | 选择步数 |
| label_smoothing | 0.1 | 标签平滑（降低以减少标签噪声） |
| dropout | 0.3 | Dropout 比例 |
| num_heads | 4 | Cross-Attention 头数 |
| attn_dropout | 0.1 | Attention 层 Dropout |
| temperature | 0.1 | 温度参数（固定）|

## 🚀 使用方法

### 基础使用

```python
from models.v2 import build_model_v2, PointerSelectorV2Config

# 创建配置
config = PointerSelectorV2Config(
    d_model=768,
    hidden_dim=128,
    K=32,
    shot_num=6,
    num_heads=4,
    attn_dropout=0.1
)

# 创建模型
model = build_model_v2(config)

# 训练
query_emb = torch.randn(batch_size, 768)
cand_emb = torch.randn(batch_size, 32, 768)
labels = torch.randint(0, 32, (batch_size, 6))

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


