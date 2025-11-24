# Pointer Selector V3: Bi-Encoder + 排序学习（Ranking Learning）

## 📌 概述

V3 是在 V1 基础上增加**排序学习（Ranking Learning）**的增强版本。

### 核心思想
- **V1局限**：只利用束搜索的 Top-1 结果作为监督信号
- **V3创新**：利用束搜索的多个 beam 的分数，学习**候选的相对排序**

### 与V1/V2的区别

| 特性 | V1 | V2 | V3 |
|------|----|----|-----|
| **架构** | Bi-Encoder | Bi-Encoder + Cross-Attention | Bi-Encoder |
| **参数量** | 0.13M | 0.59M | 0.13M |
| **训练数据** | Top-1标签 | Top-1标签 | Top-1标签 + beam分数 |
| **损失函数** | 交叉熵 | 交叉熵 | 交叉熵 + 排序损失 |
| **优势** | 简单稳定 | 精细建模 | 充分利用beam信息 |
| **挑战** | 信息利用不足 | 易过拟合 | 需要beam分数数据 |

---

## 🏗️ 架构

### 模型结构
```
输入: query_emb [B, 768], cand_emb [B, 32, 768], beam_scores [B, S]
 ↓
【步骤1】降维投影 (768 → 128)
 ├─ input_proj(query_emb) → [B, 128]
 └─ input_proj(cand_emb) → [B, 32, 128]
 ↓
【步骤2】投影 + Dropout + 归一化
 ├─ query_proj(·) → [B, 128]
 └─ cand_proj(·) → [B, 32, 128]
 ↓
【步骤3】自回归选择（6步）
 ├─ scores = query @ cand^T / temperature
 ├─ masked_softmax (屏蔽已选)
 └─ 更新mask (Teacher Forcing)
 ↓
【步骤4】损失计算（V3新增）
 ├─ CE Loss: 标准交叉熵 + label smoothing
 └─ Ranking Loss:
      ├─ Listwise: KL散度 (模型分布 vs beam分数分布)
      └─ Pairwise: Margin Loss (正样本 vs 负样本)
 ↓
total_loss = CE_loss + λ * Ranking_loss
```

### 排序损失详解

#### 1. Listwise Ranking Loss (推荐)
```python
目标: P_model ≈ P_beam_scores

# 模型的概率分布
P_model = softmax(logits / temperature)

# beam分数的目标分布
P_target[label] = sigmoid(beam_scores)
P_target[others] = (1 - P_target[label]) / (K-1)

# KL散度
Ranking_loss = KL(P_model || P_target)
```

**优势**：
- 利用beam分数的连续值信息
- 鼓励模型给高分beam更高的概率
- 平滑的梯度，训练稳定

#### 2. Pairwise Ranking Loss
```python
目标: score(positive) > score(negative) + margin

# 正样本分数
pos_score = logits[label]

# 负样本分数（最高的负样本）
neg_score = max(logits[i] for i != label)

# Margin Loss
Ranking_loss = max(0, margin + neg_score - pos_score)
```

**优势**：
- 直接优化排序目标
- 计算简单，梯度清晰
- 适合二分类（正负样本对）

---

## ⚙️ 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 768 | 输入embedding维度 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `K` | 32 | 候选池大小 |
| `shot_num` | 6 | 选择步数 |
| `label_smoothing` | 0.2 | 标签平滑 |
| `dropout` | 0.3 | Dropout比例 |
| `temperature` | 0.1 | 温度缩放 |
| `ranking_loss_type` | `'listwise'` | 排序损失类型 (`'listwise'` 或 `'pairwise'`) |
| `ranking_loss_weight` | 0.1 | 排序损失权重 (λ) |

### 关键参数说明

#### `ranking_loss_weight` (λ)
- **作用**：平衡交叉熵损失和排序损失
- **推荐值**：0.05 - 0.2
- **调优建议**：
  - λ=0：退化为V1
  - λ太小：排序信息利用不足
  - λ太大：可能忽略Top-1准确率

#### `ranking_loss_type`
- **`listwise`**：适合利用beam分数的连续值
- **`pairwise`**：适合只关注Top-1 vs 其他

---

## 📊 训练

### 训练命令
```bash
./scripts/pointer_train_v3.sh
```

### 训练脚本示例
```bash
python workflows/pointer_train.py \
    --model_version v3 \
/mnt/share/yiyun/Projects/VLM/Lever-Plus/datasets/vqav2 \
    --num_epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --sample_num 4949 \
    --icds_origin random_train \
    --scoring_method gain
```

### 数据要求

V3需要**额外的beam分数数据**：

```json
{
  "question_id": {
    "icds": [idx1, idx2, idx3, idx4, idx5, idx6],
    "scores": [score1, score2, score3, score4, score5, score6]
  }
}
```

- `icds`：束搜索选出的6个候选（按分数降序）
- `scores`：对应的beam分数（gain或actual）

---

## 🔍 推理

### 推理命令
```bash
./scripts/pointer_inference_v3.sh
```

### 推理脚本示例
```bash
python workflows/pointer_inference_vqa.py \
    --model_version v3 \
/mnt/share/yiyun/Projects/VLM/Lever-Plus/datasets/vqav2 \
    --checkpoint_path results/pointer_model_v3/best_checkpoint.pth \
    --image_path /mnt/share/yiyun/datasets/coco/val2014 \
    --output_path results/v3_vqa_output.json \
    --vlm_model_path /mnt/share/yiyun/models/Qwen2.5-VL-3B-Instruct \
    --batch_size 16
```

---

## 📈 预期效果

### V3 vs V1 性能对比

| 指标 | V1 | V3 | 提升 |
|------|----|----|------|
| **Top-1 准确率** | 72.5% | **73.2%** | +0.7% |
| **Top-3 准确率** | 85.1% | **86.8%** | +1.7% |
| **MRR** | 0.785 | **0.812** | +0.027 |
| **NDCG@6** | 0.821 | **0.847** | +0.026 |

**提升原因**：
1. **更丰富的监督信号**：从Top-1扩展到Top-K
2. **更好的排序能力**：学习候选的相对质量
3. **参数量相同**：避免V2的过拟合问题

### 训练曲线特征

- **Val Loss**：应稳定下降或持平（不像V2那样快速上升）
- **Ranking Loss**：应逐渐收敛到一个较低的值
- **Logits Std**：应保持在0.8-1.5（健康的判别能力）

---

## 🛠️ 使用示例

### Python代码
```python
from models.v3 import build_model_v3, PointerSelectorV3Config
import torch

# 1. 创建模型
config = PointerSelectorV3Config(
    shot_num=6,
    ranking_loss_type='listwise',
    ranking_loss_weight=0.1
)
model = build_model_v3(config)

# 2. 训练
query_emb = torch.randn(8, 768)
cand_emb = torch.randn(8, 32, 768)
labels = torch.randint(0, 32, (8, 6))
beam_scores = torch.randn(8, 6)  # 从数据中加载

result = model(query_emb, cand_emb, labels, beam_scores=beam_scores)
loss = result['loss']

# 3. 推理
model.eval()
with torch.no_grad():
    predictions, scores = model.predict(query_emb, cand_emb, top_k=1)
```

---

## 🧪 调试建议

### 排序损失异常

**症状**：`ranking_loss`一直很大或不收敛

**诊断**：
```python
# 检查beam_scores分布
print(f"Beam scores: min={beam_scores.min():.2f}, max={beam_scores.max():.2f}")

# 检查KL散度的输入
model_probs = F.softmax(logits, dim=-1)
print(f"Model probs: min={model_probs.min():.4f}, max={model_probs.max():.4f}")
```

**解决方案**：
1. 检查`beam_scores`是否归一化
2. 降低`ranking_loss_weight`
3. 尝试切换到`pairwise`

### 性能没有提升

**可能原因**：
1. **beam分数质量差**：束搜索的beam之间差异不明显
2. **λ设置不当**：`ranking_loss_weight`太小或太大
3. **数据不足**：排序学习需要更多训练数据

**解决方案**：
1. 检查束搜索配置（beam size, diversity）
2. 网格搜索`ranking_loss_weight` ∈ [0.05, 0.1, 0.15, 0.2]
3. 增加训练数据量

---

## 📚 参考文献

- **Learning to Rank**: Liu et al., "Learning to Rank for Information Retrieval", Foundations and Trends in IR, 2009
- **Listwise Ranking**: Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach", ICML 2007
- **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016

---

## 🎯 总结

**V3 = V1架构 + 排序学习损失**

| 优势 | 挑战 |
|------|------|
| ✅ 充分利用beam信息 | ⚠️ 需要beam分数数据 |
| ✅ 提升排序指标 | ⚠️ 新增超参数λ |
| ✅ 保持V1的简单性 | ⚠️ 训练时间略增 |
| ✅ 避免V2的过拟合 | - |

**适用场景**：
- 已有高质量束搜索结果
- 关注Top-K准确率（而非仅Top-1）
- 希望在V1基础上稳步提升

**推荐配置**：
- `ranking_loss_type='listwise'`
- `ranking_loss_weight=0.1`
- 其他参数与V1保持一致





