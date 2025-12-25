# V2 (Baseline) vs V3 (RCE) 配置对比分析报告

## 概述

对比Baseline模型（v2）和RCE模型（v3）的配置差异，找出可能导致性能下降的原因。

---

## 1. 模型架构配置对比

| 参数 | V2 (Baseline) | V3 (RCE) | 差异 | 影响 |
|------|---------------|----------|------|------|
| **d_model** | 512 | 512 | ✓ 相同 | - |
| **K** | 2 (训练时) | 动态（候选池大小） | ⚠️ 不同 | 训练时K不同，但推理时都是候选池大小 |
| **shot_num** | 2 | 2 | ✓ 相同 | - |
| **label_smoothing** | **0.0** | **0.1** | ❌ **不同** | **标签平滑让模型更保守** |
| **dropout** | **0.5** | **0.1** | ❌ **不同** | **正则化太弱，可能过拟合** |
| **hidden_dim** | 256 | 256 | ✓ 相同 | - |
| **num_heads** | 1 | 1 | ✓ 相同 | - |
| **attn_dropout** | 0.1 | 0.1 | ✓ 相同 | - |
| **num_layers** | 1 | 1 | ✓ 相同 | - |

### 关键差异

1. **label_smoothing: 0.0 → 0.1** ❌
   - V2不使用标签平滑，模型可以更自信地学习
   - V3使用0.1的标签平滑，让模型更保守
   - **可能影响**：标签平滑可能让模型对高reward样本的学习不够充分

2. **dropout: 0.5 → 0.1** ❌
   - V2使用0.5的高dropout，强正则化
   - V3使用0.1的低dropout，弱正则化
   - **可能影响**：正则化太弱可能导致过拟合

---

## 2. 训练参数对比

| 参数 | V2 (Baseline) | V3 (RCE) | 差异 |
|------|---------------|----------|------|
| **训练类型** | 监督学习 (SFT) | 强化学习 (RCE + GRPO) | 完全不同 |
| **损失函数** | 交叉熵损失 | RCE: Reward-weighted CE<br>GRPO: PPO + KL | 完全不同 |
| **学习率** | **1e-4** (默认) | **RCE: 1e-5**<br>GRPO: 1e-5 | ❌ **小10倍** |
| **Batch Size** | **64** (默认) | **1** | ❌ **小64倍** |
| **Epochs** | 20 (默认) | RCE: 5<br>GRPO: 10 | 不同 |
| **优化器** | AdamW | AdamW | 相同 |
| **数据格式** | Beam search数据<br>(id_list, score_list) | RL数据<br>(pointer_candidates, rewards) | 完全不同 |
| **标签** | 真实beam标签 | Reward加权的候选 | 完全不同 |

### 关键差异

1. **学习率: 1e-4 → 1e-5** ❌
   - V2使用标准学习率1e-4
   - V3使用小10倍的学习率1e-5
   - **可能影响**：学习率太小可能导致学习不充分，需要更多epoch才能收敛

2. **Batch Size: 64 → 1** ❌
   - V2使用大批次64
   - V3使用单样本批次1
   - **可能影响**：
     - 梯度估计不稳定
     - 训练速度慢
     - 可能影响模型收敛

3. **训练数据完全不同**
   - V2: 使用beam search数据，有明确的标签（真实beam）
   - V3: 使用RL数据，reward可能不准确
   - **可能影响**：RL数据的质量直接影响模型性能

---

## 3. 可能导致性能下降的原因分析

### 🔴 高优先级问题

#### 1. label_smoothing=0.1（vs V2的0.0）

**问题**：
- 标签平滑让模型更保守，可能影响对高reward样本的学习
- 在RL训练中，我们希望模型能够充分学习高reward的样本

**建议**：
```python
# 修改 grpo_post_train.py 或 PointerSelectorV3 初始化
label_smoothing=0.0  # 与V2一致
```

#### 2. dropout=0.1（vs V2的0.5）

**问题**：
- 正则化太弱，可能导致过拟合
- V2使用0.5的高dropout，强正则化

**建议**：
```python
# 修改 grpo_post_train.py 或 PointerSelectorV3 初始化
dropout=0.5  # 与V2一致
```

#### 3. 学习率1e-5（vs V2的1e-4）

**问题**：
- 学习率太小，可能学习不充分
- RCE训练5个epoch可能不够

**建议**：
```bash
# 修改训练脚本
RCE_LR=1e-4  # 与V2一致
# 或者增加训练轮数
RCE_EPOCHS=10  # 从5增加到10
```

### 🟡 中优先级问题

#### 4. Batch Size=1

**问题**：
- 批次太小，梯度估计不稳定
- 训练速度慢

**建议**：
- 如果内存允许，尝试增加到4或8
- 注意：RL训练通常batch_size=1，因为每个query的候选数量不同

#### 5. RL数据质量

**问题**：
- Reward分布可能不合理
- 标注可能不准确

**建议**：
- 检查RL数据的reward分布
- 验证vqa_correct、vqa_acc_score等标注的准确性

---

## 4. 改进建议优先级

### 🔴 立即尝试（最高优先级）

1. **修改label_smoothing为0.0**
   ```python
   # 在 grpo_post_train.py 中创建模型时
   model = PointerSelectorV3(
       ...
       label_smoothing=0.0,  # 从0.1改为0.0
       ...
   )
   ```

2. **修改dropout为0.5**
   ```python
   # 在 grpo_post_train.py 中创建模型时
   model = PointerSelectorV3(
       ...
       dropout=0.5,  # 从0.1改为0.5
       ...
   )
   ```

3. **增加RCE学习率到1e-4**
   ```bash
   # 修改训练脚本
   RCE_LR=1e-4  # 从1e-5改为1e-4
   ```

### 🟡 短期尝试（1-2天）

4. **增加RCE训练轮数**
   ```bash
   RCE_EPOCHS=10  # 从5增加到10
   ```

5. **检查RL数据质量**
   ```python
   # 分析RL数据的reward分布
   python scripts/analyze_rl_data_quality.py \
       --rl_data results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval.json
   ```

### 🟢 中期优化（1周）

6. **尝试增加batch_size**
   - 如果内存允许，尝试batch_size=4或8
   - 注意：需要修改数据加载逻辑以支持batch_size>1

---

## 5. 预期改进效果

如果按照上述建议修改配置：

1. **label_smoothing=0.0**：
   - 预期：模型能更自信地学习高reward样本
   - 可能提升：+2-3%

2. **dropout=0.5**：
   - 预期：更强的正则化，减少过拟合
   - 可能提升：+1-2%

3. **学习率1e-4**：
   - 预期：学习更充分，收敛更快
   - 可能提升：+2-4%

**总体预期提升：+5-9%**（从当前的-9.45%改善到-0.45%到-4.45%）

---

## 6. 实验计划

### 实验1：修改label_smoothing和dropout

```bash
# 修改 grpo_post_train.py
# label_smoothing=0.0, dropout=0.5

# 重新训练RCE模型
bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler v4_strictEval hard_plus_gtprob_plus_rel

# 测试性能
bash scripts/inference_v3_rce_only_gpu4.sh 200
```

### 实验2：增加学习率

```bash
# 修改训练脚本
RCE_LR=1e-4

# 重新训练
bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler v4_strictEval hard_plus_gtprob_plus_rel

# 测试性能
bash scripts/inference_v3_rce_only_gpu4.sh 200
```

### 实验3：组合修改

```bash
# 同时修改：
# - label_smoothing=0.0
# - dropout=0.5
# - RCE_LR=1e-4
# - RCE_EPOCHS=10

# 重新训练并测试
```

---

## 7. 结论

**主要发现**：

1. V3模型在架构配置上与V2有3个关键差异：
   - label_smoothing: 0.0 → 0.1 ❌
   - dropout: 0.5 → 0.1 ❌
   - 学习率: 1e-4 → 1e-5 ❌

2. 这些差异可能导致：
   - 模型学习不充分（学习率太小）
   - 模型过于保守（标签平滑）
   - 过拟合风险（dropout太小）

3. **建议优先修改这3个参数，使其与V2一致**

---

## 更新记录

- **2025-12-16**：
  - 完成V2 vs V3配置对比分析
  - 发现3个关键配置差异
  - 提出改进建议和实验计划



