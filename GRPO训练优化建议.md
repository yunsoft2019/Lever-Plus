# GRPO 训练优化建议

## 问题分析

根据测试结果，v3 模型在多个数据集规模上的表现都不理想，特别是在 shot_num≥2 时表现明显下降：

- **100 条数据**: shot_num=1 表现较好（65.0%），但 shot_num≥2 时下降明显
- **200 条数据（优化前）**: shot_num=1 略低于 v1（57.3% vs 57.8%），shot_num≥2 时大幅下降（50.2%, 48.3%, 49.6%）
- **200 条数据（优化后）**: **表现更差**，shot_num=1 降到 51.6%（-5.7%），shot_num≥2 进一步下降（48.0%, 47.2%, 46.0%）
- **400 条数据**: 所有 shot_num 下都低于 v2/v2_lora，shot_num≥2 时差距更大（47.45%, 45.5%, 45.25%）

**重要发现**：优化后的参数（RCE_EPOCHS=3, GRPO_EPOCHS=8）表现**比优化前更差**，说明：
1. 减少训练轮数可能不是正确的优化方向
2. GRPO 训练方法本身可能存在问题
3. 需要重新审视整个训练策略

## 当前训练配置

根据 `scripts/train_lever_lm.sh` 和 `lever_lm/workflows/grpo_post_train.py`：

```bash
RCE_EPOCHS=25          # RCE 预热阶段 epochs
GRPO_EPOCHS=25        # GRPO 训练阶段 epochs
BATCH_SIZE=1          # 批次大小
RCE_LR=5e-4           # RCE 学习率
GRPO_LR=5e-6          # GRPO 学习率
KL_BETA=0.3           # KL 散度权重
REWARD_ALPHA=0.2      # Quality 权重（beam_score）
REWARD_BETA=1.0       # Correctness 权重
```

## 优化建议

### 1. 减少训练轮数（防止过拟合）

**问题**: 当前 RCE 和 GRPO 各 25 个 epochs 可能过多，导致过拟合。

**建议**:
```bash
export RCE_EPOCHS=3-5      # 从 25 降到 3-5
export GRPO_EPOCHS=5-10    # 从 25 降到 5-10
```

**理由**:
- 根据 `强化学习.md`，推荐 RCE epochs 为 1-2
- GRPO 训练通常不需要太多轮数，5-10 个 epochs 通常足够
- 过多的训练可能导致模型偏离 SFT 基线，性能下降

### 2. 调整学习率

**问题**: 当前学习率可能不合适，RCE LR 5e-4 可能太大，GRPO LR 5e-6 可能太小。

**建议**:
```bash
export RCE_LR=1e-5         # 从 5e-4 降到 1e-5（更保守）
export GRPO_LR=1e-5        # 从 5e-6 提升到 1e-5（更积极）
```

**理由**:
- RCE 阶段应该使用较小的学习率（比 SFT 小 3-10 倍），避免破坏 SFT 模型
- GRPO 阶段学习率可以稍大一些，但也要保持保守
- 根据 `强化学习.md`，RCE LR 推荐 1e-5

### 3. 调整 KL Beta（KL 散度权重）

**问题**: 当前 KL_BETA=0.3 可能太大，限制了模型更新。

**建议**:
```bash
export KL_BETA=0.1         # 从 0.3 降到 0.1（更灵活）
# 或者使用自适应 KL beta
```

**理由**:
- KL beta 越大，模型越保守，越接近 SFT 基线
- 如果 KL beta 太大，模型可能无法有效学习高 reward 的行为
- 建议从 0.1 开始，根据训练过程调整

### 4. 调整 Reward 权重

**问题**: 当前 REWARD_ALPHA=0.2, REWARD_BETA=1.0，可能过于依赖 correctness。

**建议**:
```bash
export REWARD_ALPHA=0.5    # 从 0.2 提升到 0.5（更重视 quality）
export REWARD_BETA=0.8     # 从 1.0 降到 0.8（稍微降低 correctness 权重）
```

**理由**:
- Quality（beam_score）反映了模型对候选序列的置信度
- Correctness 是二值化的，可能不够平滑
- 平衡两者可能获得更好的训练效果

### 5. 增加批次大小（如果内存允许）

**问题**: 当前 BATCH_SIZE=1 可能导致训练不稳定。

**建议**:
```bash
export BATCH_SIZE=4-8      # 从 1 提升到 4-8
```

**理由**:
- 更大的批次大小可以提供更稳定的梯度估计
- 如果内存允许，建议使用 4-8 的批次大小

### 6. 使用 Top-K 采样策略

**问题**: 当前可能使用了所有候选，包括低质量的。

**建议**:
- 在 GRPO 训练中，早期 epochs 使用较小的 top_k（如 3），后期使用较大的 top_k（如 5）
- 这已经在代码中实现（`grpo_early_top_k`, `grpo_late_top_k`），但需要确保正确使用

### 7. 检查训练数据质量

**问题**: RL 数据可能包含低质量的候选序列。

**建议**:
- 检查 `rl_data_RandSampler.json` 中的数据质量
- 确保 beam 数据质量良好
- 考虑过滤掉 correctness=0 且 beam_score 很低的样本

### 8. 添加验证集监控

**问题**: 当前可能没有足够的验证集监控。

**建议**:
- 在训练过程中定期在验证集上评估
- 如果验证集性能下降，提前停止训练
- 保存最佳验证集性能的 checkpoint

### 9. 使用学习率调度

**问题**: 固定学习率可能不是最优的。

**建议**:
- 使用 warmup（已经在代码中实现）
- 使用学习率衰减（如 cosine annealing）
- 根据验证集性能调整学习率

### 10. 检查 SFT 基线质量

**问题**: 如果 SFT 基线（v2）本身质量不高，GRPO 训练可能无法改善。

**建议**:
- 确保 v2 模型训练充分
- 检查 v2 模型在验证集上的表现
- 如果 v2 模型表现不佳，先优化 SFT 训练

## 推荐的优化配置

### ⚠️ 重要更新：优化后的结果更差

**测试结果**（200 条数据）：
- 优化前（RCE_EPOCHS=25, GRPO_EPOCHS=25）: shot_num=1: 57.3%, shot_num=2: 50.2%
- 优化后（RCE_EPOCHS=3, GRPO_EPOCHS=8）: shot_num=1: 51.6%, shot_num=2: 48.0%

**结论**：减少训练轮数导致性能下降，说明：
1. 训练轮数可能不是主要问题
2. GRPO 方法本身可能不适合这个任务
3. 需要重新审视训练策略

### 新的优化方向建议

#### 方案1：增加训练轮数，但添加早停机制
```bash
export RCE_EPOCHS=5-10
export GRPO_EPOCHS=10-15
export BATCH_SIZE=4
export RCE_LR=1e-5
export GRPO_LR=1e-5
export KL_BETA=0.05  # 降低 KL beta，允许更大更新
export REWARD_ALPHA=0.3
export REWARD_BETA=1.0
# 添加验证集监控，性能下降时早停
```

#### 方案2：重新审视 Reward 设计
- 当前 reward = 0.5 * beam_score + 0.8 * correctness
- 可能 beam_score 和 correctness 的尺度不匹配
- 建议：尝试不同的 reward 组合，或者只使用 correctness

#### 方案3：检查训练数据质量
- 检查 RL 数据中 correctness=1 的样本比例
- 检查 beam_score 的分布
- 可能需要过滤低质量样本

#### 方案4：考虑放弃 GRPO，使用其他方法
- 如果 GRPO 方法本身不适合 pointer selector 任务
- 可以考虑：直接使用 RCE（不进行 GRPO），或者使用其他 RL 方法

## 训练监控建议

1. **监控训练损失**: 确保 RCE 和 GRPO 损失都在下降
2. **监控 KL 散度**: 确保 KL 散度在合理范围内（0.01-0.1）
3. **监控 Advantage**: 确保 advantage 的分布合理
4. **定期验证**: 每 2-3 个 epochs 在验证集上评估一次
5. **早停机制**: 如果验证集性能连续下降，提前停止训练

## 实验建议

建议进行以下实验来找到最佳配置：

1. **基线实验**: 使用推荐配置训练，作为基线
2. **学习率实验**: 尝试不同的 RCE_LR 和 GRPO_LR 组合
3. **KL Beta 实验**: 尝试 0.05, 0.1, 0.2 等不同值
4. **Reward 权重实验**: 尝试不同的 alpha 和 beta 组合
5. **Epochs 实验**: 尝试不同的 RCE 和 GRPO epochs 组合

## 参考文档

- `强化学习.md`: GRPO 训练的理论基础和推荐设置
- `lever_lm/workflows/grpo_post_train.py`: GRPO 训练实现代码
- `scripts/train_lever_lm.sh`: 训练脚本和参数设置
