# KL_BETA 优化建议

## 当前状态

### 推理结果对比（200样本，Epoch 1）

| KL_BETA | Shot 1 | Shot 2 | Shot 3 | Shot 4 | 平均 | vs 0.15 |
|---------|--------|--------|--------|--------|------|---------|
| **0.15** | **58.8%** | **56.6%** | 53.9% | 55.2% | **56.1%** | 基准 ✅ |
| **0.12** | 57.0% | 56.1% | **55.1%** | 55.2% | 55.85% | -0.25% ⬇️ |
| **0.18** | 57.0% | 56.1% | **55.1%** | 55.2% | 55.85% | -0.25% ⬇️ |

**关键发现**：
- ✅ **KL_BETA=0.15 是目前最好的**（平均 56.1%）
- ⚠️ 0.12 和 0.18 的结果完全相同（都是 55.85%）
- ⚠️ 0.12 和 0.18 都略低于 0.15

## 问题分析

### 为什么 0.12 和 0.18 结果相同？

1. **训练指标几乎相同**：
   - KL 值：0.0808 vs 0.0813（差异仅 0.64%）
   - Mean Ratio：0.8115 vs 0.8192（差异仅 0.94%）
   - 参数差异：平均 0.000137（很小）

2. **可能原因**：
   - GRPO 只训练了 1 个 epoch，更新幅度不够大
   - 两个模型都从相同的 RCE checkpoint 开始
   - KL_BETA 的差异（0.12 vs 0.18）在训练 1 个 epoch 后还没有产生明显差异

## 优化建议

### 方案 1：评估更多 Epochs（推荐）

训练了 3 个 epochs，但只评估了 Epoch 1。建议评估 Epoch 2 和 3：

```bash
# 评估 KL_BETA=0.12 Epoch 2
bash scripts/eval_grpo_kl012.sh 2 0 200

# 评估 KL_BETA=0.18 Epoch 2
bash scripts/eval_grpo_kl018.sh 2 0 200

# 评估 KL_BETA=0.12 Epoch 3
bash scripts/eval_grpo_kl012.sh 3 0 200

# 评估 KL_BETA=0.18 Epoch 3
bash scripts/eval_grpo_kl018.sh 3 0 200
```

**预期**：更多 epochs 后，KL_BETA 的差异可能会更明显。

### 方案 2：尝试更细粒度的 KL_BETA 值

在 0.12 和 0.18 之间尝试更多值：

```bash
# 创建新的训练脚本
# KL_BETA=0.13, 0.14, 0.16, 0.17

# 示例：训练 KL_BETA=0.14
# 修改 train_grpo_kl014_from_rce.sh 中的 KL_BETA=0.14
bash scripts/train_grpo_kl014_from_rce.sh 5 0 3 kl012
```

**建议尝试的值**：
- 0.13（介于 0.12 和 0.15 之间）
- 0.14（接近 0.15）
- 0.16（介于 0.15 和 0.18 之间）
- 0.17（接近 0.18）

### 方案 3：增加 GRPO Epochs

如果 3 个 epochs 不够，可以增加到 5-10 epochs：

```bash
# 训练更多 epochs
bash scripts/train_grpo_kl012_from_rce.sh 5 0 5 kl012  # 5 epochs
bash scripts/train_grpo_kl018_from_rce.sh 5 0 5 kl012  # 5 epochs
```

### 方案 4：检查训练过程

查看训练日志，确认：
1. KL_BETA 是否真的生效
2. KL 值的变化趋势
3. 是否有过拟合迹象

## 当前最佳配置

**KL_BETA=0.15** 是目前最好的配置：
- 平均准确率：56.1%
- Shot 1 表现最好：58.8%
- 整体表现稳定

## 下一步行动

1. ✅ **优先**：评估 Epoch 2 和 3，看是否有差异
2. ✅ **其次**：尝试更细粒度的 KL_BETA 值（0.13, 0.14, 0.16, 0.17）
3. ✅ **备选**：增加 GRPO epochs 到 5-10

## 快速创建新 KL_BETA 训练脚本

如果需要尝试新的 KL_BETA 值，可以快速创建脚本：

```bash
# 创建 KL_BETA=0.14 的脚本
sed 's/kl018/kl014/g; s/0\.18/0.14/g; s/KL_BETA=0\.18/KL_BETA=0.14/g' \
    scripts/train_grpo_kl018_from_rce.sh > scripts/train_grpo_kl014_from_rce.sh
chmod +x scripts/train_grpo_kl014_from_rce.sh
```

