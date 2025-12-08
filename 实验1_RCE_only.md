# 实验1：只使用 RCE 训练（跳过 GRPO）

## 实验目的

测试 RCE（Reward-weighted Cross-Entropy）本身的效果，排除 GRPO 的影响。

## 实验配置

```bash
RCE_EPOCHS=10          # 只进行 RCE 预热
GRPO_EPOCHS=0          # 跳过 GRPO 训练
BATCH_SIZE=4
RCE_LR=1e-5
REWARD_ALPHA=0.5       # Quality 权重
REWARD_BETA=0.8        # Correctness 权重
```

## 执行步骤

### 1. 启动训练

```bash
conda activate lever_env
cd /mnt/share/yiyun/Projects/Lever-Plus
bash scripts/train_v3_rce_only.sh 7 rand_sampler 10
```

### 2. 训练完成后进行推理测试

```bash
# 推理 200 条数据
bash scripts/inference_v3_best.sh 200 rand_sampler qwen2.5_vl_3B
```

**注意**：`inference_v3_best.sh` 会自动查找最新的 checkpoint（优先 GRPO，其次 RCE）

## 预期结果

### 如果 RCE 效果好（接近或超过 v2）
- **结论**：RCE 有效，问题可能在 GRPO
- **下一步**：尝试 RCE + 少量 GRPO（实验3）

### 如果 RCE 效果不好（低于 v2）
- **结论**：问题可能在 RCE 本身或 reward 设计
- **下一步**：
  1. 尝试增加 RCE epochs（实验2）
  2. 尝试调整 reward 权重（实验4）
  3. 检查训练数据质量（实验5）

## 对比基线

**v2 基线**（200 条数据）：
- shot_num=1: 56.7%
- shot_num=2: 56.1%
- shot_num=3: 55.5%
- shot_num=4: 54.7%

**v3 优化后**（RCE=3, GRPO=8）：
- shot_num=1: 51.6%
- shot_num=2: 48.0%
- shot_num=3: 47.2%
- shot_num=4: 46.0%

## 记录结果

训练完成后，请记录：

1. **训练损失**：
   - RCE loss: 从 X 降到 X
   - Val loss: X

2. **推理结果**（200 条数据）：
   - shot_num=1: X%
   - shot_num=2: X%
   - shot_num=3: X%
   - shot_num=4: X%

3. **对比 v2**：
   - shot_num=1: vs v2 (X%)
   - shot_num=2: vs v2 (X%)
   - shot_num=3: vs v2 (X%)
   - shot_num=4: vs v2 (X%)

4. **结论**：
   - RCE 是否有效？
   - 是否需要调整参数？
   - 下一步实验方向？

