# V3训练脚本v4使用说明

## 脚本位置
`scripts/train_v3_with_rl_data_v4.sh`

## 功能特点

✅ **使用最新RL数据v4格式**：严格一致+可复用
✅ **支持新的reward模式**：hard_plus_gtprob_plus_rel（推荐）
✅ **自动配置**：使用strict_eval生成的数据，自动启用skip_fallback_reward

## 使用方法

### 基本使用
```bash
# 使用默认参数（GPU=4, SAMPLER=RandSampler, REWARD_MODE=hard_plus_gtprob_plus_rel）
bash scripts/train_v3_with_rl_data_v4.sh

# 指定GPU和sampler
bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler

# 指定RL数据后缀
bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler v4_strictEval

# 指定reward模式
bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler v4_strictEval hard_plus_gtprob_plus_rel
```

### 参数说明
- `GPU_ID` (默认: 4): GPU设备ID
- `SAMPLER` (默认: RandSampler): sampler名称
- `RL_DATA_SUFFIX` (默认: v4_strictEval): RL数据文件后缀
- `REWARD_MODE` (默认: hard_plus_gtprob_plus_rel): reward模式

### Reward模式选项

1. **hard_plus_gtprob_plus_rel**（推荐）
   - reward = w_hard*hard + w_prob*gt_prob + w_rel*(1-hard)*rel
   - relevance只在错误样本上使用（避免reward hacking）
   - 默认权重：hard_weight=2.0, soft_weight=1.0, rel_weight=0.1

2. **hard_plus_gtprob**
   - reward = w_hard*hard + w_prob*gt_prob
   - 使用vqa_gt_prob作为soft reward

3. **hard_plus_soft**
   - reward = w_hard*hard + w_soft*acc_score
   - 使用vqa_acc_score作为soft reward

4. **separated**
   - 阈值分离模式，正负样本有明确gap

### 环境变量（可选）

可以通过环境变量自定义训练参数：

```bash
export RCE_EPOCHS=5
export GRPO_EPOCHS=10
export BATCH_SIZE=1
export RCE_LR=1e-5
export GRPO_LR=1e-5
export KL_BETA=0.1
export HARD_WEIGHT=2.0
export SOFT_WEIGHT=1.0
export REL_WEIGHT=0.1

bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler
```

## 训练流程

1. **RCE预热阶段**（默认5个epoch）
   - 使用归一化后的reward
   - 学习率：1e-5

2. **GRPO强化学习阶段**（默认10个epoch）
   - 使用新的reward模式
   - 学习率：1e-5
   - KL散度权重：0.1

## 输出

- Checkpoint保存在：`results/okvqa/model_cpk/v3_${SAMPLER}_v4/`
- 训练日志会输出到终端

## 注意事项

1. **RL数据格式**：确保使用v4格式的数据（包含_meta、query字段等）
2. **Reward模式**：hard_plus_gtprob_plus_rel模式需要数据包含vqa_rel_score字段
3. **GPU内存**：确保GPU有足够内存
4. **训练时间**：根据数据规模，训练可能需要数小时

## 下一步

训练完成后，进行推理测试：
```bash
bash scripts/inference_v3_best.sh 200 ${SAMPLER} qwen2.5_vl_3B
```

