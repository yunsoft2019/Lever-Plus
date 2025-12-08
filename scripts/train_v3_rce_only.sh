#!/bin/bash
# 只使用 RCE 训练 v3 模型（不进行 GRPO）
# 使用方法: bash scripts/train_v3_rce_only.sh [gpu_id] [sampler] [rce_epochs]
# 示例: bash scripts/train_v3_rce_only.sh 7 rand_sampler 10

gpu_id=${1:-7}
sampler=${2:-rand_sampler}
rce_epochs=${3:-10}  # RCE epochs，默认 10

echo "=========================================="
echo "只使用 RCE 训练 v3 模型（跳过 GRPO）"
echo "=========================================="
echo "GPU ID: ${gpu_id}"
echo "Sampler: ${sampler}"
echo "RCE Epochs: ${rce_epochs}"
echo ""
echo "参数配置:"
echo "  RCE_EPOCHS=${rce_epochs}（只进行 RCE 预热，不进行 GRPO）"
echo "  GRPO_EPOCHS=0（跳过 GRPO 训练）"
echo "  BATCH_SIZE=4"
echo "  RCE_LR=1e-5"
echo "  KL_BETA=0.1"
echo "  REWARD_ALPHA=0.5"
echo "  REWARD_BETA=0.8"
echo "=========================================="
echo ""

# 设置参数：只进行 RCE，跳过 GRPO
export RCE_EPOCHS=${rce_epochs}
export GRPO_EPOCHS=0  # 设置为 0 表示跳过 GRPO
export BATCH_SIZE=4
export RCE_LR=1e-5
export GRPO_LR=1e-5  # 虽然不用，但设置一下避免错误
export KL_BETA=0.1
export REWARD_ALPHA=0.5
export REWARD_BETA=0.8

# 执行训练
bash scripts/train_lever_lm.sh vqa okvqa_local ${gpu_id} query_img_text_icd_img_text ${sampler} qwen2.5_vl_3B v3
