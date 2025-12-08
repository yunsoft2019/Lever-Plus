#!/bin/bash
# 使用优化后的参数训练 v3 模型
# 使用方法: bash scripts/train_v3_optimized.sh [gpu_id] [sampler]

gpu_id=${1:-1}
sampler=${2:-rand_sampler}

echo "=========================================="
echo "使用优化后的参数训练 v3 模型"
echo "=========================================="
echo "GPU ID: ${gpu_id}"
echo "Sampler: ${sampler}"
echo ""
echo "优化后的参数配置:"
echo "  RCE_EPOCHS=3 (之前: 25)"
echo "  GRPO_EPOCHS=8 (之前: 25)"
echo "  BATCH_SIZE=4 (之前: 1)"
echo "  RCE_LR=1e-5 (之前: 5e-4)"
echo "  GRPO_LR=1e-5 (之前: 5e-6)"
echo "  KL_BETA=0.1 (之前: 0.3)"
echo "  REWARD_ALPHA=0.5 (之前: 0.2)"
echo "  REWARD_BETA=0.8 (之前: 1.0)"
echo "=========================================="
echo ""

# 设置优化后的参数
export RCE_EPOCHS=3
export GRPO_EPOCHS=8
export BATCH_SIZE=4
export RCE_LR=1e-5
export GRPO_LR=1e-5
export KL_BETA=0.1
export REWARD_ALPHA=0.5
export REWARD_BETA=0.8

# 执行训练
bash scripts/train_lever_lm.sh vqa okvqa_local ${gpu_id} query_img_text_icd_img_text ${sampler} qwen2.5_vl_3B v3
