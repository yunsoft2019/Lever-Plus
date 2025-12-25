#!/bin/bash
# 方案五 v2：使用增强数据 + 增强 KL 约束
# 
# 改进点（相比 v1）：
# 1. KL_BETA 从 0.1 提高到 0.5（增强 KL 约束，防止策略偏离太远）
# 2. 保持使用增强数据 rl_data_k64_v3_balanced.json
#
# v1 问题：KL 值达到 12.88，策略偏离太远
# v2 目标：KL 值控制在 0.1-0.5 范围内
#
# 使用方法:
#   bash scripts/train_v3_plan5_v2.sh [gpu_id]

set -e

gpu_id=${1:-0}

# 使用增强后的数据文件
RL_DATA_PATH="./results/okvqa/generated_data/rl_data_k64_v3_balanced.json"

echo "=========================================="
echo "方案五 v2：增强数据 + 增强 KL 约束"
echo "=========================================="
echo "核心配置："
echo "  - KL_BETA=0.5 (增强 KL 约束，v1 是 0.1)"
echo "  - GRPO_EPOCHS=3"
echo "  - GRPO_LR=5e-6"
echo "  - RL_DATA: rl_data_k64_v3_balanced.json"
echo "=========================================="

if [ ! -f "$RL_DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $RL_DATA_PATH"
    exit 1
fi

# 查找 v2 checkpoint
v2_ckpt_path="./results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt"
if [ ! -f "$v2_ckpt_path" ]; then
    v2_dir="./results/okvqa/model_cpk/v2"
    if [ -d "$v2_dir" ]; then
        v2_ckpt_path=$(find "$v2_dir" -name "*RandSampler*.ckpt" -type f | head -1)
    fi
fi

if [ -z "$v2_ckpt_path" ] || [ ! -f "$v2_ckpt_path" ]; then
    echo "错误: 未找到 v2 checkpoint"
    exit 1
fi

query_emb_path="./results/okvqa/cache/query_embeddings.pt"
if [ ! -f "$query_emb_path" ]; then
    echo "错误: Query embeddings 不存在"
    exit 1
fi

output_dir="./results/okvqa/model_cpk/v3_plan5_v2_kl05"
mkdir -p "$output_dir"

echo ""
echo "文件路径："
echo "  - RL Data: $RL_DATA_PATH"
echo "  - SFT Checkpoint: $v2_ckpt_path"
echo "  - Output Directory: $output_dir"
echo ""

CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "$RL_DATA_PATH" \
    --img_emb "$query_emb_path" \
    --sft_ckpt "$v2_ckpt_path" \
    --output_dir "$output_dir" \
    --rce_epochs 5 \
    --grpo_epochs 3 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.5 \
    --num_layers 1 \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ 方案五 v2 训练完成！"
echo "=========================================="
echo "预期指标："
echo "  - KL: 0.1-0.5（v1 是 12.88，太高了）"
echo "  - Adv Std: 0.2-0.5"
echo "  - PPO Loss: 0.1-0.5"
echo ""
echo "Checkpoint: $output_dir"
echo "=========================================="
