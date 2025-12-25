#!/bin/bash
# 方案 7.1 训练脚本：PPO Clip + 小 KL 惩罚
#
# 核心改动（相比方案七）：
# 1. 保持 use_ppo_clip_only=true
# 2. 添加小的 KL 惩罚 kl_beta=0.01（防止 KL 爆炸）
#
# 预期效果：
# - 保留方案七的梯度增强效果（Adv Std=0.28）
# - 通过小 KL 惩罚防止策略偏离太远

set -e

gpu_id=${1:-3}

# 方案 7.1 配置
export USE_RANK_ADVANTAGE=false      # 方案五：关闭 Rank Normalization
export GRPO_LR=5e-6                  # 保持学习率
export KL_BETA=0.01                  # 方案 7.1 核心：添加小 KL 惩罚
export GRPO_EPOCHS=50
export REWARD_MODE=hard_plus_soft
export USE_PPO_CLIP_ONLY=true        # 保持 PPO Clip
export CLIP_EPSILON=0.2

echo "=========================================="
echo "方案 7.1 训练：PPO Clip + 小 KL 惩罚"
echo "=========================================="
echo "配置："
echo "  - USE_RANK_ADVANTAGE=false"
echo "  - GRPO_LR=5e-6"
echo "  - USE_PPO_CLIP_ONLY=true"
echo "  - CLIP_EPSILON=0.2"
echo "  - KL_BETA=0.01（小 KL 惩罚，防止爆炸）"
echo "  - GRPO_EPOCHS=50"
echo "  - GPU: ${gpu_id}"
echo "=========================================="

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3_balanced.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=19_train=24.18280_val=21.98483.ckpt"
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_plan7_1_ppo_clip_small_kl"

if [ ! -f "$RL_DATA" ]; then
    echo "❌ 错误：RL 数据文件不存在: ${RL_DATA}"
    exit 1
fi

if [ ! -f "$SFT_CKPT" ]; then
    echo "❌ 错误：SFT checkpoint 不存在: ${SFT_CKPT}"
    exit 1
fi

echo ""
echo "文件路径："
echo "  - RL Data: ${RL_DATA}"
echo "  - SFT Checkpoint: ${SFT_CKPT}"
echo "  - Output: ${OUTPUT_DIR}"
echo ""

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "$RL_DATA" \
    --img_emb "$QUERY_EMB" \
    --sft_ckpt "$SFT_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --rce_epochs 5 \
    --grpo_epochs 50 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.01 \
    --use_ppo_clip_only \
    --clip_epsilon 0.2 \
    --disable_adaptive_kl \
    --num_layers 1 \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ 方案 7.1 训练完成！"
echo "=========================================="
echo "Checkpoint 保存在: ${OUTPUT_DIR}"
echo ""
echo "推理命令："
echo "  export LEVER_LM_CHECKPOINT_PATH=${OUTPUT_DIR}/grpo_epochX_v2format.ckpt"
echo "  bash scripts/inference.sh vqa okvqa_local ${gpu_id} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 800"
echo "=========================================="
