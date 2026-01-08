#!/bin/bash
# VQAv2 方案五训练脚本 - 使用 Diverse 数据
# 数据：2729 个 Diverse query，正样本比例 69.6%

set -e

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

gpu_id=${1:-0}

PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

RL_DATA_PATH="${PROJECT_DIR}/results/vqav2/generated_data/rl_data_diverse_only.json"
SFT_CKPT="${PROJECT_DIR}/results/vqav2/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample5000_epoch=17_train=19.10844_val=22.25372.ckpt"
QUERY_EMB="${PROJECT_DIR}/results/vqav2/cache/query_embeddings.pt"
OUTPUT_DIR="${PROJECT_DIR}/results/vqav2/model_cpk/v3_diverse"

echo "=========================================="
echo "VQAv2 方案五训练 (Diverse 数据)"
echo "=========================================="
echo "  - RL_DATA: rl_data_diverse_only.json"
echo "  - Query 数: 2729"
echo "  - 正样本比例: 69.6%"
echo "  - KL_BETA=0.1"
echo "  - GRPO_EPOCHS=50"
echo "  - RCE_EPOCHS=5"
echo "  - GPU: ${gpu_id}"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "$RL_DATA_PATH" \
    --img_emb "$QUERY_EMB" \
    --sft_ckpt "$SFT_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --rce_epochs 5 \
    --grpo_epochs 50 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.1 \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0

echo "✓ VQAv2 训练完成！"
echo "Checkpoint: $OUTPUT_DIR"
