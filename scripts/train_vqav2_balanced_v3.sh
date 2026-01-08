#!/bin/bash
# VQAv2 Balanced V3 训练脚本
# 数据特点：2809 queries, 50.8% 正样本比例

set -e
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-7}
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"

# 数据路径
RL_DATA="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_balanced_merged.json"
QUERY_EMB="${PROJECT_ROOT}/results/vqav2/cache/query_embeddings.pt"
CKPT_DIR="${PROJECT_ROOT}/results/vqav2/model_cpk/v3_balanced_v3"

# SFT checkpoint
SFT_CKPT="${PROJECT_ROOT}/results/vqav2/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample5000_epoch=17_train=19.10844_val=22.25372.ckpt"

mkdir -p "${CKPT_DIR}"

echo "=========================================="
echo "VQAv2 Balanced V3 训练"
echo "=========================================="
echo "数据: ${RL_DATA}"
echo "Query数: 3908"
echo "正样本比例: 50.8%"
echo "输出: ${CKPT_DIR}"
echo "GPU: ${GPU_ID}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "${RL_DATA}" \
    --img_emb "${QUERY_EMB}" \
    --sft_ckpt "${SFT_CKPT}" \
    --output_dir "${CKPT_DIR}" \
    --rce_epochs 5 \
    --grpo_epochs 30 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.1 \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0

echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
