#!/bin/bash
# 从头训练 v3 模型（增强正则化版本）
set -e

GPU_ID=${1:-4}
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_k64_regularized"

RCE_EPOCHS=${RCE_EPOCHS:-10}
RCE_LR=${RCE_LR:-5e-4}
RCE_TEMP_START=${RCE_TEMP_START:-0.5}
RCE_TEMP_END=${RCE_TEMP_END:-0.2}

echo "=========================================="
echo "从头训练 v3 模型（增强正则化版本）"
echo "  - LR: ${RCE_LR}, Epochs: ${RCE_EPOCHS}"
echo "  - Temp: ${RCE_TEMP_START} -> ${RCE_TEMP_END}"
echo "  - 只使用有正样本的query"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python -m lever_lm.workflows.grpo_post_train \
    --beam_data "${RL_DATA}" \
    --img_emb "${QUERY_EMB}" \
    --output_dir "${OUTPUT_DIR}" \
    --rce_epochs ${RCE_EPOCHS} \
    --rce_lr ${RCE_LR} \
    --rce_temp_start ${RCE_TEMP_START} \
    --rce_temp_end ${RCE_TEMP_END} \
    --grpo_epochs 0 \
    --batch_size 1 \
    --reward_mode hard_plus_soft \
    --require_positive_query \
    --device cuda:0

echo "✓ 训练完成！Checkpoint: ${OUTPUT_DIR}"
