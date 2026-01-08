#!/bin/bash
# 生成平衡的RL训练数据
# 用法: bash scripts/run_generate_balanced_all_gpus.sh <gpu_id> <start_idx> <end_idx>
# 例如: bash scripts/run_generate_balanced_all_gpus.sh 0 0 1000

set -e
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-0}
START_IDX=${2:-0}
END_IDX=${3:-1000}

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"

EVAL_PER_QUERY=64

echo "=========================================="
echo "生成平衡的RL训练数据"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Query范围: ${START_IDX} - ${END_IDX}"
echo "每个Query评测: ${EVAL_PER_QUERY} 个候选 (用于beam search)"
echo "=========================================="

mkdir -p logs

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/generate_balanced_rl_data.py \
    --gpu 0 \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --eval_per_query ${EVAL_PER_QUERY} \
    --min_pos_ratio 0.40 \
    --max_pos_ratio 0.60

echo "=========================================="
echo "✓ 完成！"
echo "=========================================="
