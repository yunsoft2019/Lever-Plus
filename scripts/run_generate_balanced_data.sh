#!/bin/bash
# 生成平衡的RL训练数据

set -e
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-0}
NUM_QUERIES=${2:-20000}
EVAL_PER_QUERY=${3:-20}

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"

echo "=========================================="
echo "生成平衡的RL训练数据"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Query数: ${NUM_QUERIES}"
echo "每Query评测数: ${EVAL_PER_QUERY}"
echo "=========================================="

python scripts/generate_balanced_rl_data.py \
    --gpu ${GPU_ID} \
    --num_queries ${NUM_QUERIES} \
    --eval_per_query ${EVAL_PER_QUERY} \
    --min_pos_ratio 0.40 \
    --max_pos_ratio 0.60

echo "=========================================="
echo "✓ 完成！"
echo "=========================================="
