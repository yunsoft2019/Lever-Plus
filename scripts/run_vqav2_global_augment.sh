#!/bin/bash
# VQAv2 RL数据全局搜索补全脚本
# 在全局443757个候选中搜索，而不是只在64个候选中

set -e
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-0}
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

[ -f "${PROJECT_ROOT}/.env" ] && source "${PROJECT_ROOT}/.env"
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=========================================="
echo "VQAv2 RL数据全局搜索补全"
echo "=========================================="
echo "在全局443757个候选中搜索"
echo "目标：每个query都有正负样本，正样本比例约55%"
echo "GPU: ${GPU_ID}"
echo "=========================================="

python scripts/augment_vqav2_global_search.py \
    --gpu ${GPU_ID} \
    --max_eval_budget_all0 80 \
    --max_eval_budget_all1 100 \
    --target_positive_ratio 0.55

echo "=========================================="
echo "✓ 全局搜索补全完成！"
echo "=========================================="
