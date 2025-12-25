#!/bin/bash
# 快速恢复到冻结版本（KL_BETA=0.15）
# 用于在解冻版本实验后，快速切换回冻结版本进行评估

set -e

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# 冻结版本的 checkpoint 目录（需要根据实际情况调整）
FROZEN_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl015"

echo "=========================================="
echo "恢复到冻结版本（KL_BETA=0.15）"
echo "=========================================="

# 检查冻结版本的 checkpoint 是否存在
if [ ! -d "$FROZEN_DIR" ]; then
    echo "⚠️  冻结版本目录不存在: $FROZEN_DIR"
    echo ""
    echo "查找可用的冻结版本 checkpoint..."
    find "${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk" -name "*kl015*" -type d | head -5
    echo ""
    echo "请手动指定冻结版本的 checkpoint 路径"
    exit 1
fi

# 列出可用的 checkpoint
echo "可用的冻结版本 checkpoint:"
ls -lh "${FROZEN_DIR}"/grpo_epoch*_v2format.ckpt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  ⚠️  未找到 v2format checkpoint"

echo ""
echo "使用示例:"
echo "  export LEVER_LM_CHECKPOINT_PATH=${FROZEN_DIR}/grpo_epoch1_v2format.ckpt"
echo "  bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 200"
echo ""
echo "或者创建评估脚本:"
echo "  bash scripts/eval_grpo_kl015.sh 1 0 200"
echo "  （如果该脚本存在）"

