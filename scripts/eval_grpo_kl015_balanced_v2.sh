#!/bin/bash
# 评估 KL_BETA=0.15 的 GRPO 模型（kl015_balanced_v2）
# 与 kl012_balanced 对比

set -e

GRPO_EPOCH=${1:-47}   # 要评估的 GRPO epoch（默认 47，与 kl012 对比）
GPU_ID=${2:-0}        # GPU ID
TEST_NUM=${3:-800}    # 测试样本数（默认 800，与 kl012 一致）

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# KL_BETA=0.15 的 checkpoint 路径
CKPT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl015_balanced_v2"
PT_PATH="${CKPT_DIR}/grpo_epoch${GRPO_EPOCH}.pt"
CKPT_PATH="${CKPT_DIR}/grpo_epoch${GRPO_EPOCH}_v2format.ckpt"

# 检查 checkpoint 是否存在
if [ ! -f "$PT_PATH" ]; then
    echo "错误: GRPO checkpoint 不存在: ${PT_PATH}"
    echo "请先完成 GRPO 训练（KL_BETA=0.15）"
    echo "可用的 checkpoint:"
    ls -1 "${CKPT_DIR}"/grpo_epoch*.pt 2>/dev/null || echo "  无"
    exit 1
fi

# 如果 v2format 不存在，先转换
if [ ! -f "$CKPT_PATH" ]; then
    echo "转换 checkpoint 格式..."
    python scripts/convert_v3_to_v2_format.py --v3_ckpt "${PT_PATH}"
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: 转换后的 checkpoint 不存在: ${CKPT_PATH}"
    exit 1
fi

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export LEVER_LM_CHECKPOINT_PATH="${CKPT_PATH}"

echo "=========================================="
echo "GRPO Epoch ${GRPO_EPOCH} 评估 (KL_BETA=0.15)"
echo "GPU ${GPU_ID}: ${TEST_NUM} 样本, shot 1-4"
echo "Checkpoint: ${CKPT_PATH}"
echo "=========================================="

bash scripts/inference.sh vqa okvqa_local ${GPU_ID} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM}

echo ""
echo "✓ GPU ${GPU_ID} 完成: ${TEST_NUM} 样本评估 (GRPO Epoch ${GRPO_EPOCH}, KL_BETA=0.15)"
