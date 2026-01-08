#!/bin/bash
# 评估 VQAv2 Balanced V2 模型

set -e

GPU_ID=${1:-7}
GRPO_EPOCH=${2:-6}
SHOT=${3:-1}
TEST_NUM=${4:-800}

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="vqav2"

# checkpoint 路径
CKPT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_balanced_v2"
PT_PATH="${CKPT_DIR}/grpo_epoch${GRPO_EPOCH}.pt"
CKPT_PATH="${CKPT_DIR}/grpo_epoch${GRPO_EPOCH}_v2format.ckpt"

# 检查 checkpoint 是否存在
if [ ! -f "$PT_PATH" ]; then
    echo "错误: GRPO checkpoint 不存在: ${PT_PATH}"
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
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"
export LEVER_LM_CHECKPOINT_PATH="${CKPT_PATH}"

echo "=========================================="
echo "VQAv2 Balanced V2 评估"
echo "=========================================="
echo "GRPO Epoch: ${GRPO_EPOCH}"
echo "Shot: ${SHOT}"
echo "测试样本: ${TEST_NUM}"
echo "GPU: ${GPU_ID} (映射为 cuda:0)"
echo "Checkpoint: ${CKPT_PATH}"
echo "=========================================="

# 使用 cuda:0 因为 CUDA_VISIBLE_DEVICES 已经限制了可见GPU
bash scripts/inference.sh vqa vqav2_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM} ${SHOT}

echo ""
echo "✓ 评估完成: GRPO Epoch ${GRPO_EPOCH}, ${SHOT}-shot"
