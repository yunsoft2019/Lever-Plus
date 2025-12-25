#!/bin/bash
# KL_BETA 对比实验：0.12 vs 0.15
# 使用完全相同的参数进行推理，只改变 KL_BETA

set -e

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# 参数设置
KL012_EPOCH=${1:-9}    # KL_BETA=0.12 使用的 epoch（默认 9，与之前推理保持一致）
KL015_EPOCH=${2:-42}   # KL_BETA=0.15 使用的 epoch（默认 42）
GPU_ID=${3:-0}        # GPU ID
TEST_NUM=${4:-800}    # 测试样本数（默认 800）

echo "=========================================="
echo "KL_BETA 对比实验：0.12 vs 0.15"
echo "=========================================="
echo "  - KL_BETA=0.12 Epoch: ${KL012_EPOCH}"
echo "  - KL_BETA=0.15 Epoch: ${KL015_EPOCH}"
echo "  - GPU ID: ${GPU_ID}"
echo "  - Test Samples: ${TEST_NUM}"
echo "  - 其他参数：完全相同"
echo "=========================================="

# ==========================================
# 1. KL_BETA=0.12 推理
# ==========================================
echo ""
echo "=========================================="
echo "步骤 1/2: KL_BETA=0.12 推理"
echo "=========================================="

KL012_CKPT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl012_balanced"
KL012_PT_PATH="${KL012_CKPT_DIR}/grpo_epoch${KL012_EPOCH}.pt"
KL012_CKPT_PATH="${KL012_CKPT_DIR}/grpo_epoch${KL012_EPOCH}_v2format.ckpt"

# 检查 checkpoint 是否存在
if [ ! -f "$KL012_PT_PATH" ]; then
    echo "错误: KL_BETA=0.12 checkpoint 不存在: ${KL012_PT_PATH}"
    echo "可用的 checkpoint:"
    ls -1 "${KL012_CKPT_DIR}"/grpo_epoch*.pt 2>/dev/null | tail -5 || echo "  无"
    exit 1
fi

# 如果 v2format 不存在，先转换
if [ ! -f "$KL012_CKPT_PATH" ]; then
    echo "转换 KL_BETA=0.12 checkpoint 格式..."
    cd "${PROJECT_ROOT}"
    source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
    conda activate lever_env
    python scripts/convert_v3_to_v2_format.py --v3_ckpt "${KL012_PT_PATH}"
fi

if [ ! -f "$KL012_CKPT_PATH" ]; then
    echo "错误: KL_BETA=0.12 转换后的 checkpoint 不存在: ${KL012_CKPT_PATH}"
    exit 1
fi

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export LEVER_LM_CHECKPOINT_PATH="${KL012_CKPT_PATH}"

echo "KL_BETA=0.12 推理配置:"
echo "  - Checkpoint: ${KL012_CKPT_PATH}"
echo "  - Epoch: ${KL012_EPOCH}"
echo "  - Test Samples: ${TEST_NUM}"
echo "  - GPU: ${GPU_ID} (CUDA_VISIBLE_DEVICES=${GPU_ID}, 在PyTorch中映射为cuda:0)"
echo "开始推理..."

# 注意：由于设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到指定的 GPU，并将其映射为 cuda:0
# 所以这里传递 0 作为 device 参数
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM}

echo ""
echo "✓ KL_BETA=0.12 推理完成"

# ==========================================
# 2. KL_BETA=0.15 推理
# ==========================================
echo ""
echo "=========================================="
echo "步骤 2/2: KL_BETA=0.15 推理"
echo "=========================================="

KL015_CKPT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl015_balanced_v2"
KL015_PT_PATH="${KL015_CKPT_DIR}/grpo_epoch${KL015_EPOCH}.pt"
KL015_CKPT_PATH="${KL015_CKPT_DIR}/grpo_epoch${KL015_EPOCH}_v2format.ckpt"

# 检查 checkpoint 是否存在
if [ ! -f "$KL015_PT_PATH" ]; then
    echo "错误: KL_BETA=0.15 checkpoint 不存在: ${KL015_PT_PATH}"
    echo "可用的 checkpoint:"
    ls -1 "${KL015_CKPT_DIR}"/grpo_epoch*.pt 2>/dev/null | tail -5 || echo "  无"
    exit 1
fi

# 如果 v2format 不存在，先转换
if [ ! -f "$KL015_CKPT_PATH" ]; then
    echo "转换 KL_BETA=0.15 checkpoint 格式..."
    python scripts/convert_v3_to_v2_format.py --v3_ckpt "${KL015_PT_PATH}"
fi

if [ ! -f "$KL015_CKPT_PATH" ]; then
    echo "错误: KL_BETA=0.15 转换后的 checkpoint 不存在: ${KL015_CKPT_PATH}"
    exit 1
fi

export LEVER_LM_CHECKPOINT_PATH="${KL015_CKPT_PATH}"

echo "KL_BETA=0.15 推理配置:"
echo "  - Checkpoint: ${KL015_CKPT_PATH}"
echo "  - Epoch: ${KL015_EPOCH}"
echo "  - Test Samples: ${TEST_NUM}"
echo "  - GPU: ${GPU_ID} (CUDA_VISIBLE_DEVICES=${GPU_ID}, 在PyTorch中映射为cuda:0)"
echo "开始推理..."

# 注意：由于设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到指定的 GPU，并将其映射为 cuda:0
# 所以这里传递 0 作为 device 参数
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM}

echo ""
echo "✓ KL_BETA=0.15 推理完成"

# ==========================================
# 3. 结果对比提示
# ==========================================
echo ""
echo "=========================================="
echo "✓ 对比实验完成！"
echo "=========================================="
echo ""
echo "结果文件位置："
echo "  KL_BETA=0.12: results/${DATASET_NAME}/icl_inference/v3/"
echo "  KL_BETA=0.15: results/${DATASET_NAME}/icl_inference/v3/"
echo ""
echo "请查看结果文件进行对比分析。"
echo "=========================================="

