#!/bin/bash
# VQAv2 Diverse 数据训练模型推理脚本

set -e

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

gpu_id=${1:-0}
stage=${2:-grpo}
epoch=${3:-1}
sample_num=${4:-800}

PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

export CHECKPOINT_PATH="${PROJECT_DIR}/checkpoints"
export COCO_PATH="${PROJECT_DIR}/datasets/mscoco"
export VQAV2_PATH="${PROJECT_DIR}/datasets/vqav2"
export RESULT_DIR="${PROJECT_DIR}/results"

CKPT_DIR="${PROJECT_DIR}/results/vqav2/model_cpk/v3_diverse"
CKPT_PATH="${CKPT_DIR}/${stage}_epoch${epoch}.pt"

echo "=========================================="
echo "VQAv2 Diverse 模型推理"
echo "=========================================="
echo "  - Checkpoint: ${stage}_epoch${epoch}.pt"
echo "  - Sample Num: ${sample_num}"
echo "  - GPU: ${gpu_id}"
echo "=========================================="

if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: Checkpoint 不存在: $CKPT_PATH"
    exit 1
fi

# 转换为 v2 格式
V2_CKPT="${CKPT_PATH%.pt}_v2format.ckpt"
if [ ! -f "$V2_CKPT" ]; then
    echo "转换 checkpoint 为 v2 格式..."
    python "${PROJECT_DIR}/scripts/convert_v3_to_v2_format.py" --v3_ckpt "$CKPT_PATH"
fi

export LEVER_LM_CHECKPOINT_PATH="${V2_CKPT}"

CUDA_VISIBLE_DEVICES=${gpu_id} python "${PROJECT_DIR}/icl_inference.py" \
    train="query_img_text_icd_img_text_v2" \
    ex_name="main_vqa_RandSampler_Qwen2_5_VL_3B_Instruct_query_img_text_icd_img_text" \
    dataset=vqav2_local \
    task=vqa \
    device="cuda:0" \
    inference_bs=1 \
    test_data_num=${sample_num} \
    index_data_num=5000 \
    test_lever_lm=true \
    infer_model=qwen2.5_vl_3B \
    infer_model.load_from_local=false

echo "✓ 推理完成！"
