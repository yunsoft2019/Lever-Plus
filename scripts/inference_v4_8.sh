#!/bin/bash
# V4-8 方案推理脚本
#
# 使用方法:
#   bash scripts/inference_v4_8.sh [gpu_id] [stage] [epoch] [num_slot_layers]
#
# 参数:
#   gpu_id: GPU ID (默认 0)
#   stage: rce 或 grpo (默认 grpo)
#   epoch: checkpoint epoch (默认 1)
#   num_slot_layers: slot self-attention 层数 (默认 2)
#
# 示例:
#   bash scripts/inference_v4_8.sh 4 grpo 36 2
#   bash scripts/inference_v4_8.sh 4 rce 2 2

set -e

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 解析参数
gpu_id=${1:-0}
stage=${2:-grpo}
epoch=${3:-1}
num_slot_layers=${4:-2}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Checkpoint 路径
ckpt_dir="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_8_slots${num_slot_layers}"
ckpt_path="${ckpt_dir}/${stage}_epoch${epoch}.pt"
converted_ckpt="${ckpt_dir}/${stage}_epoch${epoch}_v2format.ckpt"

echo "=========================================="
echo "V4-8 推理 (Slot Decoder ${num_slot_layers} layers)"
echo "=========================================="
echo "  - Checkpoint: $ckpt_path"
echo "  - Stage: $stage"
echo "  - Epoch: $epoch"
echo "  - GPU: $gpu_id"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -f "$ckpt_path" ]; then
    echo "错误: Checkpoint 不存在: $ckpt_path"
    echo "可用的 checkpoints:"
    ls -la "$ckpt_dir"/*.pt 2>/dev/null || echo "  无"
    exit 1
fi

# 转换 checkpoint 到 V2 格式（如果不存在）
if [ ! -f "$converted_ckpt" ]; then
    echo ""
    echo "步骤 1: 转换 checkpoint 到 V2 格式..."
    python scripts/convert_v4_8_to_v2_format.py \
        --input "$ckpt_path" \
        --output "$converted_ckpt"
    echo "✓ Checkpoint 转换完成"
fi

# 设置环境变量
export LEVER_LM_CHECKPOINT_PATH="$converted_ckpt"
export LEVER_LM_MODEL_TYPE="v4_8"
export LEVER_LM_NUM_SLOT_LAYERS="$num_slot_layers"

echo ""
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo "Model Type: V4-8 (Slot Decoder ${num_slot_layers} layers)"
echo ""

# 运行推理（800 个样本，shot 1-4）
cd ${PROJECT_DIR}

echo "=========================================="
echo "Running Shot 1-4..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=${gpu_id} python icl_inference.py \
    train="query_img_text_icd_img_text_v2" \
    ex_name="main_vqa_RandSampler_Qwen2_5_VL_3B_Instruct_query_img_text_icd_img_text" \
    dataset=okvqa_local \
    task=vqa \
    device=cuda:0 \
    inference_bs=1 \
    test_data_num=800 \
    test_lever_lm=true \
    infer_model=qwen2.5_vl_3B \
    infer_model.load_from_local=false

echo ""
echo "=========================================="
echo "✓ V4-8 推理完成！"
echo "=========================================="
