#!/bin/bash
# V4-7 方案原生推理脚本（直接加载 .pt 文件，保留完整 DPP 机制）

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 使用方法: bash scripts/inference_v4_7_native.sh [gpu_id] [phase] [epoch] [dpp_rank]
#
# 参数:
#   gpu_id: GPU 编号（默认 0）
#   phase: 训练阶段 rce 或 grpo（默认 grpo）
#   epoch: epoch 编号（默认 4）
#   dpp_rank: DPP 低秩投影维度（默认 32）
#
# 示例:
#   bash scripts/inference_v4_7_native.sh 2 grpo 4 32

set -e

# 解析参数
gpu_id=${1:-0}
phase=${2:-grpo}
epoch=${3:-4}
dpp_rank=${4:-32}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# V4-7 checkpoint 目录
CKPT_DIR="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_7_rank${dpp_rank}"
V4_7_CKPT="${CKPT_DIR}/${phase}_epoch${epoch}.pt"

echo "=========================================="
echo "V4-7 原生推理 (DPP rank=${dpp_rank})"
echo "=========================================="
echo "使用完整 V4-7 模型，保留 DPP 多样性增益机制"
echo ""
echo "Phase: ${phase}, Epoch: ${epoch}"
echo "DPP Rank: ${dpp_rank}"
echo "GPU: ${gpu_id}"
echo "测试样本数: 800"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -f "$V4_7_CKPT" ]; then
    echo "错误: Checkpoint 文件不存在: $V4_7_CKPT"
    echo ""
    echo "可用的 checkpoint:"
    ls -1 "${CKPT_DIR}"/*.pt 2>/dev/null | while read f; do echo "  - $(basename $f)"; done
    exit 1
fi

# 设置环境变量，直接使用 .pt 文件
export LEVER_LM_CHECKPOINT_PATH="$V4_7_CKPT"
export LEVER_LM_MODEL_TYPE="v4_7_native"
export LEVER_LM_DPP_RANK="$dpp_rank"

echo ""
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo "Model Type: V4-7 Native (完整 DPP)"
echo ""

# 运行推理
cd ${PROJECT_DIR}

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
echo "✓ V4-7 原生推理完成！"
echo "=========================================="
