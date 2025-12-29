#!/bin/bash
# V4-7 方案推理脚本

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 使用方法: bash scripts/inference_v4_7.sh [gpu_id] [phase] [epoch] [dpp_rank]
#
# 参数:
#   gpu_id: GPU 编号（默认 0）
#   phase: 训练阶段 rce 或 grpo（默认 rce）
#   epoch: epoch 编号（默认 2）
#   dpp_rank: DPP 低秩投影维度（默认 32）
#
# 示例:
#   bash scripts/inference_v4_7.sh 3 rce 2 32
#   bash scripts/inference_v4_7.sh 3 grpo 1 32

set -e

# 解析参数
gpu_id=${1:-0}
phase=${2:-rce}
epoch=${3:-2}
dpp_rank=${4:-32}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# V4-7 checkpoint 目录（根据 dpp_rank 区分）
CKPT_DIR="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_7_rank${dpp_rank}"

echo "=========================================="
echo "V4-7 方案推理 (DPP rank=${dpp_rank})"
echo "=========================================="

# 检查 checkpoint 目录是否存在
if [ ! -d "$CKPT_DIR" ]; then
    echo "错误: Checkpoint 目录不存在: $CKPT_DIR"
    echo "请先运行训练: bash scripts/train_v3_plan_v4_7.sh [gpu_id] ${dpp_rank}"
    exit 1
fi

# 构建 checkpoint 路径
V4_7_CKPT="${CKPT_DIR}/${phase}_epoch${epoch}.pt"
V2_CKPT="${CKPT_DIR}/${phase}_epoch${epoch}_v2format.ckpt"

echo ""
echo "使用 Phase: ${phase}, Epoch: ${epoch}"
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

# 转换为 v2 格式（如果不存在）
if [ ! -f "$V2_CKPT" ]; then
    echo "正在转换 checkpoint 为 v2 格式..."
    python ${PROJECT_DIR}/scripts/convert_v4_7_to_v2_format.py \
        --v4_7_ckpt "$V4_7_CKPT" \
        --output "$V2_CKPT" \
        --dpp_rank "$dpp_rank"
    if [ ! -f "$V2_CKPT" ]; then
        echo "转换失败，请检查错误信息"
        exit 1
    fi
    echo "✓ Checkpoint 转换完成"
fi

# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH="$V2_CKPT"
# 设置模型类型为 V4-7
export LEVER_LM_MODEL_TYPE="v4_7"
# 设置 dpp_rank
export LEVER_LM_DPP_RANK="$dpp_rank"

echo ""
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo "Model Type: V4-7 (GRU + DPP rank=${dpp_rank})"
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
echo "✓ V4-7 (DPP rank=${dpp_rank}) 推理完成！"
echo "=========================================="
echo "请查看结果并与其他方案对比"
echo "=========================================="
