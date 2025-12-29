#!/bin/bash
# V4-10 方案推理脚本 (STOP 自适应 shot)

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 使用方法: bash scripts/inference_v4_10.sh [gpu_id] [stage] [epoch]
#
# 参数:
#   gpu_id: GPU 编号（默认 0）
#   stage: rce 或 grpo（默认 grpo）
#   epoch: epoch 编号（默认 46，根据训练日志 Val Loss 最小）
#
# 示例:
#   bash scripts/inference_v4_10.sh 1 grpo 46
#   bash scripts/inference_v4_10.sh 1 rce 15

set -e

# 解析参数
gpu_id=${1:-0}
stage=${2:-grpo}
epoch=${3:-46}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# V4-10 checkpoint 目录
CKPT_DIR="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_10"

echo "=========================================="
echo "V4-10 方案推理 (STOP 自适应 shot)"
echo "=========================================="

# 检查 checkpoint 目录是否存在
if [ ! -d "$CKPT_DIR" ]; then
    echo "错误: Checkpoint 目录不存在: $CKPT_DIR"
    echo "请先运行训练: bash scripts/train_v3_plan_v4_10.sh [gpu_id]"
    exit 1
fi

# 构建 checkpoint 路径
V4_10_CKPT="${CKPT_DIR}/${stage}_epoch${epoch}.pt"
V2_CKPT="${CKPT_DIR}/${stage}_epoch${epoch}_v2format.ckpt"

echo ""
echo "Stage: ${stage}"
echo "Epoch: ${epoch}"
echo "GPU: ${gpu_id}"
echo "测试样本数: 800"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -f "$V4_10_CKPT" ]; then
    echo "错误: Checkpoint 文件不存在: $V4_10_CKPT"
    echo ""
    echo "可用的 checkpoint:"
    ls -1 "${CKPT_DIR}"/*.pt 2>/dev/null | while read f; do echo "  - $(basename $f)"; done
    exit 1
fi

# 转换为 v2 格式（如果不存在）
if [ ! -f "$V2_CKPT" ]; then
    echo "正在转换 checkpoint 为 v2 格式..."
    python ${PROJECT_DIR}/scripts/convert_v4_10_to_v2_format.py \
        --input "$V4_10_CKPT" \
        --output "$V2_CKPT"
    if [ ! -f "$V2_CKPT" ]; then
        echo "转换失败，请检查错误信息"
        exit 1
    fi
    echo "✓ Checkpoint 转换完成"
fi

# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH="$V2_CKPT"
# 设置模型类型为 V4-10
export LEVER_LM_MODEL_TYPE="v4_10"

echo ""
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo "Model Type: V4-10 (STOP 自适应 shot)"
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
echo "✓ V4-10 (STOP 自适应 shot) 推理完成！"
echo "=========================================="
echo "请查看结果并与其他方案对比"
echo "=========================================="
