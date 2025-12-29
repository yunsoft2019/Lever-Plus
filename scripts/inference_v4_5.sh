#!/bin/bash
# V4-5 方案推理脚本

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 使用方法: bash scripts/inference_v4_5.sh [gpu_id] [epoch] [attention_type]
#
# 参数:
#   gpu_id: GPU 编号（默认 0）
#   epoch: GRPO epoch 编号（默认 1，需要根据训练日志选择 Val Loss 最小的）
#   attention_type: 打分头类型（默认 additive，可选 bilinear）
#
# 示例:
#   bash scripts/inference_v4_5.sh 4 1 additive
#   bash scripts/inference_v4_5.sh 4 2 bilinear

set -e

# 解析参数
gpu_id=${1:-0}
epoch=${2:-1}
attention_type=${3:-additive}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# V4-5 checkpoint 目录（根据 attention_type 区分）
CKPT_DIR="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_5_${attention_type}"

echo "=========================================="
echo "V4-5 方案推理 (${attention_type^^} Attention)"
echo "=========================================="

# 检查 checkpoint 目录是否存在
if [ ! -d "$CKPT_DIR" ]; then
    echo "错误: Checkpoint 目录不存在: $CKPT_DIR"
    echo "请先运行训练: bash scripts/train_v3_plan_v4_5.sh [gpu_id] ${attention_type}"
    exit 1
fi

# 构建 checkpoint 路径
V4_5_CKPT="${CKPT_DIR}/grpo_epoch${epoch}.pt"
V2_CKPT="${CKPT_DIR}/grpo_epoch${epoch}_v2format.ckpt"

echo ""
echo "使用 Epoch: ${epoch}"
echo "Attention Type: ${attention_type}"
echo "GPU: ${gpu_id}"
echo "测试样本数: 800"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -f "$V4_5_CKPT" ]; then
    echo "错误: Checkpoint 文件不存在: $V4_5_CKPT"
    echo ""
    echo "可用的 checkpoint:"
    ls -1 "${CKPT_DIR}"/grpo_epoch*.pt 2>/dev/null | while read f; do echo "  - $(basename $f)"; done
    exit 1
fi

# 转换为 v2 格式（如果不存在）
if [ ! -f "$V2_CKPT" ]; then
    echo "正在转换 checkpoint 为 v2 格式..."
    python ${PROJECT_DIR}/scripts/convert_v4_5_to_v2_format.py \
        --v4_5_ckpt "$V4_5_CKPT" \
        --output "$V2_CKPT" \
        --attention_type "$attention_type"
    if [ ! -f "$V2_CKPT" ]; then
        echo "转换失败，请检查错误信息"
        exit 1
    fi
    echo "✓ Checkpoint 转换完成"
fi

# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH="$V2_CKPT"
# 设置模型类型为 V4-5
export LEVER_LM_MODEL_TYPE="v4_5"
# 设置 attention type
export LEVER_LM_ATTENTION_TYPE="$attention_type"

echo ""
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo "Model Type: V4-5 (GRU + ${attention_type^^} Attention)"
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
echo "✓ V4-5 (${attention_type^^}) 推理完成！"
echo "=========================================="
echo "请查看结果并与方案五、V4-1、V4-2、V4-3 对比"
echo "=========================================="
