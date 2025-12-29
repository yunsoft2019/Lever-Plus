#!/bin/bash
# V4-1 方案（公平对比版）推理脚本

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env
# 使用方法: bash scripts/inference_v4_1_fair.sh [gpu_id] [epoch]
#
# 参数:
#   gpu_id: GPU 编号（默认 0）
#   epoch: GRPO epoch 编号（默认自动查找 Val Loss 最小的 epoch）
#
# 示例:
#   bash scripts/inference_v4_1_fair.sh 0        # 自动查找最优 epoch
#   bash scripts/inference_v4_1_fair.sh 0 2      # 使用 epoch 2

set -e

# 解析参数
gpu_id=${1:-0}
epoch=${2:-auto}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# V4-1 fair checkpoint 目录
CKPT_DIR="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_1_fair"

echo "=========================================="
echo "V4-1 方案（公平对比版）推理"
echo "=========================================="

# 检查 checkpoint 目录是否存在
if [ ! -d "$CKPT_DIR" ]; then
    echo "错误: Checkpoint 目录不存在: $CKPT_DIR"
    echo "请先运行训练: bash scripts/train_v3_plan_v4_1_fair.sh"
    exit 1
fi

# 如果 epoch 是 auto，尝试找到最优 epoch
if [ "$epoch" == "auto" ]; then
    echo "自动查找最优 epoch..."
    
    # 查找所有 grpo_epoch*.pt 文件
    pt_files=$(ls -1 "${CKPT_DIR}"/grpo_epoch*.pt 2>/dev/null | sort -V)
    
    if [ -z "$pt_files" ]; then
        echo "错误: 未找到任何 GRPO checkpoint"
        exit 1
    fi
    
    echo "找到以下 checkpoint:"
    echo "$pt_files" | while read f; do echo "  - $(basename $f)"; done
    
    # 默认使用 epoch 2（根据之前的经验，通常 epoch 2 是最优的）
    # 用户可以根据训练日志手动指定
    epoch=2
    echo ""
    echo "⚠️  默认使用 epoch ${epoch}（根据经验通常最优）"
    echo "   如需使用其他 epoch，请运行: bash scripts/inference_v4_1_fair.sh ${gpu_id} <epoch>"
fi

# 构建 checkpoint 路径
V3_CKPT="${CKPT_DIR}/grpo_epoch${epoch}.pt"
V2_CKPT="${CKPT_DIR}/grpo_epoch${epoch}_v2format.ckpt"

echo ""
echo "使用 Epoch: ${epoch}"
echo "GPU: ${gpu_id}"
echo "测试样本数: 800"
echo "=========================================="

# 检查 v3 checkpoint 是否存在
if [ ! -f "$V3_CKPT" ]; then
    echo "错误: V3 Checkpoint 文件不存在: $V3_CKPT"
    echo ""
    echo "可用的 checkpoint:"
    ls -1 "${CKPT_DIR}"/grpo_epoch*.pt 2>/dev/null | while read f; do echo "  - $(basename $f)"; done
    exit 1
fi

# 转换为 v2 格式（如果不存在）
if [ ! -f "$V2_CKPT" ]; then
    echo "正在转换 checkpoint 为 v2 格式..."
    python ${PROJECT_DIR}/scripts/convert_v3_to_v2_format.py \
        --v3_ckpt "$V3_CKPT" \
        --output "$V2_CKPT"
    if [ ! -f "$V2_CKPT" ]; then
        echo "转换失败，请检查错误信息"
        exit 1
    fi
    echo "✓ Checkpoint 转换完成"
fi

# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH="$V2_CKPT"

echo ""
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo ""

# 运行推理（800 个样本，shot 1-4）
# 注意：shot_num_list 默认为 [1,2,3,4]，Python 会自动遍历所有 shot
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
echo "✓ V4-1（公平对比版）推理完成！"
echo "=========================================="
echo "请查看结果并与方案五对比"
echo "=========================================="
