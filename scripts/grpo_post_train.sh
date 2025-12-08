#!/bin/bash
# GRPO Post-Training 启动脚本
# 来自强化学习.md 2.5节

# 使用方法：
# bash scripts/grpo_post_train.sh \
#     --sft_ckpt results/okvqa/model_cpk/v2/xxx.ckpt \
#     --beam_data results/okvqa/generated_data/xxx.json \
#     --img_emb results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-ImgFeatures.pth \
#     --text_emb results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-TextFeatures.pth \
#     --output_dir results/okvqa/model_cpk/v3/

set -e

# 默认参数
SFT_CKPT=""
BEAM_DATA=""
IMG_EMB=""
TEXT_EMB=""
OUTPUT_DIR="results/grpo"
RCE_EPOCHS=1
GRPO_EPOCHS=3
BATCH_SIZE=32
RCE_LR=1e-4
GRPO_LR=1e-5
GPU=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --sft_ckpt)
            SFT_CKPT="$2"
            shift 2
            ;;
        --beam_data)
            BEAM_DATA="$2"
            shift 2
            ;;
        --img_emb)
            IMG_EMB="$2"
            shift 2
            ;;
        --text_emb)
            TEXT_EMB="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --rce_epochs)
            RCE_EPOCHS="$2"
            shift 2
            ;;
        --grpo_epochs)
            GRPO_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --rce_lr)
            RCE_LR="$2"
            shift 2
            ;;
        --grpo_lr)
            GRPO_LR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$SFT_CKPT" ]; then
    echo "错误: 请指定 --sft_ckpt"
    exit 1
fi

if [ -z "$BEAM_DATA" ]; then
    echo "错误: 请指定 --beam_data"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "GRPO Post-Training"
echo "========================================"
echo "SFT检查点: $SFT_CKPT"
echo "束搜索数据: $BEAM_DATA"
echo "图像Embedding: $IMG_EMB"
echo "文本Embedding: $TEXT_EMB"
echo "输出目录: $OUTPUT_DIR"
echo "RCE epochs: $RCE_EPOCHS"
echo "GRPO epochs: $GRPO_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "RCE LR: $RCE_LR"
echo "GRPO LR: $GRPO_LR"
echo "GPU: $GPU"
echo "========================================"

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.grpo_post_train \
    --sft_ckpt \"$SFT_CKPT\" \
    --beam_data \"$BEAM_DATA\" \
    --output_dir \"$OUTPUT_DIR\" \
    --rce_epochs $RCE_EPOCHS \
    --grpo_epochs $GRPO_EPOCHS \
    --batch_size $BATCH_SIZE \
    --rce_lr $RCE_LR \
    --grpo_lr $GRPO_LR \
    --device cuda:0"

# 添加可选的embedding参数
if [ -n "$IMG_EMB" ]; then
    CMD="$CMD --img_emb \"$IMG_EMB\""
fi

if [ -n "$TEXT_EMB" ]; then
    CMD="$CMD --text_emb \"$TEXT_EMB\""
fi

# 运行训练
eval $CMD

echo "========================================"
echo "✓ GRPO Post-Training 完成！"
echo "========================================"
