#!/bin/bash
# GRPO强化学习模型推理脚本
# 用于评估V3模型（GRPO训练后）的范例选择效果

set -e

# 默认参数
GRPO_CKPT=""
IMG_EMB=""
BEAM_DATA=""
DATASET="okvqa"
MODEL_NAME="Qwen2.5-VL-3B-Instruct"
SHOT_NUM=2
TEST_NUM=100
OUTPUT_DIR="results/v3_eval"
BATCH_SIZE=32
GPU=0
SKIP_VQA=false

# 显示帮助
show_help() {
    echo "GRPO强化学习模型推理脚本"
    echo ""
    echo "用法："
    echo "  bash scripts/grpo_inference.sh [选项]"
    echo ""
    echo "选项："
    echo "  --grpo_ckpt PATH    GRPO检查点路径 (必需)"
    echo "  --img_emb PATH      图像embedding路径 (必需)"
    echo "  --beam_data PATH    束搜索数据JSON路径 (必需)"
    echo "  --dataset NAME      数据集名称 (默认: okvqa)"
    echo "  --model_name NAME   VLM模型名称 (默认: Qwen2.5-VL-3B-Instruct)"
    echo "  --shot_num N        范例数量 (默认: 2)"
    echo "  --test_num N        测试样本数 (默认: 100)"
    echo "  --output_dir DIR    输出目录 (默认: results/v3_eval)"
    echo "  --batch_size N      批次大小 (默认: 32)"
    echo "  --gpu N             GPU编号 (默认: 0)"
    echo "  --skip_vqa          跳过VQA推理，只做范例选择"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "示例："
    echo "  # 推理前100条（仅范例选择）"
    echo "  bash scripts/grpo_inference.sh \\"
    echo "      --grpo_ckpt results/okvqa/model_cpk/v3/grpo_epoch3.pt \\"
    echo "      --img_emb results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-ImgFeatures.pth \\"
    echo "      --beam_data results/okvqa/generated_data/xxx.json \\"
    echo "      --test_num 100 \\"
    echo "      --skip_vqa"
    echo ""
    echo "  # 推理前100条（含VQA评估）"
    echo "  bash scripts/grpo_inference.sh \\"
    echo "      --grpo_ckpt results/okvqa/model_cpk/v3/grpo_epoch3.pt \\"
    echo "      --img_emb results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-ImgFeatures.pth \\"
    echo "      --beam_data results/okvqa/generated_data/xxx.json \\"
    echo "      --test_num 100"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --grpo_ckpt)
            GRPO_CKPT="$2"
            shift 2
            ;;
        --img_emb)
            IMG_EMB="$2"
            shift 2
            ;;
        --beam_data)
            BEAM_DATA="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --shot_num)
            SHOT_NUM="$2"
            shift 2
            ;;
        --test_num)
            TEST_NUM="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
        --skip_vqa)
            SKIP_VQA=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$GRPO_CKPT" ]; then
    echo "错误: 请指定 --grpo_ckpt"
    show_help
    exit 1
fi

if [ -z "$IMG_EMB" ]; then
    echo "错误: 请指定 --img_emb"
    show_help
    exit 1
fi

if [ -z "$BEAM_DATA" ]; then
    echo "错误: 请指定 --beam_data"
    show_help
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "GRPO V3模型推理"
echo "========================================"
echo "GRPO检查点: $GRPO_CKPT"
echo "图像Embedding: $IMG_EMB"
echo "束搜索数据: $BEAM_DATA"
echo "数据集: $DATASET"
echo "VLM模型: $MODEL_NAME"
echo "范例数: $SHOT_NUM"
echo "测试数量: $TEST_NUM"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "GPU: $GPU"
echo "跳过VQA: $SKIP_VQA"
echo "========================================"

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt \"$GRPO_CKPT\" \
    --img_emb \"$IMG_EMB\" \
    --beam_data \"$BEAM_DATA\" \
    --dataset $DATASET \
    --model_name \"$MODEL_NAME\" \
    --shot_num $SHOT_NUM \
    --test_num $TEST_NUM \
    --output_dir \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --device cuda:0"

# 添加skip_vqa选项
if [ "$SKIP_VQA" = true ]; then
    CMD="$CMD --skip_vqa"
fi

# 运行推理
echo ""
echo "执行命令:"
echo "$CMD"
echo ""
eval $CMD

echo "========================================"
echo "✓ GRPO V3模型推理完成！"
echo "  结果保存在: $OUTPUT_DIR"
echo "========================================"
