#!/bin/bash
# 使用 inference.sh (V2推理流程) 测试 v3 训练的模型
# 这样可以使用完整训练集作为候选池，与2025-12-12的结果对比

set -e

# 加载环境变量
if [ -f .env ]; then
    source .env
    echo "✓ 已加载.env文件环境变量"
fi

# 配置
GPU=${1:-0}
TEST_NUM=${2:-800}  # 测试样本数，-1表示全部
CHECKPOINT_DIR=${3:-"results/okvqa/model_cpk/v3_final_merged_20251219"}

echo "=========================================="
echo "使用 inference.sh 测试 v3 模型"
echo "=========================================="
echo "GPU: ${GPU}"
echo "测试样本数: ${TEST_NUM}"
echo "Checkpoint目录: ${CHECKPOINT_DIR}"

# 找到最新的checkpoint
LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/grpo_epoch*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/rce_epoch*.pt 2>/dev/null | head -1)
fi

if [ -z "$LATEST_CKPT" ]; then
    echo "错误: 未找到checkpoint文件"
    exit 1
fi

echo "使用checkpoint: ${LATEST_CKPT}"

# 转换为v2格式
V2_CKPT="${LATEST_CKPT%.pt}_v2format.ckpt"
if [ ! -f "$V2_CKPT" ]; then
    echo "转换checkpoint为v2格式..."
    python scripts/convert_v3_to_v2_format.py --v3_ckpt "$LATEST_CKPT" --output "$V2_CKPT"
fi

echo "V2格式checkpoint: ${V2_CKPT}"

# 设置环境变量，让inference.sh使用指定的checkpoint
export LEVER_LM_CHECKPOINT_PATH="${V2_CKPT}"

# 运行推理
echo ""
echo "开始推理..."
echo "=========================================="

# 使用inference.sh进行推理
# 参数: task dataset device lever_lm sampler beam_model version test_data_num
bash scripts/inference.sh vqa okvqa_local ${GPU} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM}

echo ""
echo "=========================================="
echo "推理完成！"
echo "=========================================="
