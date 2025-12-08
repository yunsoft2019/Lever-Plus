#!/bin/bash
# 查找v2 checkpoint文件的帮助脚本

# 参数
VERSION=${1:-v2}
DATASET=${2:-okvqa}
SAMPLER=${3:-RandSampler}
MODEL=${4:-Qwen2_5_VL_3B_Instruct}

# 构建检查点目录
CHECKPOINT_DIR="./results/${DATASET}/model_cpk/${VERSION}"

echo "查找检查点文件..."
echo "目录: ${CHECKPOINT_DIR}"
echo "采样器: ${SAMPLER}"
echo "模型: ${MODEL}"
echo ""

# 查找匹配的文件
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "找到的文件："
    ls -lh "${CHECKPOINT_DIR}"/*${SAMPLER}*.ckpt 2>/dev/null | head -10
    
    echo ""
    echo "最新的文件："
    LATEST=$(ls -t "${CHECKPOINT_DIR}"/*${SAMPLER}*.ckpt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "  ${LATEST}"
        echo ""
        echo "使用以下路径："
        echo "  --sft_ckpt ${LATEST}"
    else
        echo "  未找到匹配的文件"
    fi
else
    echo "目录不存在: ${CHECKPOINT_DIR}"
fi
