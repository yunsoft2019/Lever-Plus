#!/bin/bash

# 基线推理脚本（使用随机范例）
# 用法: bash scripts/baseline.sh [task] [dataset] [device] [model]
# 示例: bash scripts/baseline.sh vqa okvqa_local 0 flamingo_3B
#       bash scripts/baseline.sh vqa okvqa_local 1 qwen2.5_vl_3B

# Set default values
task=${1:-vqa}
dataset=${2:-okvqa_local}
device=${3:-0}
model=${4:-flamingo_3B}

# 检查是否提供了模型参数
if [ -z "$model" ]; then
    echo "错误: 必须指定模型参数"
    echo "用法: bash scripts/baseline.sh [task] [dataset] [device] [model]"
    echo "示例: bash scripts/baseline.sh vqa okvqa_local 0 flamingo_3B"
    echo "      bash scripts/baseline.sh vqa okvqa_local 1 qwen2.5_vl_3B"
    exit 1
fi

# 将 GPU 编号转换为 device 格式（cuda:0, cuda:1 等）
if [[ "$device" =~ ^[0-9]+$ ]]; then
    device_arg="cuda:${device}"
else
    # 如果已经是 cuda:0 格式，直接使用
    device_arg="${device}"
fi

echo "=========================================="
echo "Baseline Inference Configuration:"
echo "  Task: ${task}"
echo "  Dataset: ${dataset}"
echo "  GPU ID: ${device} → ${device_arg}"
echo "  Model: ${model}"
echo "  说明: 使用随机范例（RandomRetriever），测试指定模型"
echo "  Shot Num: 1, 2, 3, 4, 6, 8"
echo "=========================================="

# 运行基线推理（只测试指定模型）
python random_predict.py task=${task} \
                         dataset=${dataset} \
                         device=${device_arg} \
                         infer_model=${model}

