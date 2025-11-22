#!/bin/bash

# 前台运行 generate_data.sh 脚本
# 用法: bash scripts/run_generate_data.sh [task] [dataset] [gpu_ids] [sampler] [beam_model]
# 示例: bash scripts/run_generate_data.sh vqa okvqa_local "[0]" rand_sampler qwen2.5_vl_3B

task=${1:-vqa}
dataset=${2:-okvqa_local}
gpu_ids=${3:-"[0]"}
sampler=${4:-rand_sampler}
beam_model=${5:-flamingo_3B}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目目录
cd "$PROJECT_DIR" || exit 1

# 构建完整的命令
cmd="bash scripts/generate_data.sh $task $dataset \"$gpu_ids\" $sampler $beam_model"

# 前台运行
echo "=========================================="
echo "🚀 启动任务（前台运行）"
echo "=========================================="
echo "任务: $task"
echo "数据集: $dataset"
echo "GPU IDs: $gpu_ids"
echo "采样器: $sampler"
echo "模型: $beam_model"
echo "命令: $cmd"
echo "=========================================="
echo ""

# 直接执行命令（前台运行）
$cmd

