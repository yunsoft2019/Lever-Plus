#!/bin/bash

# 后台运行 generate_data.sh 脚本
# 用法: bash scripts/run_generate_data_background.sh [task] [dataset] [gpu_ids] [sampler] [beam_model] [log_file]
# 示例: bash scripts/run_generate_data_background.sh vqa okvqa_local "[0]" rand_sampler qwen2.5_vl_3B

task=${1:-vqa}
dataset=${2:-okvqa_local}
gpu_ids=${3:-"[0]"}
sampler=${4:-rand_sampler}
beam_model=${5:-flamingo_3B}
log_file=${6:-""}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 如果没有指定日志文件，自动生成
if [ -z "$log_file" ]; then
    # 清理数据集名称（去掉_local后缀）
    dataset_clean=$(echo "$dataset" | sed 's/_local$//')
    # 清理模型名称（替换特殊字符）
    model_clean=$(echo "$beam_model" | sed 's/\./_/g' | sed 's/\//_/g')
    # 生成日志文件名
    log_file="logs/generate_data/${task}_${dataset_clean}_${sampler}_${model_clean}.log"
fi

# 创建日志目录（如果不存在）
mkdir -p "$(dirname "$log_file")"

# 切换到项目目录
cd "$PROJECT_DIR" || exit 1

# 构建完整的命令
cmd="bash scripts/generate_data.sh $task $dataset \"$gpu_ids\" $sampler $beam_model"

# 后台运行，输出重定向到日志文件
echo "=========================================="
echo "🚀 启动后台任务"
echo "=========================================="
echo "任务: $task"
echo "数据集: $dataset"
echo "GPU IDs: $gpu_ids"
echo "采样器: $sampler"
echo "模型: $beam_model"
echo "命令: $cmd"
echo "日志文件: $log_file"
echo "PID文件: ${log_file}.pid"
echo "=========================================="

# 使用 nohup 后台运行，并将输出重定向到日志文件
nohup $cmd > "$log_file" 2>&1 &
PID=$!

# 保存进程ID
echo $PID > "${log_file}.pid"

echo ""
echo "✅ 后台任务已启动"
echo "   进程ID: $PID"
echo ""
echo "📋 常用命令:"
echo "   查看日志: tail -f $log_file"
echo "   查看最新日志: tail -n 100 $log_file"
echo "   停止任务: kill \$(cat ${log_file}.pid)"
echo "   检查状态: ps -p \$(cat ${log_file}.pid)"
echo "=========================================="



