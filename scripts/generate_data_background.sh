#!/bin/bash
# 后台运行 generate_data.sh，支持实时查看进度

task=${1:-vqa}
dataset=${2:-vqav2_local}
gpu_ids=${3:-"[0]"}
sampler=${4:-rand_sampler}
beam_model=${5:-qwen2.5_vl_3B}

# 生成日志文件名
log_file="logs/generate_data_${task}_${dataset}_${sampler}_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "启动后台数据生成任务"
echo "=========================================="
echo "任务: $task"
echo "数据集: $dataset"
echo "GPU IDs: $gpu_ids"
echo "采样器: $sampler"
echo "模型: $beam_model"
echo "日志文件: $log_file"
echo "=========================================="
echo ""

# 使用 python -u 禁用缓冲，stdbuf -oL 设置行缓冲
# 这样进度条可以实时写入日志文件
PYTHONUNBUFFERED=1 stdbuf -oL -eL bash scripts/generate_data.sh "$task" "$dataset" "$gpu_ids" "$sampler" "$beam_model" > "$log_file" 2>&1 &
PID=$!

echo "✓ 后台任务已启动"
echo "  进程ID: $PID"
echo "  日志文件: $log_file"
echo ""
echo "📋 查看进度命令:"
echo "  tail -f $log_file"
echo "  或使用: bash scripts/check_generate_data_progress.sh"
echo ""
echo "🛑 停止任务:"
echo "  kill $PID"
echo ""

# 保存 PID 到文件
echo $PID > "${log_file}.pid"
echo "PID 已保存到: ${log_file}.pid"



