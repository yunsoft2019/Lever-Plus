#!/bin/bash

# 后台运行推理脚本
# 用法: bash scripts/run_inference_background.sh [task] [dataset] [device] [lever_lm] [sampler] [beam_model] [version]
# 示例: bash scripts/run_inference_background.sh vqa okvqa_local 3 query_img_text_icd_img_text rand_sampler flamingo_3B v1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 获取参数
task=${1:-vqa}
dataset=${2:-okvqa_local}
device=${3:-0}
lever_lm=${4:-query_img_text_icd_img_text}
sampler=${5:-rand_sampler}
beam_model=${6:-flamingo_3B}
version=${7:-v0}

# 创建日志目录
log_dir="${PROJECT_DIR}/logs/inference"
mkdir -p "$log_dir"

# 生成日志文件名（包含时间戳和参数信息）
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/inference_${task}_${dataset}_${device}_${sampler}_${beam_model}_${version}_${timestamp}.log"

# 初始化 conda（如果可用）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
fi

# 构建命令（激活 conda 环境后运行）
cmd="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate lever_env && cd ${PROJECT_DIR} && bash ${SCRIPT_DIR}/inference.sh ${task} ${dataset} ${device} ${lever_lm} ${sampler} ${beam_model} ${version}"

echo "=========================================="
echo "后台运行推理任务"
echo "=========================================="
echo "任务: ${task}"
echo "数据集: ${dataset}"
echo "设备: ${device}"
echo "Lever LM: ${lever_lm}"
echo "采样器: ${sampler}"
echo "模型: ${beam_model}"
echo "版本: ${version}"
echo "Conda 环境: lever_env"
echo "日志文件: ${log_file}"
echo "=========================================="

# 使用 nohup 后台运行，并将输出重定向到日志文件
nohup bash -c "$cmd" > "$log_file" 2>&1 &

# 获取进程 ID
pid=$!

# 保存 PID 到文件（可选，用于后续管理）
pid_file="${log_dir}/inference_${task}_${dataset}_${device}_${sampler}_${beam_model}_${version}_${timestamp}.pid"
echo $pid > "$pid_file"

echo "✅ 推理任务已在后台启动"
echo "进程 ID: $pid"
echo "PID 文件: $pid_file"
echo "日志文件: $log_file"
echo ""
echo "查看日志: tail -f ${log_file}"
echo "查看进程: ps -p $pid"
echo "停止任务: kill $pid"

