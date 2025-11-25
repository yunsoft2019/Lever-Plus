#!/bin/bash

# 后台运行训练脚本
# 用法: bash scripts/run_train_background.sh [task] [dataset] [gpu_id] [lever_lm] [sampler] [beam_model] [version]
# 示例: bash scripts/run_train_background.sh vqa okvqa_local 1 query_img_text_icd_img_text mix_sampler qwen2.5_vl_3B v1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 获取参数
task=${1:-vqa}
dataset=${2:-okvqa_local}
gpu_id=${3:-0}
lever_lm=${4:-query_img_text_icd_img_text}
sampler=${5:-rand_sampler}
beam_model=${6:-flamingo_3B}
version=${7:-v0}
conda_env="lever_env" # 指定 conda 环境名称

# 创建日志目录
log_dir="${PROJECT_DIR}/logs/inference"
mkdir -p "$log_dir"

# 生成日志文件名（包含时间戳和参数信息）
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/train_${task}_${dataset}_${gpu_id}_${sampler}_${beam_model}_${version}_${timestamp}.log"

# 构建命令
# 激活 conda 环境并在其中执行 train_lever_lm.sh
cmd="source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${conda_env} && cd ${PROJECT_DIR} && bash ${SCRIPT_DIR}/train_lever_lm.sh ${task} ${dataset} ${gpu_id} ${lever_lm} ${sampler} ${beam_model} ${version}"

echo "=========================================="
echo "后台运行训练任务"
echo "=========================================="
echo "任务: ${task}"
echo "数据集: ${dataset}"
echo "GPU ID: ${gpu_id}"
echo "Lever LM: ${lever_lm}"
echo "采样器: ${sampler}"
echo "模型: ${beam_model}"
echo "版本: ${version}"
echo "Conda 环境: ${conda_env}"
echo "日志文件: ${log_file}"
echo "=========================================="

# 使用 nohup 后台运行，并将输出重定向到日志文件
nohup bash -c "${cmd}" > "$log_file" 2>&1 &

# 获取进程 ID
pid=$!

# 保存 PID 到文件（可选，用于后续管理）
pid_file="${log_dir}/train_${task}_${dataset}_${gpu_id}_${sampler}_${beam_model}_${version}_${timestamp}.pid"
echo $pid > "$pid_file"

echo "✅ 训练任务已在后台启动"
echo "进程 ID: $pid"
echo "PID 文件: $pid_file"
echo "日志文件: $log_file"
echo ""
echo "查看日志: tail -f ${log_file}"
echo "查看进程: ps -p $pid"
echo "停止任务: kill $pid"

