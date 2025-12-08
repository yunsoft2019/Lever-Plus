#!/bin/bash
# 导出 query 和 candidate embeddings 用于 RL 数据生成
# 使用方法: bash scripts/export_embeddings.sh <sft_ckpt> <dataset> <output_dir> <device> [train_config]

sft_ckpt=${1:-"results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=19_train=24.18280_val=21.98483.ckpt"}
dataset=${2:-"okvqa_local"}
output_dir=${3:-"results/okvqa/cache"}
device=${4:-"cuda:0"}
train_config=${5:-""}

# 创建输出目录
mkdir -p "$output_dir"

# 构建命令
cmd="python -m lever_lm.models.v3.export_embeddings \
    --sft_ckpt \"$sft_ckpt\" \
    --dataset \"$dataset\" \
    --output_dir \"$output_dir\" \
    --device \"$device\" \
    --batch_size 32"

# 如果提供了 train_config，添加到命令中
if [ -n "$train_config" ]; then
    cmd="$cmd --train_config \"$train_config\""
fi

echo "=========================================="
echo "导出 Embeddings"
echo "=========================================="
echo "SFT Checkpoint: $sft_ckpt"
echo "Dataset: $dataset"
echo "Output Directory: $output_dir"
echo "Device: $device"
echo "=========================================="

# 执行命令
eval $cmd

echo "=========================================="
echo "✓ Embeddings 导出完成！"
echo "输出文件:"
echo "  - $output_dir/query_embeddings.pt"
echo "  - $output_dir/candidate_embeddings.pt"
echo "=========================================="
