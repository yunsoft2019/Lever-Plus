#!/bin/bash
# VQAv2 单 GPU 分片生成 RL 数据
# 
# 使用方法: 
#   bash scripts/generate_rl_data_vqav2_shard.sh <gpu_id> <start_idx> <end_idx>
#
# 示例:
#   bash scripts/generate_rl_data_vqav2_shard.sh 0 0 1000      # GPU 0 处理 0-1000
#   bash scripts/generate_rl_data_vqav2_shard.sh 1 1000 2000   # GPU 1 处理 1000-2000
#   bash scripts/generate_rl_data_vqav2_shard.sh 2 2000 3000   # GPU 2 处理 2000-3000
#
# 支持断点续传：如果中断，重新运行相同命令会跳过已完成的 query

set -e

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: bash scripts/generate_rl_data_vqav2_shard.sh <gpu_id> <start_idx> <end_idx>"
    echo ""
    echo "示例:"
    echo "  bash scripts/generate_rl_data_vqav2_shard.sh 0 0 1000"
    echo "  bash scripts/generate_rl_data_vqav2_shard.sh 1 1000 2000"
    echo ""
    echo "VQAv2 共 5000 个 query，建议分配:"
    echo "  GPU 0: 0-833"
    echo "  GPU 1: 833-1666"
    echo "  GPU 2: 1666-2499"
    echo "  GPU 3: 2499-3332"
    echo "  GPU 4: 3332-4165"
    echo "  GPU 5: 4165-5000"
    exit 1
fi

gpu_id=$1
start_idx=$2
end_idx=$3

dataset="vqav2_local"
sampler_name="RandSampler"
model_name="Qwen2_5-VL-3B-Instruct"
dataset_name="vqav2"
beam_model="qwen2.5_vl_3B"

echo "=========================================="
echo "VQAv2 RL 数据生成 (分片模式)"
echo "=========================================="
echo "GPU: ${gpu_id}"
echo "范围: ${start_idx} - ${end_idx}"
echo "=========================================="

# 创建输出目录
output_dir="./results/${dataset_name}/generated_data/rl_data_shards"
mkdir -p "$output_dir"

# 查找 SFT checkpoint
sft_ckpt=$(find ./results/${dataset_name}/model_cpk/v2 -name "*${sampler_name}*.ckpt" -type f | head -1)
if [ -z "$sft_ckpt" ]; then
    echo "错误: 未找到 SFT checkpoint"
    exit 1
fi
echo "SFT Checkpoint: ${sft_ckpt}"

# Beam 数据路径
beam_data="./results/${dataset_name}/generated_data/vqa-${dataset_name}-${model_name}-${sampler_name}-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:5000.json"
query_emb="./results/${dataset_name}/cache/query_embeddings.pt"
cand_emb="./results/${dataset_name}/cache/candidate_embeddings.pt"

# VQA 标注文件
val_ques="./datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
val_ann="./datasets/vqav2/v2_mscoco_val2014_annotations.json"
train_ques="./datasets/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
train_ann="./datasets/vqav2/v2_mscoco_train2014_annotations.json"

# 输出文件
output_file="${output_dir}/shard_${start_idx}_${end_idx}.json"

echo "输出文件: ${output_file}"
echo "=========================================="

# 运行
CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt "$sft_ckpt" \
    --beam_data "$beam_data" \
    --output_path "$output_file" \
    --query_emb "$query_emb" \
    --cand_emb "$cand_emb" \
    --device "cuda:0" \
    --vqa_model "$beam_model" \
    --dataset "$dataset" \
    --train_ques_path "$train_ques" \
    --train_ann_path "$train_ann" \
    --val_ques_path "$val_ques" \
    --val_ann_path "$val_ann" \
    --start_idx "$start_idx" \
    --end_idx "$end_idx"

echo "=========================================="
echo "✓ 完成！输出: ${output_file}"
echo "=========================================="
