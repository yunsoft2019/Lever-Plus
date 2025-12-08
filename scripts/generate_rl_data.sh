#!/bin/bash
# 生成 RL 数据（包含 beam、温度采样、随机组合和 correctness）
# 使用方法: bash scripts/generate_rl_data.sh <sft_ckpt> <beam_data> <output_path> <query_emb> <cand_emb> <device> [vqa_model] [dataset]

sft_ckpt=${1:-"results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=19_train=24.18280_val=21.98483.ckpt"}
beam_data=${2:-"results/okvqa/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json"}
output_path=${3:-"results/okvqa/generated_data/rl_data_RandSampler.json"}
query_emb=${4:-"results/okvqa/cache/query_embeddings.pt"}
cand_emb=${5:-"results/okvqa/cache/candidate_embeddings.pt"}
device=${6:-"cuda:0"}
vqa_model=${7:-"qwen2.5_vl_3B"}
dataset=${8:-"okvqa_local"}

# 创建输出目录
output_dir=$(dirname "$output_path")
mkdir -p "$output_dir"

echo "=========================================="
echo "生成 RL 数据"
echo "=========================================="
echo "SFT Checkpoint: $sft_ckpt"
echo "Beam Data: $beam_data"
echo "Output Path: $output_path"
echo "Query Embeddings: $query_emb"
echo "Candidate Embeddings: $cand_emb"
echo "Device: $device"
echo "VQA Model: $vqa_model"
echo "Dataset: $dataset"
echo "=========================================="

# 执行命令
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt "$sft_ckpt" \
    --beam_data "$beam_data" \
    --output_path "$output_path" \
    --query_emb "$query_emb" \
    --cand_emb "$cand_emb" \
    --device "$device" \
    --vqa_model "$vqa_model" \
    --dataset "$dataset"

echo "=========================================="
echo "✓ RL 数据生成完成！"
echo "输出文件: $output_path"
echo "=========================================="
