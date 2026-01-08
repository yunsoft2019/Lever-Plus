#!/bin/bash
# VQAv2 多 GPU 并行生成 RL 数据
# 使用方法: bash scripts/generate_rl_data_vqav2_parallel.sh
# 
# 这个脚本会在 6 个 GPU 上并行生成 RL 数据，每个 GPU 处理不同的 query 范围
# 支持断点续传：如果中断，重新运行会跳过已完成的 query

set -e

dataset="vqav2_local"
sampler="rand_sampler"
beam_model="qwen2.5_vl_3B"
total_queries=5000
num_gpus=6

# 转换名称
sampler_name="RandSampler"
model_name="Qwen2_5-VL-3B-Instruct"
dataset_name="vqav2"

# 计算每个 GPU 的范围
queries_per_gpu=$((total_queries / num_gpus))  # 833
remainder=$((total_queries % num_gpus))  # 2

echo "=========================================="
echo "VQAv2 多 GPU 并行生成 RL 数据"
echo "=========================================="
echo "总 Query 数: ${total_queries}"
echo "GPU 数量: ${num_gpus}"
echo "每 GPU Query 数: ~${queries_per_gpu}"
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

echo "=========================================="
echo "启动 ${num_gpus} 个 GPU 任务..."
echo "=========================================="

# 启动每个 GPU 的任务
for gpu_id in $(seq 0 $((num_gpus - 1))); do
    start_idx=$((gpu_id * queries_per_gpu))
    
    # 最后一个 GPU 处理余数
    if [ $gpu_id -eq $((num_gpus - 1)) ]; then
        end_idx=$((start_idx + queries_per_gpu + remainder))
    else
        end_idx=$((start_idx + queries_per_gpu))
    fi
    
    output_file="${output_dir}/shard_gpu${gpu_id}.json"
    log_file="${output_dir}/gpu${gpu_id}.log"
    
    echo "GPU ${gpu_id}: queries ${start_idx}-${end_idx}"
    echo "  输出: ${output_file}"
    echo "  日志: ${log_file}"
    
    # 在后台启动任务
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -m lever_lm.models.v3.generate_rl_data \
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
        --end_idx "$end_idx" \
        > "$log_file" 2>&1 &
    
    pid=$!
    echo "  PID: $pid"
    echo "$pid" > "${output_dir}/gpu${gpu_id}.pid"
    
    sleep 5  # 避免同时加载模型导致 OOM
done

echo "=========================================="
echo "所有任务已在后台启动！"
echo ""
echo "查看进度:"
echo "  tail -f ${output_dir}/gpu*.log"
echo ""
echo "查看所有任务状态:"
echo "  for i in \$(seq 0 5); do echo \"GPU \$i:\"; tail -1 ${output_dir}/gpu\$i.log 2>/dev/null || echo '未启动'; done"
echo ""
echo "完成后合并结果:"
echo "  python scripts/merge_rl_data_shards.py"
echo "=========================================="
