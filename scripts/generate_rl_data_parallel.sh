#!/bin/bash
# 多 GPU 并行生成 RL 数据，支持断点续传
# 使用方法: bash scripts/generate_rl_data_parallel.sh <dataset> <total_queries> <num_gpus>
# 示例: bash scripts/generate_rl_data_parallel.sh vqav2_local 5000 6

dataset=${1:-vqav2_local}
total_queries=${2:-5000}
num_gpus=${3:-6}
sampler=${4:-rand_sampler}
beam_model=${5:-qwen2.5_vl_3B}

# 计算每个 GPU 处理的 query 数量
queries_per_gpu=$((total_queries / num_gpus))
remainder=$((total_queries % num_gpus))

echo "=========================================="
echo "多 GPU 并行生成 RL 数据"
echo "=========================================="
echo "数据集: ${dataset}"
echo "总 Query 数: ${total_queries}"
echo "GPU 数量: ${num_gpus}"
echo "每 GPU Query 数: ${queries_per_gpu} (余数: ${remainder})"
echo "Sampler: ${sampler}"
echo "Beam Model: ${beam_model}"
echo "=========================================="

# 将 sampler 转换为大驼峰格式
case "$sampler" in
    rand_sampler) sampler_name="RandSampler" ;;
    text_sim_sampler) sampler_name="TextSimSampler" ;;
    img_sim_sampler) sampler_name="ImgSimSampler" ;;
    mix_sampler) sampler_name="MixSampler" ;;
    *) sampler_name="${sampler}" ;;
esac

# 将 beam_model 映射到文件名
case "$beam_model" in
    flamingo_3B) model_name="flamingo_3B" ;;
    qwen2.5_vl_3B|qwen2_5_vl_3B) model_name="Qwen2_5-VL-3B-Instruct" ;;
    *) model_name=$(echo "$beam_model" | sed 's/\./_/g' | sed 's/\//_/g' | sed 's/ /_/g') ;;
esac

# 根据数据集设置参数
case "$dataset" in
    okvqa*|OKVQA*) dataset_name="okvqa" ;;
    vqav2*|VQAV2*) dataset_name="vqav2" ;;
    *) dataset_name="${dataset}" ;;
esac

# 输出目录
output_dir="./results/${dataset_name}/generated_data/rl_data_parallel"
mkdir -p "$output_dir"

# 启动每个 GPU 的任务
for gpu_id in $(seq 0 $((num_gpus - 1))); do
    start_idx=$((gpu_id * queries_per_gpu))
    
    # 最后一个 GPU 处理余数
    if [ $gpu_id -eq $((num_gpus - 1)) ]; then
        end_idx=$((start_idx + queries_per_gpu + remainder))
    else
        end_idx=$((start_idx + queries_per_gpu))
    fi
    
    output_file="${output_dir}/rl_data_gpu${gpu_id}_${start_idx}_${end_idx}.json"
    log_file="${output_dir}/gpu${gpu_id}.log"
    
    echo "GPU ${gpu_id}: queries ${start_idx}-${end_idx} -> ${output_file}"
    
    # 在后台启动任务
    nohup python -m lever_lm.models.v3.generate_rl_data_shard \
        --dataset "$dataset" \
        --sampler "$sampler" \
        --beam_model "$beam_model" \
        --gpu_id "$gpu_id" \
        --start_idx "$start_idx" \
        --end_idx "$end_idx" \
        --output_file "$output_file" \
        > "$log_file" 2>&1 &
    
    echo "  PID: $!"
    sleep 2  # 避免同时加载模型导致 OOM
done

echo "=========================================="
echo "所有任务已在后台启动！"
echo "查看进度: tail -f ${output_dir}/gpu*.log"
echo "合并结果: python scripts/merge_rl_data_shards.py --input_dir ${output_dir} --output results/${dataset_name}/generated_data/rl_data_${sampler_name}_${model_name}.json"
echo "=========================================="
