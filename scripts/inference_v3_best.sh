#!/bin/bash
# 使用最佳 v3 checkpoint 进行推理
# 使用方法: bash scripts/inference_v3_best.sh [test_data_num] [sampler] [beam_model]
# 示例: bash scripts/inference_v3_best.sh 400 rand_sampler qwen2.5_vl_3B

test_data_num=${1:-400}  # 默认 400 条数据，设置为 -1 表示使用全部数据
sampler=${2:-rand_sampler}
beam_model=${3:-qwen2.5_vl_3B}
gpu_id=${4:-7}  # 默认使用 7 号 GPU，可以通过第4个参数指定

# 将 sampler 转换为大驼峰格式
case "$sampler" in
    rand_sampler)
        sampler_name="RandSampler"
        ;;
    text_sim_sampler)
        sampler_name="TextSimSampler"
        ;;
    img_sim_sampler)
        sampler_name="ImgSimSampler"
        ;;
    mix_sampler)
        sampler_name="MixSampler"
        ;;
    *)
        sampler_name="${sampler}"
        ;;
esac

# 根据数据集设置
dataset="okvqa_local"
dataset_name="okvqa"
task="vqa"
lever_lm="query_img_text_icd_img_text"
version="v3"

# 构建检查点目录
checkpoint_dir="./results/${dataset_name}/model_cpk/v3"

echo "=========================================="
echo "使用最佳 v3 checkpoint 进行推理"
echo "=========================================="
echo "GPU ID: ${gpu_id}"
echo "Sampler: ${sampler} → ${sampler_name}"
echo "Beam Model: ${beam_model}"
echo "Test Data Num: ${test_data_num}"
echo "Checkpoint Dir: ${checkpoint_dir}"
echo "=========================================="

# 查找最佳的 checkpoint（优先 GRPO，其次 RCE）
best_checkpoint=""

# 1. 优先查找最新的 grpo_epoch*.pt（GRPO checkpoint）
if ls ${checkpoint_dir}/grpo_epoch*.pt 1> /dev/null 2>&1; then
    best_checkpoint=$(ls -t ${checkpoint_dir}/grpo_epoch*.pt | head -1)
    echo "✓ 找到最新的 GRPO checkpoint: $(basename ${best_checkpoint})"
# 2. 如果没找到 GRPO，查找最新的 rce_epoch*.pt（RCE checkpoint）
elif ls ${checkpoint_dir}/rce_epoch*.pt 1> /dev/null 2>&1; then
    best_checkpoint=$(ls -t ${checkpoint_dir}/rce_epoch*.pt | head -1)
    echo "✓ 找到最新的 RCE checkpoint: $(basename ${best_checkpoint})"
    echo "  注意：这是只使用 RCE 训练的模型（未进行 GRPO）"
# 3. 如果还是没找到，查找所有 .pt 文件
elif ls ${checkpoint_dir}/*.pt 1> /dev/null 2>&1; then
    best_checkpoint=$(ls -t ${checkpoint_dir}/*.pt | head -1)
    echo "✓ 找到 checkpoint: $(basename ${best_checkpoint})"
else
    echo "错误: 未找到 v3 checkpoint 文件"
    echo "查找目录: ${checkpoint_dir}"
    exit 1
fi

echo "使用 checkpoint: ${best_checkpoint}"
echo "=========================================="

# 设置环境变量，指定使用这个 checkpoint
export LEVER_LM_CHECKPOINT_PATH="${best_checkpoint}"
export LEVER_LM_CHECKPOINT_VERSION="v3"

# 调用 inference.sh，传入所有参数
bash scripts/inference.sh ${task} ${dataset} ${gpu_id} ${lever_lm} ${sampler} ${beam_model} ${version} ${test_data_num}
