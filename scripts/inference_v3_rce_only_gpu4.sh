#!/bin/bash
# 使用RCE-only模型（不进行GRPO）进行推理
# 使用方法: bash scripts/inference_v3_rce_only_gpu4.sh [test_data_num]

test_data_num=${1:-200}  # 默认200条数据，设置为-1表示使用全部数据

# 数据集和模型配置
dataset="okvqa_local"
dataset_name="okvqa"
task="vqa"
lever_lm="query_img_text_icd_img_text"
sampler="rand_sampler"
sampler_name="RandSampler"
beam_model="qwen2.5_vl_3B"
version="v3"
gpu_id=4

# Checkpoint路径（使用RCE阶段的模型，不进行GRPO）
checkpoint_dir="./results/okvqa/model_cpk/v3_RandSampler_v4"
checkpoint_path=""

echo "=========================================="
echo "使用RCE-only模型进行推理（GPU 4）"
echo "=========================================="
echo "GPU ID: ${gpu_id}"
echo "Sampler: ${sampler} → ${sampler_name}"
echo "Beam Model: ${beam_model}"
echo "Test Data Num: ${test_data_num}"
echo "Checkpoint Dir: ${checkpoint_dir}"
echo "=========================================="

# 查找最新的RCE checkpoint
if ls ${checkpoint_dir}/rce_epoch*.pt 1> /dev/null 2>&1; then
    checkpoint_path=$(ls -t ${checkpoint_dir}/rce_epoch*.pt | head -1)
    echo "✓ 找到RCE checkpoint: $(basename ${checkpoint_path})"
else
    echo "错误: 未找到RCE checkpoint文件"
    echo "查找目录: ${checkpoint_dir}"
    echo "查找模式: rce_epoch*.pt"
    exit 1
fi

echo "使用 checkpoint: ${checkpoint_path}"
echo "=========================================="

# 检查checkpoint是否存在
if [ ! -f "${checkpoint_path}" ]; then
    echo "错误: Checkpoint文件不存在: ${checkpoint_path}"
    exit 1
fi

# 设置环境变量
export LEVER_LM_CHECKPOINT_PATH="${checkpoint_path}"
export LEVER_LM_CHECKPOINT_VERSION="v3"

echo ""
echo "注意：这是RCE-only模型（未进行GRPO训练）"
echo "用于对比GRPO模型的效果"
echo "=========================================="
echo ""

# 运行推理
bash scripts/inference.sh ${task} ${dataset} ${gpu_id} ${lever_lm} ${sampler} ${beam_model} ${version} ${test_data_num}

