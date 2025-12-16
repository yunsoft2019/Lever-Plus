#!/bin/bash
# 使用GPU 4进行v3模型推理
# 使用方法: bash scripts/inference_v3_gpu4.sh [test_data_num]

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

# Checkpoint路径（使用最新训练的模型）
checkpoint_path="./results/okvqa/model_cpk/v3_RandSampler_v4/grpo_epoch10.pt"

echo "=========================================="
echo "使用GPU 4进行v3模型推理"
echo "=========================================="
echo "GPU ID: ${gpu_id}"
echo "Sampler: ${sampler} → ${sampler_name}"
echo "Beam Model: ${beam_model}"
echo "Test Data Num: ${test_data_num}"
echo "Checkpoint: ${checkpoint_path}"
echo "=========================================="

# 检查checkpoint是否存在
if [ ! -f "${checkpoint_path}" ]; then
    echo "错误: Checkpoint文件不存在: ${checkpoint_path}"
    exit 1
fi

# 设置环境变量
export LEVER_LM_CHECKPOINT_PATH="${checkpoint_path}"
export LEVER_LM_CHECKPOINT_VERSION="v3"

# 运行推理
bash scripts/inference.sh ${task} ${dataset} ${gpu_id} ${lever_lm} ${sampler} ${beam_model} ${version} ${test_data_num}

