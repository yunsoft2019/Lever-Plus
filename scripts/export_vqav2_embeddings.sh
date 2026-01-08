#!/bin/bash
# 为 VQAv2 数据集导出 query 和 candidate embeddings
# 使用 OKVQA 训练的 Qwen2.5-VL-3B RandSampler v2 checkpoint
#
# 使用方法: bash scripts/export_vqav2_embeddings.sh <gpu_id>
# 示例: bash scripts/export_vqav2_embeddings.sh 4

set -e

gpu_id=${1:-4}

# 使用 OKVQA 的 Qwen2.5-VL-3B RandSampler v2 checkpoint
# 选择验证集表现最好的 checkpoint
sft_ckpt="results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt"

# VQAv2 数据集
dataset="vqav2_local"

# 输出目录
output_dir="results/vqav2/cache"

# 设备
device="cuda:${gpu_id}"

echo "=========================================="
echo "为 VQAv2 导出 Embeddings"
echo "=========================================="
echo "GPU ID: ${gpu_id}"
echo "SFT Checkpoint: ${sft_ckpt}"
echo "Dataset: ${dataset}"
echo "Output Directory: ${output_dir}"
echo "Device: ${device}"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -f "${sft_ckpt}" ]; then
    echo "错误: SFT checkpoint 不存在: ${sft_ckpt}"
    exit 1
fi

# 创建输出目录
mkdir -p "${output_dir}"

# 执行导出
CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.models.v3.export_embeddings \
    --sft_ckpt "${sft_ckpt}" \
    --dataset "${dataset}" \
    --output_dir "${output_dir}" \
    --device "cuda:0" \
    --batch_size 32

echo "=========================================="
echo "✓ VQAv2 Embeddings 导出完成！"
echo "输出文件:"
echo "  - ${output_dir}/query_embeddings.pt"
echo "  - ${output_dir}/candidate_embeddings.pt"
echo "=========================================="
