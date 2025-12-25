#!/bin/bash
# 评估 GRPO Low KL beta 模型

set -e

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
CKPT_DIR="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_k64_grpo_lowkl"

echo "=========================================="
echo "转换 checkpoint 格式"
echo "=========================================="

for epoch in 1 2 3; do
    pt_file="${CKPT_DIR}/grpo_epoch${epoch}.pt"
    v2_file="${CKPT_DIR}/grpo_epoch${epoch}_v2format.ckpt"
    
    if [ -f "$pt_file" ] && [ ! -f "$v2_file" ]; then
        echo "转换 grpo_epoch${epoch}.pt -> grpo_epoch${epoch}_v2format.ckpt ..."
        python scripts/convert_v3_to_v2_format.py --v3_ckpt "$pt_file" --output "$v2_file"
    fi
done

echo "=========================================="
echo "评估 GRPO Epoch 1 (Low KL beta=0.05)"
echo "=========================================="

# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH="${CKPT_DIR}/grpo_epoch1_v2format.ckpt"

# 并行评估不同样本数
for gpu_samples in "0 100" "1 200" "3 400" "4 800"; do
    read gpu samples <<< "$gpu_samples"
    echo "GPU $gpu: $samples 样本, shot 1-4"
    (
        bash scripts/inference.sh vqa okvqa_local $gpu query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 $samples
    ) &
done

wait
echo "✓ 评估完成"
