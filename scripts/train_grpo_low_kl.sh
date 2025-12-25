#!/bin/bash
# GRPO 训练 - 降低 KL beta 版本
# 目标：让模型有更大的探索空间

set -e

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 配置
KL_BETA=0.05  # 从 0.15 降低到 0.05
GRPO_LR=5e-6
GRPO_EPOCHS=3
OUTPUT_DIR="results/okvqa/model_cpk/v3_k64_grpo_lowkl"

# 基础 checkpoint（RCE epoch 2，与之前 GRPO 训练相同的起点）
RCE_CKPT="results/okvqa/model_cpk/v3_k64_scratch/rce_epoch2.pt"

# RL 数据
BEAM_DATA="results/okvqa/generated_data/rl_data_k64_v3.json"
IMG_EMB="results/okvqa/cache/query_embeddings.pt"

echo "========================================"
echo "GRPO 训练 - 低 KL beta 版本"
echo "========================================"
echo "KL beta: $KL_BETA (原来 0.15)"
echo "GRPO LR: $GRPO_LR"
echo "GRPO epochs: $GRPO_EPOCHS"
echo "基础 checkpoint: $RCE_CKPT"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# 运行训练
python -m lever_lm.workflows.grpo_post_train \
    --sft_ckpt "$RCE_CKPT" \
    --beam_data "$BEAM_DATA" \
    --img_emb "$IMG_EMB" \
    --output_dir "$OUTPUT_DIR" \
    --rce_epochs 0 \
    --grpo_epochs $GRPO_EPOCHS \
    --grpo_lr $GRPO_LR \
    --kl_beta $KL_BETA \
    --reward_mode hard_plus_soft \
    --freeze_backbone_in_grpo \
    --device cuda:0

echo "========================================"
echo "✓ GRPO 训练完成！"
echo "检查点保存在: $OUTPUT_DIR"
echo "========================================"
