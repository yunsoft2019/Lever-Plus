#!/bin/bash
# 在 RCE 基础上加 GRPO 训练
# 从最佳 RCE checkpoint (epoch 2) 开始，进行少量 GRPO 训练

set -e

GPU_ID=${1:-4}
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# 数据路径
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"

# 从最佳 RCE checkpoint 开始（epoch 2，val_loss 最低）
RCE_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_k64_scratch/rce_epoch2.pt"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_k64_grpo"

# GRPO 训练参数（保守设置）
GRPO_EPOCHS=${GRPO_EPOCHS:-3}
GRPO_LR=${GRPO_LR:-5e-6}
KL_BETA=${KL_BETA:-1.0}  # 使用较大的 KL Beta 来约束更新

# RCE epochs 设为 0，因为我们直接加载已训练好的 RCE checkpoint
RCE_EPOCHS=0

echo "=========================================="
echo "在 RCE 基础上加 GRPO 训练"
echo "  - RCE checkpoint: ${RCE_CKPT}"
echo "  - GRPO epochs: ${GRPO_EPOCHS}"
echo "  - GRPO LR: ${GRPO_LR}"
echo "  - KL beta: ${KL_BETA}"
echo "  - 冻结 backbone: 是"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python -m lever_lm.workflows.grpo_post_train \
    --beam_data "${RL_DATA}" \
    --img_emb "${QUERY_EMB}" \
    --sft_ckpt "${RCE_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --rce_epochs ${RCE_EPOCHS} \
    --grpo_epochs ${GRPO_EPOCHS} \
    --grpo_lr ${GRPO_LR} \
    --kl_beta ${KL_BETA} \
    --disable_adaptive_kl \
    --batch_size 1 \
    --reward_mode hard_plus_soft \
    --freeze_backbone_in_grpo \
    --device cuda:0

echo "✓ GRPO 训练完成！Checkpoint: ${OUTPUT_DIR}"
