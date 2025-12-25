#!/bin/bash
# 使用 KL_BETA=0.12 从 v2 baseline 开始训练 GRPO
# 
# 与 v3_k64_grpo 的区别：
#   - 基础模型：v2 baseline（而不是 v3_k64_scratch/rce_epoch2.pt）
#   - KL_BETA: 0.12（而不是 0.15）

set -e

GPU_ID=${1:-4}
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# 数据路径
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"

# 从 v2 baseline 开始（与 2025-12-12 正确率报告中的 baseline 一致）
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_k64_grpo_kl012"

# 训练参数
RCE_EPOCHS=2          # 先做 2 个 epoch RCE 预热
GRPO_EPOCHS=3         # 然后 3 个 epoch GRPO
GRPO_LR=5e-6
KL_BETA=0.12          # 使用 0.12 的 KL beta

echo "=========================================="
echo "从 v2 baseline 开始，使用 KL_BETA=0.12 训练 GRPO"
echo "=========================================="
echo "  - SFT checkpoint: ${SFT_CKPT}"
echo "  - RCE epochs: ${RCE_EPOCHS}"
echo "  - GRPO epochs: ${GRPO_EPOCHS}"
echo "  - GRPO LR: ${GRPO_LR}"
echo "  - KL beta: ${KL_BETA}"
echo "  - 冻结 backbone: 是"
echo "=========================================="

# 检查文件
if [ ! -f "$RL_DATA" ]; then
    echo "错误: RL数据文件不存在: $RL_DATA"
    exit 1
fi

if [ ! -f "$SFT_CKPT" ]; then
    echo "错误: SFT checkpoint 不存在: $SFT_CKPT"
    # 尝试查找其他 v2 checkpoint
    FOUND_CKPT=$(find "${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2" -name "*RandSampler*.ckpt" -type f | head -1)
    if [ -n "$FOUND_CKPT" ]; then
        echo "找到替代: $FOUND_CKPT"
        SFT_CKPT="$FOUND_CKPT"
    else
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python -m lever_lm.workflows.grpo_post_train \
    --beam_data "${RL_DATA}" \
    --img_emb "${QUERY_EMB}" \
    --sft_ckpt "${SFT_CKPT}" \
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

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "  Checkpoint: ${OUTPUT_DIR}"
echo "=========================================="
