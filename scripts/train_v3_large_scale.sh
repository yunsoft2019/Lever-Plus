#!/bin/bash
# 使用大规模RL数据（rl_data_v4_large_scale.json）训练v3模型
#
# 使用方法：
#   bash scripts/train_v3_large_scale.sh [GPU_ID]
#
# 示例：
#   bash scripts/train_v3_large_scale.sh 4

set -e

# 参数
GPU_ID=${1:-4}

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa"

# 文件路径 - 使用大规模RL数据
RL_DATA="${PROJECT_ROOT}/results/${DATASET}/generated_data/rl_data_v4_large_scale.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET}/cache/query_embeddings.pt"
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt"
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET}/model_cpk/v3_large_scale"

# 训练参数（根据2025-12-13需求.md的建议）
RCE_EPOCHS=${RCE_EPOCHS:-5}
GRPO_EPOCHS=${GRPO_EPOCHS:-1}  # 先做RCE-only baseline，设为0；然后做轻量GRPO，设为1
BATCH_SIZE=${BATCH_SIZE:-1}
RCE_LR=${RCE_LR:-1e-4}
GRPO_LR=${GRPO_LR:-5e-6}  # 轻量GRPO
KL_BETA=${KL_BETA:-0.15}  # 稍强的KL约束
REWARD_MODE=${REWARD_MODE:-hard_plus_soft}  # 使用hard+soft reward模式
HARD_WEIGHT=${HARD_WEIGHT:-1.0}
SOFT_WEIGHT=${SOFT_WEIGHT:-1.0}

echo "=========================================="
echo "使用大规模RL数据训练v3模型"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "RL Data: ${RL_DATA}"
echo "SFT Checkpoint: ${SFT_CKPT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo ""
echo "训练参数："
echo "  - RCE Epochs: ${RCE_EPOCHS}"
echo "  - GRPO Epochs: ${GRPO_EPOCHS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - RCE LR: ${RCE_LR}"
echo "  - GRPO LR: ${GRPO_LR}"
echo "  - KL Beta: ${KL_BETA}"
echo "  - Reward Mode: ${REWARD_MODE}"
echo "  - Hard Weight: ${HARD_WEIGHT}"
echo "  - Soft Weight: ${SOFT_WEIGHT}"
echo "=========================================="

# 检查RL数据是否存在
if [ ! -f "$RL_DATA" ]; then
    echo "错误: RL数据文件不存在: $RL_DATA"
    exit 1
fi

# 检查SFT checkpoint
if [ ! -f "$SFT_CKPT" ]; then
    echo "警告: SFT checkpoint 不存在: $SFT_CKPT"
    echo "尝试查找其他可用的 checkpoint..."
    FOUND_CKPT=$(find "${PROJECT_ROOT}/results/${DATASET}/model_cpk/v2" -name "*RandSampler*.ckpt" -type f | head -1)
    if [ -n "$FOUND_CKPT" ]; then
        echo "找到: $FOUND_CKPT"
        SFT_CKPT="$FOUND_CKPT"
    else
        echo "未找到任何 v2 checkpoint，请先训练 v2 模型"
        exit 1
    fi
fi

echo ""
echo "开始训练..."
echo ""

cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "${RL_DATA}" \
    --img_emb "${QUERY_EMB}" \
    --sft_ckpt "${SFT_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --rce_epochs ${RCE_EPOCHS} \
    --grpo_epochs ${GRPO_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --rce_lr ${RCE_LR} \
    --grpo_lr ${GRPO_LR} \
    --kl_beta ${KL_BETA} \
    --reward_mode ${REWARD_MODE} \
    --hard_weight ${HARD_WEIGHT} \
    --soft_weight ${SOFT_WEIGHT} \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ v3训练完成！"
echo "Checkpoint保存在: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "下一步：进行推理测试"
echo "  python -m lever_lm.workflows.evaluate_v3 --beam_data ${RL_DATA} --test_num 200"

