#!/bin/bash
# 使用新生成的RL数据训练v3模型
#
# 使用方法：
#   bash scripts/train_v3_with_new_rl_data.sh [GPU_ID] [SAMPLER]
#
# 示例：
#   bash scripts/train_v3_with_new_rl_data.sh 0 rand_sampler

set -e

# 参数
GPU_ID=${1:-0}
SAMPLER=${2:-rand_sampler}

# 将 sampler 转换为大驼峰格式
case "$SAMPLER" in
    rand_sampler)
        SAMPLER_NAME="RandSampler"
        ;;
    text_sim_sampler)
        SAMPLER_NAME="TextSimSampler"
        ;;
    img_sim_sampler)
        SAMPLER_NAME="ImgSimSampler"
        ;;
    mix_sampler)
        SAMPLER_NAME="MixSampler"
        ;;
    *)
        SAMPLER_NAME="${SAMPLER}"
        ;;
esac

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa"

# 文件路径 - 使用新生成的RL数据（v2版本）
RL_DATA="${PROJECT_ROOT}/results/${DATASET}/generated_data/rl_data_${SAMPLER_NAME}_v2.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET}/cache/query_embeddings.pt"
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_${SAMPLER_NAME}_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET}/model_cpk/v3_new"

# 训练参数（根据强化学习.md的建议）
RCE_EPOCHS=${RCE_EPOCHS:-5}
GRPO_EPOCHS=${GRPO_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
RCE_LR=${RCE_LR:-1e-5}
GRPO_LR=${GRPO_LR:-1e-5}
KL_BETA=${KL_BETA:-0.1}
REWARD_ALPHA=${REWARD_ALPHA:-0.5}
REWARD_BETA=${REWARD_BETA:-0.8}

echo "=========================================="
echo "使用新RL数据训练v3模型"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Sampler: ${SAMPLER} -> ${SAMPLER_NAME}"
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
echo "  - Reward Alpha: ${REWARD_ALPHA}"
echo "  - Reward Beta: ${REWARD_BETA}"
echo "=========================================="

# 检查RL数据是否存在
if [ ! -f "$RL_DATA" ]; then
    echo "错误: RL数据文件不存在: $RL_DATA"
    echo ""
    echo "请先运行数据生成脚本："
    echo "  bash scripts/regenerate_rl_data.sh ${GPU_ID} ${SAMPLER}"
    exit 1
fi

# 检查SFT checkpoint
if [ ! -f "$SFT_CKPT" ]; then
    echo "警告: SFT checkpoint 不存在: $SFT_CKPT"
    echo "尝试查找其他可用的 checkpoint..."
    FOUND_CKPT=$(find "${PROJECT_ROOT}/results/${DATASET}/model_cpk/v2" -name "*${SAMPLER_NAME}*.ckpt" -type f | head -1)
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
    --reward_alpha ${REWARD_ALPHA} \
    --reward_beta ${REWARD_BETA} \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ v3训练完成！"
echo "Checkpoint保存在: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "下一步：进行推理测试"
echo "  bash scripts/inference_v3_best.sh 200 ${SAMPLER} qwen2.5_vl_3B"
