#!/bin/bash
# 使用最终合并的RL数据训练（包含all0激进补全）
# 使用方法: bash scripts/train_v3_final_merged.sh [GPU_ID]

set -e

GPU_ID=${1:-2}

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

# 加载.env文件
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
    echo "✓ 已加载.env文件"
fi

# 数据集配置
DATASET_NAME="okvqa"

# 文件路径 - 使用最终合并的数据
MERGED_RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_RandSampler_v4_strictEval_final_merged.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/candidate_embeddings.pt"

# V2 checkpoint
V2_CKPT_PATH="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
if [ ! -f "$V2_CKPT_PATH" ]; then
    V2_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2"
    if [ -d "$V2_DIR" ]; then
        V2_CKPT_PATH=$(find "$V2_DIR" -name "*RandSampler*.ckpt" -type f | head -1)
    fi
fi

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_final_merged_$(date +%Y%m%d)"

# 训练参数
RCE_EPOCHS=8
GRPO_EPOCHS=5
RCE_LR=5e-5
GRPO_LR=1e-5
KL_BETA=0.1
BATCH_SIZE=1

echo "=========================================="
echo "使用最终合并的RL数据训练"
echo "=========================================="
echo "GPU ID: ${GPU_ID}"
echo "RL数据: ${MERGED_RL_DATA}"
echo "V2 Checkpoint: ${V2_CKPT_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "训练参数："
echo "  RCE Epochs: ${RCE_EPOCHS}"
echo "  GRPO Epochs: ${GRPO_EPOCHS}"
echo "  RCE LR: ${RCE_LR}"
echo "  GRPO LR: ${GRPO_LR}"
echo "  KL Beta: ${KL_BETA}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Require Positive Query: true"
echo "=========================================="

# 检查文件
if [ ! -f "${MERGED_RL_DATA}" ]; then
    echo "错误: RL数据文件不存在: ${MERGED_RL_DATA}"
    exit 1
fi

if [ ! -f "${QUERY_EMB}" ]; then
    echo "错误: Query embeddings不存在: ${QUERY_EMB}"
    exit 1
fi

if [ ! -f "${V2_CKPT_PATH}" ]; then
    echo "错误: V2 checkpoint不存在: ${V2_CKPT_PATH}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行训练
echo ""
echo "开始训练..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "${MERGED_RL_DATA}" \
    --img_emb "${QUERY_EMB}" \
    --sft_ckpt "${V2_CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --rce_epochs ${RCE_EPOCHS} \
    --grpo_epochs ${GRPO_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --rce_lr ${RCE_LR} \
    --grpo_lr ${GRPO_LR} \
    --kl_beta ${KL_BETA} \
    --device cuda:0 \
    --require_positive_query

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
echo "输出目录: ${OUTPUT_DIR}"
