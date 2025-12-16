#!/bin/bash
# 使用最新RL数据v4训练v3模型（严格一致+可复用）
#
# 使用方法：
#   bash scripts/train_v3_with_rl_data_v4.sh [GPU_ID] [SAMPLER] [RL_DATA_SUFFIX] [REWARD_MODE]
#
# 参数说明:
#   GPU_ID: GPU设备ID（默认: 4）
#   SAMPLER: sampler名称（默认: RandSampler）
#   RL_DATA_SUFFIX: RL数据文件后缀（默认: v4_strictEval）
#   REWARD_MODE: reward模式（默认: hard_plus_gtprob_plus_rel）
#     - hard_plus_gtprob_plus_rel: 使用hard + gt_prob + relevance（推荐）
#     - hard_plus_gtprob: 使用hard + gt_prob
#     - hard_plus_soft: 使用hard + acc_score
#     - separated: 阈值分离模式
#
# 示例：
#   bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler v4_strictEval hard_plus_gtprob_plus_rel
#   bash scripts/train_v3_with_rl_data_v4.sh 4 RandSampler v4_strictEval hard_plus_gtprob

set -e

# ========== 参数配置 ==========
GPU_ID=${1:-4}
SAMPLER=${2:-RandSampler}
RL_DATA_SUFFIX=${3:-v4_strictEval}
REWARD_MODE=${4:-hard_plus_soft}  # 默认使用hard + vqa_acc_score

# ========== 路径配置 ==========
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa_local"
DATASET_NAME="okvqa"

# RL数据路径（v4格式）
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_${SAMPLER}_${RL_DATA_SUFFIX}.json"

# Embeddings路径
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/candidate_embeddings.pt"

# SFT模型checkpoint（v2格式）
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_${SAMPLER}_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_${SAMPLER}_v4"

# ========== 训练参数配置 ==========
# 可以根据需要调整这些参数
RCE_EPOCHS=${RCE_EPOCHS:-5}
GRPO_EPOCHS=${GRPO_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
RCE_LR=${RCE_LR:-1e-4}  # 修改：从1e-5改为1e-4，与V2一致
GRPO_LR=${GRPO_LR:-1e-5}
KL_BETA=${KL_BETA:-0.1}

# Reward权重（使用vqa_acc_score）
HARD_WEIGHT=${HARD_WEIGHT:-1.0}      # hard correctness权重
SOFT_WEIGHT=${SOFT_WEIGHT:-1.0}      # vqa_acc_score权重
REL_WEIGHT=${REL_WEIGHT:-0.1}        # relevance权重（hard_plus_soft模式不使用）

# ========== 显示配置信息 ==========
echo "=========================================="
echo "使用RL数据v4训练v3模型"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Sampler: ${SAMPLER}"
echo "RL Data: ${RL_DATA}"
echo "RL Data Suffix: ${RL_DATA_SUFFIX}"
echo "SFT Checkpoint: ${SFT_CKPT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo ""
echo "训练参数："
echo "  - RCE Epochs: ${RCE_EPOCHS}"
echo "  - GRPO Epochs: ${GRPO_EPOCHS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - RCE LR: ${RCE_LR} (已修改：与V2一致)"
echo "  - GRPO LR: ${GRPO_LR}"
echo "  - KL Beta: ${KL_BETA}"
echo ""
echo "模型配置（已修复）："
echo "  - label_smoothing: 0.0 (与V2一致)"
echo "  - dropout: 0.5 (与V2一致)"
echo ""
echo "Reward配置："
echo "  - Reward Mode: ${REWARD_MODE} (使用vqa_acc_score)"
echo "  - Hard Weight: ${HARD_WEIGHT}"
echo "  - Soft Weight: ${SOFT_WEIGHT} (vqa_acc_score权重)"
if [ "$REWARD_MODE" == "hard_plus_gtprob_plus_rel" ]; then
    echo "  - Rel Weight: ${REL_WEIGHT}"
fi
echo ""
echo "Reward公式：reward = ${HARD_WEIGHT} * vqa_correct + ${SOFT_WEIGHT} * vqa_acc_score"
echo "=========================================="

# ========== 检查文件是否存在 ==========
echo ""
echo "检查必需文件..."

if [ ! -f "$RL_DATA" ]; then
    echo "错误: RL数据文件不存在: $RL_DATA"
    echo ""
    echo "请先运行RL数据生成脚本："
    echo "  bash scripts/generate_rl_data_v4.sh ${GPU_ID} ${SAMPLER} ${RL_DATA_SUFFIX}"
    exit 1
fi

if [ ! -f "$QUERY_EMB" ]; then
    echo "错误: Query embeddings 不存在: $QUERY_EMB"
    exit 1
fi

if [ ! -f "$CAND_EMB" ]; then
    echo "错误: Candidate embeddings 不存在: $CAND_EMB"
    exit 1
fi

if [ ! -f "$SFT_CKPT" ]; then
    echo "警告: SFT checkpoint 不存在: $SFT_CKPT"
    echo "尝试查找其他可用的 checkpoint..."
    FOUND_CKPT=$(find "${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2" -name "*${SAMPLER}*.ckpt" -type f | head -1)
    if [ -n "$FOUND_CKPT" ]; then
        echo "找到: $FOUND_CKPT"
        SFT_CKPT="$FOUND_CKPT"
    else
        echo "未找到任何 v2 checkpoint，请先训练 v2 模型"
        exit 1
    fi
fi

echo "✓ 所有必需文件检查通过"
echo ""

# ========== 创建输出目录 ==========
mkdir -p "${OUTPUT_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# ========== 执行训练 ==========
echo "开始训练..."
echo ""

cd "${PROJECT_ROOT}"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 激活conda环境（如果需要）
if [ -f "/mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh" ]; then
    source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
    conda activate lever_env 2>/dev/null || true
fi

# 构建训练命令
TRAIN_CMD="python -m lever_lm.workflows.grpo_post_train \
    --beam_data \"${RL_DATA}\" \
    --img_emb \"${QUERY_EMB}\" \
    --sft_ckpt \"${SFT_CKPT}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --rce_epochs ${RCE_EPOCHS} \
    --grpo_epochs ${GRPO_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --rce_lr ${RCE_LR} \
    --grpo_lr ${GRPO_LR} \
    --kl_beta ${KL_BETA} \
    --reward_mode ${REWARD_MODE} \
    --hard_weight ${HARD_WEIGHT} \
    --soft_weight ${SOFT_WEIGHT} \
    --rel_weight ${REL_WEIGHT} \
    --device cuda:0"

# 默认启用skip_fallback_reward（使用strict_eval生成的数据应该都是vqaEval）
# 注意：grpo_post_train默认已启用skip_fallback_reward，无需额外参数
# 如果需要禁用，可以使用 --no_skip_fallback_reward

# 执行训练命令
eval ${TRAIN_CMD}

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ v3训练完成！"
    echo "=========================================="
    echo "Checkpoint保存在: ${OUTPUT_DIR}"
    echo ""
    echo "训练配置："
    echo "  - RL数据: ${RL_DATA}"
    echo "  - Reward模式: ${REWARD_MODE}"
    echo "  - Hard Weight: ${HARD_WEIGHT}"
    echo "  - Soft Weight: ${SOFT_WEIGHT}"
    if [ "$REWARD_MODE" == "hard_plus_gtprob_plus_rel" ]; then
        echo "  - Rel Weight: ${REL_WEIGHT}"
    fi
    echo ""
    echo "下一步：进行推理测试"
    echo "  bash scripts/inference_v3_best.sh 200 ${SAMPLER} qwen2.5_vl_3B"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败！"
    echo "=========================================="
    echo "请检查日志文件或错误信息"
    exit 1
fi

