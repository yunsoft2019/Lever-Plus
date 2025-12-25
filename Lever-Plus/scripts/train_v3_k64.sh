#!/bin/bash
# 使用修复后的K=64候选池RL数据训练v3模型
#
# 关键特点：
#   - 每个query使用独立的64个候选池（candidate_pool_ids字段）
#   - 轨迹使用局部索引 [0, 63]
#   - 与SFT阶段的候选池大小一致（K=64）
#
# 使用方法：
#   bash scripts/train_v3_k64.sh [GPU_ID]
#
# 示例：
#   bash scripts/train_v3_k64.sh 4
#   bash scripts/train_v3_k64.sh 5

set -e

# ========== 参数配置 ==========
GPU_ID=${1:-4}

# ========== 路径配置 ==========
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"
SAMPLER="RandSampler"

# RL数据路径（修复后的K=64数据）
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3.json"

# Embeddings路径
# 注意：使用完整的img_emb_data，因为每个query的候选池可能包含不同的ICD
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"

# SFT模型checkpoint（v2格式）
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_${SAMPLER}_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"

# 如果best.ckpt不存在，尝试查找其他checkpoint
if [ ! -f "$SFT_CKPT" ]; then
    SFT_CKPT=$(find "${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2" -name "*${SAMPLER}*.ckpt" -type f | head -1)
fi

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_k64"

# ========== 训练参数配置 ==========
# RCE阶段（预热）
RCE_EPOCHS=${RCE_EPOCHS:-5}
RCE_LR=${RCE_LR:-1e-4}

# GRPO阶段（强化学习）
GRPO_EPOCHS=${GRPO_EPOCHS:-10}
GRPO_LR=${GRPO_LR:-1e-5}

# 其他参数
BATCH_SIZE=${BATCH_SIZE:-1}
KL_BETA=${KL_BETA:-0.1}

# Reward配置
REWARD_MODE=${REWARD_MODE:-hard_plus_soft}
HARD_WEIGHT=${HARD_WEIGHT:-1.0}
SOFT_WEIGHT=${SOFT_WEIGHT:-1.0}
REL_WEIGHT=${REL_WEIGHT:-0.1}

# ========== 显示配置信息 ==========
echo "=========================================="
echo "使用修复后的K=64候选池RL数据训练v3模型"
echo "=========================================="
echo ""
echo "【关键修复】"
echo "  - 每个query使用独立的64个候选池（candidate_pool_ids字段）"
echo "  - 轨迹使用局部索引 [0, 63]"
echo "  - 与SFT阶段的候选池大小一致（K=64）"
echo ""
echo "配置信息："
echo "  GPU: ${GPU_ID}"
echo "  RL Data: ${RL_DATA}"
echo "  SFT Checkpoint: ${SFT_CKPT}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""
echo "训练参数："
echo "  - RCE Epochs: ${RCE_EPOCHS}"
echo "  - RCE LR: ${RCE_LR}"
echo "  - GRPO Epochs: ${GRPO_EPOCHS}"
echo "  - GRPO LR: ${GRPO_LR}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - KL Beta: ${KL_BETA}"
echo ""
echo "Reward配置："
echo "  - Reward Mode: ${REWARD_MODE}"
echo "  - Hard Weight: ${HARD_WEIGHT}"
echo "  - Soft Weight: ${SOFT_WEIGHT}"
echo "=========================================="

# ========== 检查文件是否存在 ==========
echo ""
echo "检查必需文件..."

if [ ! -f "$RL_DATA" ]; then
    echo "错误: RL数据文件不存在: $RL_DATA"
    echo ""
    echo "请先运行修复脚本："
    echo "  python scripts/fix_rl_candidate_pool_k64_v3.py \\"
    echo "      --input_path results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_final_merged.json \\"
    echo "      --sampler_cache_path \"results/okvqa/cache/okvqa-RandSampler-anchor_sample_num: 800:64.json\" \\"
    echo "      --output_path results/okvqa/generated_data/rl_data_k64_v3.json"
    exit 1
fi
echo "  ✓ RL数据: ${RL_DATA}"

if [ ! -f "$QUERY_EMB" ]; then
    echo "错误: Query embeddings 不存在: $QUERY_EMB"
    exit 1
fi
echo "  ✓ Query embeddings: ${QUERY_EMB}"

if [ ! -f "$SFT_CKPT" ]; then
    echo "错误: SFT checkpoint 不存在"
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
echo "  ✓ SFT checkpoint: ${SFT_CKPT}"

echo ""
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

# 激活conda环境
if [ -f "/mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh" ]; then
    source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
    conda activate lever_env 2>/dev/null || true
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

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

echo "执行命令："
echo "${TRAIN_CMD}"
echo ""

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
