#!/bin/bash
# 从头训练 v3 模型（不加载 SFT checkpoint）
#
# 关键改动：
#   - 不加载 SFT checkpoint，模型从随机初始化开始
#   - RCE 阶段会根据 reward 学习区分正负样本
#   - 增加 RCE epochs（因为从头学习需要更多训练）
#
# 使用方法：
#   bash scripts/train_v3_k64_scratch.sh [GPU_ID]

set -e

# ========== 参数配置 ==========
GPU_ID=${1:-4}

# ========== 路径配置 ==========
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# RL数据路径（使用 v3 修复后的数据）
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3.json"

# Embeddings路径
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_k64_scratch"

# ========== 训练参数配置 ==========
# RCE阶段（从头训练需要更多 epochs）
RCE_EPOCHS=${RCE_EPOCHS:-20}
RCE_LR=${RCE_LR:-1e-3}

# RCE 温度调度（关键！初始温度太高会让负样本权重过大）
RCE_TEMP_START=${RCE_TEMP_START:-0.5}
RCE_TEMP_END=${RCE_TEMP_END:-0.1}

# GRPO阶段（可选，默认关闭）
GRPO_EPOCHS=${GRPO_EPOCHS:-0}
GRPO_LR=${GRPO_LR:-1e-5}

# 其他参数
BATCH_SIZE=${BATCH_SIZE:-1}
KL_BETA=${KL_BETA:-0.1}

# Reward配置
REWARD_MODE=${REWARD_MODE:-hard_plus_soft}
HARD_WEIGHT=${HARD_WEIGHT:-1.0}
SOFT_WEIGHT=${SOFT_WEIGHT:-1.0}

# ========== 显示配置信息 ==========
echo "=========================================="
echo "从头训练 v3 模型（不加载 SFT checkpoint）"
echo "=========================================="
echo ""
echo "【关键改动】"
echo "  - 不加载 SFT checkpoint，模型从随机初始化开始"
echo "  - RCE 阶段会根据 reward 学习区分正负样本"
echo "  - 增加 RCE epochs 和学习率"
echo ""
echo "配置信息："
echo "  GPU: ${GPU_ID}"
echo "  RL Data: ${RL_DATA}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""
echo "训练参数："
echo "  - RCE Epochs: ${RCE_EPOCHS}"
echo "  - RCE LR: ${RCE_LR}"
echo "  - RCE Temp: ${RCE_TEMP_START} -> ${RCE_TEMP_END}"
echo "  - GRPO Epochs: ${GRPO_EPOCHS}"
echo "  - Batch Size: ${BATCH_SIZE}"
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
    exit 1
fi
echo "  ✓ RL数据: ${RL_DATA}"

if [ ! -f "$QUERY_EMB" ]; then
    echo "错误: Query embeddings 不存在: $QUERY_EMB"
    exit 1
fi
echo "  ✓ Query embeddings: ${QUERY_EMB}"

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

# 构建训练命令（注意：不传 --sft_ckpt 参数）
TRAIN_CMD="python -m lever_lm.workflows.grpo_post_train \
    --beam_data \"${RL_DATA}\" \
    --img_emb \"${QUERY_EMB}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --rce_epochs ${RCE_EPOCHS} \
    --rce_lr ${RCE_LR} \
    --rce_temp_start ${RCE_TEMP_START} \
    --rce_temp_end ${RCE_TEMP_END} \
    --grpo_epochs ${GRPO_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --grpo_lr ${GRPO_LR} \
    --kl_beta ${KL_BETA} \
    --reward_mode ${REWARD_MODE} \
    --hard_weight ${HARD_WEIGHT} \
    --soft_weight ${SOFT_WEIGHT} \
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
    echo "✓ 从头训练完成！"
    echo "=========================================="
    echo "Checkpoint保存在: ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败！"
    echo "=========================================="
    exit 1
fi
