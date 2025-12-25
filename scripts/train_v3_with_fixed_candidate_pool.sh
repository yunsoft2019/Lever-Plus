#!/bin/bash
# 使用修复候选池后的RL数据重新训练checkpoint（方案1：限制候选池大小为64）
# 使用方法: bash scripts/train_v3_with_fixed_candidate_pool.sh [GPU_ID]

set -e

GPU_ID=${1:-7}

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

# 加载.env文件（如果存在）
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
    echo "✓ 已加载.env文件"
fi

# 数据集配置
DATASET="okvqa_local"
DATASET_NAME="okvqa"
SAMPLER="rand_sampler"
SAMPLER_NAME="RandSampler"
BEAM_MODEL="qwen2.5_vl_3B"
MODEL_NAME="Qwen2_5-VL-3B-Instruct"

# 文件路径（使用修复后的数据）
FIXED_RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_RandSampler_v4_strictEval_merged_fixed_candidate_pool.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/candidate_embeddings.pt"

# 查找v2 checkpoint
V2_CKPT_PATH="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_${SAMPLER_NAME}_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
if [ ! -f "$V2_CKPT_PATH" ]; then
    V2_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2"
    if [ -d "$V2_DIR" ]; then
        V2_CKPT_PATH=$(find "$V2_DIR" -name "*${SAMPLER_NAME}*.ckpt" -type f | head -1)
    fi
fi

# 输出目录（使用fixed_candidate_pool后缀）
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_fixed_candidate_pool"

# 优化后的训练参数（针对修复后的数据）
# 1. 增加RCE预热轮数（适应数据分布变化）
RCE_EPOCHS=${RCE_EPOCHS:-10}
# 2. 保持GRPO轮数
GRPO_EPOCHS=${GRPO_EPOCHS:-8}
# 3. 调整学习率（更保守的RCE，更激进的GRPO）
RCE_LR=${RCE_LR:-5e-5}
GRPO_LR=${GRPO_LR:-1e-5}
# 4. 调整KL约束（更宽松的约束，允许更多探索）
KL_BETA=${KL_BETA:-0.1}
# 5. 批次大小
BATCH_SIZE=${BATCH_SIZE:-1}
# 6. 只训练有正样本的query（提高数据质量）
REQUIRE_POSITIVE_QUERY=${REQUIRE_POSITIVE_QUERY:-true}

echo "=========================================="
echo "使用修复候选池后的RL数据重新训练checkpoint"
echo "=========================================="
echo "GPU ID: ${GPU_ID}"
echo "修复后的RL数据: ${FIXED_RL_DATA}"
echo "V2 Checkpoint: ${V2_CKPT_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "【关键修复】"
echo "  - 使用所有query的候选索引（确保所有pointer都在候选池中）"
echo "  - 不使用 --candidate_size 限制（会导致pointer索引不在候选池中）"
echo "  - 推理时也不使用 --full_train_set，而是使用相同的候选池"
echo ""
echo "训练参数："
echo "  RCE Epochs: ${RCE_EPOCHS} (增加，适应数据分布变化)"
echo "  GRPO Epochs: ${GRPO_EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  RCE LR: ${RCE_LR} (更保守)"
echo "  GRPO LR: ${GRPO_LR} (更激进)"
echo "  KL Beta: ${KL_BETA} (更宽松的约束)"
echo "  Require Positive Query: ${REQUIRE_POSITIVE_QUERY} (只训练有正样本的query)"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "${FIXED_RL_DATA}" ]; then
    echo "错误: 修复后的RL数据文件不存在: ${FIXED_RL_DATA}"
    exit 1
fi

if [ ! -f "${QUERY_EMB}" ]; then
    echo "错误: Query embeddings文件不存在: ${QUERY_EMB}"
    exit 1
fi

if [ ! -f "${CAND_EMB}" ]; then
    echo "错误: Candidate embeddings文件不存在: ${CAND_EMB}"
    exit 1
fi

if [ ! -f "${V2_CKPT_PATH}" ]; then
    echo "错误: V2 checkpoint文件不存在: ${V2_CKPT_PATH}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 构建训练命令
train_cmd="CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.workflows.grpo_post_train \
    --beam_data \"${FIXED_RL_DATA}\" \
    --img_emb \"${QUERY_EMB}\" \
    --sft_ckpt \"${V2_CKPT_PATH}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --rce_epochs ${RCE_EPOCHS} \
    --grpo_epochs ${GRPO_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --rce_lr ${RCE_LR} \
    --grpo_lr ${GRPO_LR} \
    --kl_beta ${KL_BETA} \
    --device cuda:0"

# 如果启用require_positive_query，添加参数
if [ "${REQUIRE_POSITIVE_QUERY}" == "true" ]; then
    train_cmd="${train_cmd} --require_positive_query"
fi

# 运行训练
echo ""
echo "开始训练..."
eval $train_cmd

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "新的checkpoint文件："
echo "  - rce_epoch*.pt (RCE阶段)"
echo "  - grpo_epoch*.pt (GRPO阶段)"
echo ""
echo "可以使用以下命令进行推理："
echo "bash scripts/grpo_inference.sh \\"
echo "    --grpo_ckpt ${OUTPUT_DIR}/grpo_epoch${GRPO_EPOCHS}.pt \\"
echo "    --img_emb ${QUERY_EMB} \\"
echo "    --beam_data ${FIXED_RL_DATA} \\"
echo "    --test_num 200 \\"
echo "    --gpu ${GPU_ID} \\"
echo "    --full_train_set"

