#!/bin/bash
# 使用过滤后的平衡数据训练 GRPO（KL_BETA=0.12）
# 过滤掉了全正/全负样本的 query，只保留混合 query

set -e

# 解析参数
rce_epoch=${1:-5}      # 从哪个 RCE epoch 开始（默认 epoch 5）
gpu_id=${2:-0}         # GPU ID
grpo_epochs=${3:-3}    # GRPO 训练多少个 epochs（默认 3）

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# 使用过滤后的平衡数据（620 个混合 query）
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3_balanced.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"

# 从指定的 RCE checkpoint 开始
RCE_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl012/rce_epoch${rce_epoch}.pt"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl012_balanced"

# 训练参数
RCE_EPOCHS=0          # 不做 RCE，直接加载已训练好的 RCE checkpoint
GRPO_EPOCHS=${grpo_epochs}
GRPO_LR=5e-6
KL_BETA=0.12
BATCH_SIZE=1

echo "=========================================="
echo "使用平衡数据训练 GRPO（KL_BETA=0.12）"
echo "=========================================="
echo "  - 数据: rl_data_k64_v3_balanced.json (620 个混合 query)"
echo "  - RCE checkpoint: ${RCE_CKPT}"
echo "  - GRPO epochs: ${GRPO_EPOCHS}"
echo "  - GRPO LR: ${GRPO_LR}"
echo "  - KL beta: ${KL_BETA}"
echo "  - GPU: ${gpu_id}"
echo "=========================================="

# 检查文件
if [ ! -f "$RL_DATA" ]; then
    echo "错误: RL数据文件不存在: $RL_DATA"
    echo "请先运行: python scripts/filter_balanced_queries.py"
    exit 1
fi

if [ ! -f "$RCE_CKPT" ]; then
    echo "错误: RCE checkpoint 不存在: $RCE_CKPT"
    echo "可用的 RCE checkpoint:"
    ls -1 "${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl012"/rce_epoch*.pt 2>/dev/null || echo "  无"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

export CUDA_VISIBLE_DEVICES=${gpu_id}
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
    --batch_size ${BATCH_SIZE} \
    --reward_mode hard_plus_soft \
    --freeze_backbone_in_grpo \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ GRPO 训练完成！"
echo "  Checkpoint: ${OUTPUT_DIR}"
echo "=========================================="

# 自动转换格式
echo "转换 checkpoint 格式..."
for epoch in $(seq 1 ${GRPO_EPOCHS}); do
    pt_path="${OUTPUT_DIR}/grpo_epoch${epoch}.pt"
    v2format_path="${OUTPUT_DIR}/grpo_epoch${epoch}_v2format.ckpt"
    
    if [ -f "$pt_path" ] && [ ! -f "$v2format_path" ]; then
        echo "转换 grpo_epoch${epoch}.pt..."
        python scripts/convert_v3_to_v2_format.py --v3_ckpt "${pt_path}"
    fi
done

echo ""
echo "=========================================="
echo "✓ 完成！"
echo "=========================================="
