#!/bin/bash
# 从 RCE checkpoint 继续训练 GRPO（KL_BETA=0.155）
# 与 train_v3_kl0155.sh 配合使用，用于在 RCE 训练后继续 GRPO 训练

set -e

# 解析参数
rce_epoch=${1:-5}      # 从哪个 RCE epoch 开始（默认 epoch 5）
gpu_id=${2:-0}         # GPU ID
grpo_epochs=${3:-3}    # GRPO 训练多少个 epochs（默认 3）
rce_dir=${4:-kl012}   # RCE checkpoint 所在目录（默认 kl012，可以使用 kl012 的 RCE checkpoint）

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# 数据路径（使用与 KL_BETA=0.15 相同的数据文件，确保公平对比）
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"

# 从指定的 RCE checkpoint 开始（可以使用其他目录的 RCE checkpoint）
RCE_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_${rce_dir}/rce_epoch${rce_epoch}.pt"

# 输出目录（KL_BETA=0.155 的输出目录）
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl0155"

# 训练参数（与参考配置一致，KL_BETA=0.155）
RCE_EPOCHS=0          # 不做 RCE，直接加载已训练好的 RCE checkpoint
GRPO_EPOCHS=${grpo_epochs}
GRPO_LR=5e-6
KL_BETA=0.155
BATCH_SIZE=1

echo "=========================================="
echo "从 RCE epoch ${rce_epoch} 开始，使用 KL_BETA=0.155 训练 GRPO"
echo "=========================================="
echo "  - RCE checkpoint: ${RCE_CKPT}"
echo "  - RCE 目录: ${rce_dir}"
echo "  - RCE epochs: ${RCE_EPOCHS} (直接加载)"
echo "  - GRPO epochs: ${GRPO_EPOCHS}"
echo "  - GRPO LR: ${GRPO_LR}"
echo "  - KL beta: ${KL_BETA}"
echo "  - GPU: ${gpu_id}"
echo "  - 冻结 backbone: 是（与参考配置一致）"
echo "=========================================="

# 检查文件
if [ ! -f "$RL_DATA" ]; then
    echo "错误: RL数据文件不存在: $RL_DATA"
    exit 1
fi

if [ ! -f "$RCE_CKPT" ]; then
    echo "错误: RCE checkpoint 不存在: $RCE_CKPT"
    echo "可用的 RCE checkpoint:"
    find "${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_${rce_dir}" -name "rce_epoch*.pt" | xargs -n 1 basename
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
echo "  GRPO checkpoints: grpo_epoch1.pt ~ grpo_epoch${GRPO_EPOCHS}.pt"
echo ""

# 自动转换格式
echo "=========================================="
echo "自动转换 checkpoint 格式..."
echo "=========================================="
for epoch in $(seq 1 ${GRPO_EPOCHS}); do
    pt_path="${OUTPUT_DIR}/grpo_epoch${epoch}.pt"
    v2format_path="${OUTPUT_DIR}/grpo_epoch${epoch}_v2format.ckpt"
    
    if [ -f "$pt_path" ] && [ ! -f "$v2format_path" ]; then
        echo "转换 grpo_epoch${epoch}.pt..."
        python scripts/convert_v3_to_v2_format.py --v3_ckpt "${pt_path}"
        if [ -f "$v2format_path" ]; then
            echo "  ✓ 转换成功: grpo_epoch${epoch}_v2format.ckpt"
        else
            echo "  ✗ 转换失败: grpo_epoch${epoch}.pt"
        fi
    fi
done

echo ""
echo "=========================================="
echo "✓ 所有转换完成！"
echo "=========================================="
echo "下一步：运行评估脚本"
echo "  bash scripts/eval_grpo_kl0155.sh 1 ${gpu_id} 200"
echo "=========================================="

