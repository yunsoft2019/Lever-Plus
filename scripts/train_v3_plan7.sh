#!/bin/bash
# 方案七训练脚本：PPO Clip 代替 KL 惩罚
#
# 核心改动：
# 1. 使用 --use_ppo_clip_only 参数，只使用 PPO Clip 约束，不使用 KL 惩罚
# 2. 保持方案五的其他配置不变（USE_RANK_ADVANTAGE=false, GRPO_LR=5e-6）
# 3. 使用与方案五完全相同的数据
#
# 预期效果：
# - PPO Clip 允许更大的策略更新（不受 KL 惩罚限制）
# - 可能解决 Adv Std 太小的问题
# - 训练更激进，但有 clip 机制保护

set -e

# 参数
gpu_id=${1:-0}

# 方案七配置：PPO Clip Only
export USE_RANK_ADVANTAGE=false      # 方案五：关闭 Rank Normalization
export GRPO_LR=5e-6                  # 方案五：保持学习率
export KL_BETA=0.0                   # 方案七：KL 权重设为 0（实际由 use_ppo_clip_only 控制）
export GRPO_EPOCHS=50                # 与方案五一致
export REWARD_MODE=hard_plus_soft    # 与方案五一致
export USE_PPO_CLIP_ONLY=true        # 方案七核心：只使用 PPO Clip
export CLIP_EPSILON=0.2              # PPO Clip 参数（默认 0.2）

echo "=========================================="
echo "方案七训练：PPO Clip 代替 KL 惩罚"
echo "=========================================="
echo "配置："
echo "  - USE_RANK_ADVANTAGE=false (方案五)"
echo "  - GRPO_LR=5e-6 (方案五)"
echo "  - USE_PPO_CLIP_ONLY=true (方案七核心)"
echo "  - CLIP_EPSILON=0.2"
echo "  - KL_BETA=0.0 (不使用 KL 惩罚)"
echo "  - GRPO_EPOCHS=50"
echo "  - GPU: ${gpu_id}"
echo "=========================================="

# 数据路径（与方案五完全相同）
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"
RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_RandSampler_Qwen2_5-VL-3B-Instruct.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_plan7_ppo_clip_only"

# 检查文件是否存在
if [ ! -f "$RL_DATA" ]; then
    echo "❌ 错误：RL 数据文件不存在: ${RL_DATA}"
    exit 1
fi

if [ ! -f "$SFT_CKPT" ]; then
    echo "❌ 错误：SFT checkpoint 不存在: ${SFT_CKPT}"
    exit 1
fi

echo ""
echo "文件路径："
echo "  - RL Data: ${RL_DATA}"
echo "  - SFT Checkpoint: ${SFT_CKPT}"
echo "  - Output: ${OUTPUT_DIR}"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行训练
CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "$RL_DATA" \
    --img_emb "$QUERY_EMB" \
    --sft_ckpt "$SFT_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --rce_epochs 5 \
    --grpo_epochs 50 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.0 \
    --use_ppo_clip_only \
    --clip_epsilon 0.2 \
    --disable_adaptive_kl \
    --num_layers 1 \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ 方案七训练完成！"
echo "=========================================="
echo "Checkpoint 保存在: ${OUTPUT_DIR}"
echo ""
echo "推理命令："
echo "  export LEVER_LM_CHECKPOINT_PATH=${OUTPUT_DIR}/grpo_epoch2_v2format.ckpt"
echo "  bash scripts/inference.sh vqa okvqa_local ${gpu_id} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 800"
echo "=========================================="
