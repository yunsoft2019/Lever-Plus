#!/bin/bash
# 方案九：Curriculum Learning（渐进式训练）
#
# 核心思路：
# 1. 第一阶段：只用高质量 query（正负样本差距大的，约 53% 数据）训练
# 2. 第二阶段：用全部数据继续训练（从第一阶段的 checkpoint 继续）
#
# 预期效果：
# - 模型先学会简单的区分，建立良好的基础
# - 再处理难样本，避免被噪声干扰

set -e

gpu_id=${1:-3}
phase=${2:-1}  # 1 或 2

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=19_train=24.18280_val=21.98483.ckpt"

# 方案九配置（基于方案五）
export USE_RANK_ADVANTAGE=false
export GRPO_LR=5e-6
export KL_BETA=0.1
export REWARD_MODE=hard_plus_soft
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ "$phase" == "1" ]; then
    echo "=========================================="
    echo "方案九 - 第一阶段：高质量数据训练"
    echo "=========================================="
    
    RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3_balanced_phase1.json"
    OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_plan9_curriculum_phase1"
    GRPO_EPOCHS=30  # 第一阶段训练 30 epochs
    
    echo "配置："
    echo "  - 数据：高质量 query（约 53%）"
    echo "  - GRPO_EPOCHS=30"
    echo "  - USE_RANK_ADVANTAGE=false"
    echo "  - GRPO_LR=5e-6"
    echo "  - KL_BETA=0.1"
    echo "  - GPU: ${gpu_id}"
    echo "=========================================="
    
    if [ ! -f "$RL_DATA" ]; then
        echo "❌ 错误：第一阶段数据文件不存在: ${RL_DATA}"
        exit 1
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
        --beam_data "$RL_DATA" \
        --img_emb "$QUERY_EMB" \
        --sft_ckpt "$SFT_CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --rce_epochs 5 \
        --grpo_epochs ${GRPO_EPOCHS} \
        --batch_size 1 \
        --rce_lr 1e-4 \
        --grpo_lr 5e-6 \
        --kl_beta 0.1 \
        --disable_adaptive_kl \
        --num_layers 1 \
        --reward_mode hard_plus_soft \
        --hard_weight 1.0 \
        --soft_weight 1.0 \
        --rce_use_raw_reward \
        --device cuda:0
    
    echo ""
    echo "=========================================="
    echo "✓ 方案九第一阶段训练完成！"
    echo "=========================================="
    echo "Checkpoint 保存在: ${OUTPUT_DIR}"
    echo ""
    echo "下一步：运行第二阶段训练"
    echo "  bash scripts/train_v3_plan9_curriculum.sh ${gpu_id} 2"
    echo "=========================================="

elif [ "$phase" == "2" ]; then
    echo "=========================================="
    echo "方案九 - 第二阶段：全部数据继续训练"
    echo "=========================================="
    
    RL_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_k64_v3_balanced.json"
    PHASE1_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_plan9_curriculum_phase1"
    OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_plan9_curriculum_phase2"
    GRPO_EPOCHS=20  # 第二阶段训练 20 epochs
    
    # 找到第一阶段最优的 checkpoint（Val Loss 最小）
    # 默认使用 grpo_epoch5（通常是早期 epoch 效果最好）
    PHASE1_CKPT="${PHASE1_DIR}/grpo_epoch5.pt"
    
    if [ ! -f "$PHASE1_CKPT" ]; then
        echo "❌ 错误：第一阶段 checkpoint 不存在: ${PHASE1_CKPT}"
        echo "请先运行第一阶段训练：bash scripts/train_v3_plan9_curriculum.sh ${gpu_id} 1"
        exit 1
    fi
    
    echo "配置："
    echo "  - 数据：全部 query（100%）"
    echo "  - 从第一阶段 checkpoint 继续: ${PHASE1_CKPT}"
    echo "  - GRPO_EPOCHS=20"
    echo "  - USE_RANK_ADVANTAGE=false"
    echo "  - GRPO_LR=5e-6"
    echo "  - KL_BETA=0.1"
    echo "  - GPU: ${gpu_id}"
    echo "=========================================="
    
    mkdir -p "$OUTPUT_DIR"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
        --beam_data "$RL_DATA" \
        --img_emb "$QUERY_EMB" \
        --sft_ckpt "$PHASE1_CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --rce_epochs 0 \
        --grpo_epochs ${GRPO_EPOCHS} \
        --batch_size 1 \
        --rce_lr 1e-4 \
        --grpo_lr 5e-6 \
        --kl_beta 0.1 \
        --disable_adaptive_kl \
        --num_layers 1 \
        --reward_mode hard_plus_soft \
        --hard_weight 1.0 \
        --soft_weight 1.0 \
        --rce_use_raw_reward \
        --device cuda:0
    
    echo ""
    echo "=========================================="
    echo "✓ 方案九第二阶段训练完成！"
    echo "=========================================="
    echo "Checkpoint 保存在: ${OUTPUT_DIR}"
    echo ""
    echo "推理命令："
    echo "  export LEVER_LM_CHECKPOINT_PATH=${OUTPUT_DIR}/grpo_epochX_v2format.ckpt"
    echo "  bash scripts/inference.sh vqa okvqa_local ${gpu_id} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 800"
    echo "=========================================="

else
    echo "用法: bash scripts/train_v3_plan9_curriculum.sh <gpu_id> <phase>"
    echo "  phase=1: 第一阶段（高质量数据）"
    echo "  phase=2: 第二阶段（全部数据）"
    exit 1
fi
