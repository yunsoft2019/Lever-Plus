#!/bin/bash
# V4-1 方案：Query 状态更新（history-aware 多步选择）
# 
# 核心改动：
# 1. 在 PointerSelectorV2 中加入 query_update_gate
# 2. 每步选完一个 demo 后更新 query_state
# 3. 让多步选择变成条件概率链：p(a1|q) p(a2|q,a1) ...
#
# 基于方案五的配置：
# - USE_RANK_ADVANTAGE=false (使用 Z-score 归一化)
# - KL_BETA=0.1 (降低 KL 惩罚)
# - GRPO_EPOCHS=50 (与方案五一致)
# - GRPO_LR=5e-6 (保持学习率)
#
# 预期效果：
# - shot≥3 不再明显下降（解决冗余选择问题）
# - 整体正确率提升
#
# 使用方法:
#   bash scripts/train_v3_plan_v4_1.sh [gpu_id]
#
# 示例:
#   bash scripts/train_v3_plan_v4_1.sh 0

set -e

# 解析参数
gpu_id=${1:-0}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 使用增强后的数据文件
RL_DATA_PATH="${PROJECT_DIR}/results/okvqa/generated_data/rl_data_k64_v3_balanced.json"

echo "=========================================="
echo "V4-1 方案：Query 状态更新（history-aware）"
echo "=========================================="
echo "核心改动："
echo "  - query_update_gate: 向量 gate 融合已选 demo"
echo "  - 每步更新 query_state，实现条件概率链"
echo ""
echo "训练配置（基于方案五）："
echo "  - USE_RANK_ADVANTAGE=false (Z-score 归一化)"
echo "  - KL_BETA=0.1"
echo "  - GRPO_EPOCHS=50"
echo "  - GRPO_LR=5e-6"
echo "  - RCE_EPOCHS=5"
echo "=========================================="

# 检查数据文件是否存在
if [ ! -f "$RL_DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $RL_DATA_PATH"
    exit 1
fi

# 查找 v2 checkpoint
v2_ckpt_path="${PROJECT_DIR}/results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
if [ ! -f "$v2_ckpt_path" ]; then
    v2_dir="${PROJECT_DIR}/results/okvqa/model_cpk/v2"
    if [ -d "$v2_dir" ]; then
        v2_ckpt_path=$(find "$v2_dir" -name "*RandSampler*.ckpt" -type f | head -1)
    fi
fi

if [ -z "$v2_ckpt_path" ] || [ ! -f "$v2_ckpt_path" ]; then
    echo "错误: 未找到 v2 checkpoint"
    exit 1
fi

# Query embeddings 路径
query_emb_path="${PROJECT_DIR}/results/okvqa/cache/query_embeddings.pt"
if [ ! -f "$query_emb_path" ]; then
    echo "错误: Query embeddings 不存在: $query_emb_path"
    exit 1
fi

# 输出目录
output_dir="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_1"
mkdir -p "$output_dir"

echo ""
echo "文件路径："
echo "  - RL Data: $RL_DATA_PATH"
echo "  - SFT Checkpoint: $v2_ckpt_path"
echo "  - Query Embeddings: $query_emb_path"
echo "  - Output Directory: $output_dir"
echo ""

# 执行训练
CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "$RL_DATA_PATH" \
    --img_emb "$query_emb_path" \
    --sft_ckpt "$v2_ckpt_path" \
    --output_dir "$output_dir" \
    --rce_epochs 5 \
    --grpo_epochs 50 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.1 \
    --num_layers 1 \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ V4-1 训练完成！"
echo "=========================================="
echo "Checkpoint 保存在: $output_dir"
echo ""
echo "下一步：运行推理评估"
echo "  bash scripts/inference_epoch2_800samples.sh"
echo "=========================================="
