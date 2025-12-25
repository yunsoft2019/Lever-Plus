#!/bin/bash
# 方案五：关闭 Rank Normalization（使用 Z-score 归一化）+ 使用增强数据
# 
# 核心改动：
# 1. 确保使用 Z-score 归一化（默认行为，use_rank=False）
# 2. 降低 KL_BETA 到 0.1（减少 KL 惩罚对 PPO 信号的淹没）
# 3. 增加 GRPO_EPOCHS 到 3（给模型更多学习机会）
# 4. 使用增强数据文件 rl_data_k64_v3_balanced.json（99.8% query 有正样本）
#
# 预期效果：
# - Adv Std 从 0.006 提升到 0.5-1.0
# - PPO Loss 从 0.0009 提升到 0.01-0.1
# - 梯度信号增强 100 倍
#
# 使用方法:
#   bash scripts/train_v3_plan5.sh [gpu_id]
#
# 示例:
#   bash scripts/train_v3_plan5.sh 0

set -e

# 解析参数
gpu_id=${1:-0}

# 关键：使用增强后的数据文件（99.8% query 有正样本，reward std=0.96）
RL_DATA_PATH="./results/okvqa/generated_data/rl_data_k64_v3_balanced.json"

echo "=========================================="
echo "方案五：关闭 Rank Normalization + 使用增强数据"
echo "=========================================="
echo "核心配置："
echo "  - USE_RANK_ADVANTAGE=false (使用 Z-score 归一化)"
echo "  - KL_BETA=0.1 (降低 KL 惩罚)"
echo "  - GRPO_EPOCHS=3 (增加训练轮数)"
echo "  - GRPO_LR=5e-6 (保持学习率)"
echo "  - RL_DATA: rl_data_k64_v3_balanced.json (增强数据)"
echo "    - 99.8% query 有正样本"
echo "    - 56.4% candidate 是正样本"
echo "    - Reward std = 0.96"
echo "=========================================="

# 检查数据文件是否存在
if [ ! -f "$RL_DATA_PATH" ]; then
    echo "错误: 增强数据文件不存在: $RL_DATA_PATH"
    exit 1
fi

# 设置环境变量
export KL_BETA=0.1
export GRPO_EPOCHS=3
export GRPO_LR=5e-6
export RCE_EPOCHS=5
export REWARD_MODE=hard_plus_soft

# 查找 v2 checkpoint
v2_ckpt_path="./results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
if [ ! -f "$v2_ckpt_path" ]; then
    # 尝试查找任何匹配的 v2 checkpoint
    v2_dir="./results/okvqa/model_cpk/v2"
    if [ -d "$v2_dir" ]; then
        v2_ckpt_path=$(find "$v2_dir" -name "*RandSampler*.ckpt" -type f | head -1)
    fi
fi

if [ -z "$v2_ckpt_path" ] || [ ! -f "$v2_ckpt_path" ]; then
    echo "错误: 未找到 v2 checkpoint"
    exit 1
fi

# Query embeddings 路径
query_emb_path="./results/okvqa/cache/query_embeddings.pt"
if [ ! -f "$query_emb_path" ]; then
    echo "错误: Query embeddings 不存在: $query_emb_path"
    exit 1
fi

# 输出目录
output_dir="./results/okvqa/model_cpk/v3_plan5_balanced"
mkdir -p "$output_dir"

echo ""
echo "文件路径："
echo "  - RL Data: $RL_DATA_PATH"
echo "  - SFT Checkpoint: $v2_ckpt_path"
echo "  - Query Embeddings: $query_emb_path"
echo "  - Output Directory: $output_dir"
echo ""

# 直接执行 GRPO 训练（跳过数据生成步骤）
CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "$RL_DATA_PATH" \
    --img_emb "$query_emb_path" \
    --sft_ckpt "$v2_ckpt_path" \
    --output_dir "$output_dir" \
    --rce_epochs 5 \
    --grpo_epochs 3 \
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
echo "✓ 方案五训练完成！"
echo "=========================================="
echo "请检查训练日志中的以下指标："
echo "  - Adv Std: 预期 0.5-1.0（之前 0.006）"
echo "  - PPO Loss: 预期 0.01-0.1（之前 0.0009）"
echo "  - KL: 预期 0.05-0.15"
echo ""
echo "Checkpoint 保存在: $output_dir"
echo "=========================================="
