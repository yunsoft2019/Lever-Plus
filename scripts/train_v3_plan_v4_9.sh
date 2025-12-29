#!/bin/bash
# V4-9 方案：Two-Stage Coarse-to-Fine TopM 精排
#
# 核心改动：
# 1. 第一阶段：cheap score（点积）快速筛选 TopM 候选
# 2. 第二阶段：heavy refine（Cross-Attention / MLP）精排 TopM
# 3. 主要利好效率，也可能提升鲁棒性（减少噪声候选干扰）
#
# 训练配置：
# - 使用相同的 RL 数据：rl_data_k64_v3_balanced.json
# - KL_BETA=0.1
# - GRPO_EPOCHS=50
# - GRPO_LR=5e-6
# - RCE_EPOCHS=15
#
# 使用方法:
#   bash scripts/train_v3_plan_v4_9.sh [gpu_id] [top_m] [refine_type]
#
# 示例:
#   bash scripts/train_v3_plan_v4_9.sh 4 8 attn
#   bash scripts/train_v3_plan_v4_9.sh 4 16 mlp

set -e

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 解析参数
gpu_id=${1:-0}
top_m=${2:-8}  # 默认精排 TopM=8
refine_type=${3:-attn}  # 默认使用 attention 精排

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 使用与其他方案相同的数据文件
RL_DATA_PATH="${PROJECT_DIR}/results/okvqa/generated_data/rl_data_k64_v3_balanced.json"

echo "=========================================="
echo "V4-9 方案：Two-Stage Coarse-to-Fine"
echo "=========================================="
echo "核心改动："
echo "  - 第一阶段：cheap score（点积）快速筛选 TopM"
echo "  - 第二阶段：heavy refine（${refine_type}）精排"
echo "  - TopM: ${top_m}"
echo "  - Refine Type: ${refine_type}"
echo ""
echo "训练配置："
echo "  - RL_DATA: rl_data_k64_v3_balanced.json"
echo "  - KL_BETA=0.1"
echo "  - GRPO_EPOCHS=50"
echo "  - GRPO_LR=5e-6"
echo "  - RCE_EPOCHS=15"
echo "=========================================="

# 检查数据文件是否存在
if [ ! -f "$RL_DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $RL_DATA_PATH"
    exit 1
fi

# 查找 v2 checkpoint（用于初始化共享参数）
v2_ckpt_path="${PROJECT_DIR}/results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
if [ ! -f "$v2_ckpt_path" ]; then
    v2_dir="${PROJECT_DIR}/results/okvqa/model_cpk/v2"
    if [ -d "$v2_dir" ]; then
        v2_ckpt_path=$(find "$v2_dir" -name "*RandSampler*.ckpt" -type f | head -1)
    fi
fi

if [ -z "$v2_ckpt_path" ] || [ ! -f "$v2_ckpt_path" ]; then
    echo "警告: 未找到 v2 checkpoint，将从头训练"
    v2_ckpt_path=""
fi

# Query embeddings 路径
query_emb_path="${PROJECT_DIR}/results/okvqa/cache/query_embeddings.pt"
if [ ! -f "$query_emb_path" ]; then
    echo "错误: Query embeddings 不存在: $query_emb_path"
    exit 1
fi

# 输出目录
output_dir="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_9_topm${top_m}_${refine_type}"
mkdir -p "$output_dir"

echo ""
echo "文件路径："
echo "  - RL Data: $RL_DATA_PATH"
echo "  - SFT Checkpoint: ${v2_ckpt_path:-无}"
echo "  - Query Embeddings: $query_emb_path"
echo "  - Output Directory: $output_dir"
echo "  - GPU: ${gpu_id}"
echo "  - TopM: ${top_m}"
echo "  - Refine Type: ${refine_type}"
echo ""

# 构建训练命令
train_cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train_v4_9 \
    --beam_data \"$RL_DATA_PATH\" \
    --img_emb \"$query_emb_path\" \
    --output_dir \"$output_dir\" \
    --rce_epochs 15 \
    --grpo_epochs 50 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.1 \
    --num_layers 1 \
    --top_m $top_m \
    --refine_type $refine_type \
    --use_gru \
    --use_step_emb \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0"

# 如果有 SFT checkpoint，添加参数
if [ -n "$v2_ckpt_path" ] && [ -f "$v2_ckpt_path" ]; then
    train_cmd="$train_cmd --sft_ckpt \"$v2_ckpt_path\""
fi

# 执行训练
eval $train_cmd

echo ""
echo "=========================================="
echo "✓ V4-9 (Two-Stage TopM=${top_m}, ${refine_type}) 训练完成！"
echo "=========================================="
echo "Checkpoint 保存在: $output_dir"
echo ""
echo "下一步：找到最优 epoch 并运行推理评估"
echo "  1. 查看训练日志，找到 Val Loss 最小的 epoch"
echo "  2. 运行推理: bash scripts/inference_v4_9.sh [gpu_id] [stage] [epoch] [top_m] [refine_type]"
echo "=========================================="
