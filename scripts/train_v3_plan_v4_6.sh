#!/bin/bash
# V4-6 方案：GRU Pointer Decoder + Coverage / Topic 原型覆盖
#
# 核心改动：
# 1. 引入 M 个可学习的"原型向量"（topic prototypes）
# 2. 每个 candidate 计算其 topic 分布
# 3. query 预测需要哪些 topics
# 4. 选择时倾向于覆盖未覆盖的 topics，强化互补覆盖、减少重复
#
# 训练配置（与方案五相同，RCE epochs 增加到 15）：
# - 使用相同的 RL 数据：rl_data_k64_v3_balanced.json
# - KL_BETA=0.1
# - GRPO_EPOCHS=50
# - GRPO_LR=5e-6
# - RCE_EPOCHS=15（增加，因为 Coverage 参数需要更多训练）
#
# 使用方法:
#   bash scripts/train_v3_plan_v4_6.sh [gpu_id] [num_topics]
#
# 示例:
#   bash scripts/train_v3_plan_v4_6.sh 4 16
#   bash scripts/train_v3_plan_v4_6.sh 4 32

set -e

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 解析参数
gpu_id=${1:-0}
num_topics=${2:-16}  # 默认使用 16 个 topics

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 使用与方案五完全相同的数据文件
RL_DATA_PATH="${PROJECT_DIR}/results/okvqa/generated_data/rl_data_k64_v3_balanced.json"

echo "=========================================="
echo "V4-6 方案：GRU + Coverage (${num_topics} topics)"
echo "=========================================="
echo "核心改动："
echo "  - GRUCell decoder: history-aware 能力"
echo "  - Step embedding: 不同 step 学到不同策略"
echo "  - Coverage 机制: ${num_topics} 个可学习原型向量"
echo "  - 每个 candidate 计算 topic 分布"
echo "  - query 预测需要哪些 topics"
echo "  - 选择时倾向覆盖未覆盖的 topics"
echo ""
echo "训练配置（RCE epochs 增加到 15）："
echo "  - RL_DATA: rl_data_k64_v3_balanced.json"
echo "  - KL_BETA=0.1"
echo "  - GRPO_EPOCHS=50"
echo "  - GRPO_LR=5e-6"
echo "  - RCE_EPOCHS=15（增加，因为 Coverage 参数需要更多训练）"
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
    echo "错误: 未找到 v2 checkpoint"
    exit 1
fi

# Query embeddings 路径
query_emb_path="${PROJECT_DIR}/results/okvqa/cache/query_embeddings.pt"
if [ ! -f "$query_emb_path" ]; then
    echo "错误: Query embeddings 不存在: $query_emb_path"
    exit 1
fi

# 输出目录（根据 num_topics 区分）
output_dir="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_6_topics${num_topics}"
mkdir -p "$output_dir"

echo ""
echo "文件路径："
echo "  - RL Data: $RL_DATA_PATH"
echo "  - SFT Checkpoint: $v2_ckpt_path"
echo "  - Query Embeddings: $query_emb_path"
echo "  - Output Directory: $output_dir"
echo "  - GPU: ${gpu_id}"
echo "  - Num Topics: ${num_topics}"
echo ""

# 执行训练
CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train_v4_6 \
    --beam_data "$RL_DATA_PATH" \
    --img_emb "$query_emb_path" \
    --sft_ckpt "$v2_ckpt_path" \
    --output_dir "$output_dir" \
    --rce_epochs 15 \
    --grpo_epochs 50 \
    --batch_size 1 \
    --rce_lr 1e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.1 \
    --num_layers 1 \
    --use_step_emb \
    --use_gru \
    --num_topics "$num_topics" \
    --cover_lambda_init 0.0 \
    --reward_mode hard_plus_soft \
    --hard_weight 1.0 \
    --soft_weight 1.0 \
    --rce_use_raw_reward \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ V4-6 (Coverage ${num_topics} topics) 训练完成！"
echo "=========================================="
echo "Checkpoint 保存在: $output_dir"
echo ""
echo "下一步：找到最优 epoch 并运行推理评估"
echo "  1. 查看训练日志，找到 Val Loss 最小的 epoch"
echo "  2. 运行推理: bash scripts/inference_v4_6.sh [gpu_id] [best_epoch] [num_topics]"
echo "=========================================="
