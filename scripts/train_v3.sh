#!/bin/bash
# V3 训练脚本（统一参数格式）
# 使用方法: bash scripts/train_v3.sh <task> <dataset> <gpu_id> <lever_lm> <sampler> <beam_model>
# 示例: bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
#
# 该脚本会自动执行 v3 训练的完整流程：
#   Step 0: 导出 Embeddings（如果不存在）
#   Step 1: 生成 RL 数据（如果不存在）
#   Step 2: 执行 GRPO 强化学习训练
#
# 参数说明:
#   task: 任务类型（vqa, caption）
#   dataset: 数据集名称（okvqa_local, vqav2_local）
#   gpu_id: GPU 编号（0, 1, 2, ...）
#   lever_lm: 配置名称（query_img_text_icd_img_text）
#   sampler: 采样器类型（rand_sampler, text_sim_sampler, img_sim_sampler, mix_sampler）
#   beam_model: 束搜索模型（flamingo_3B, qwen2.5_vl_3B）
#
# 环境变量（可选，用于自定义训练参数）:
#   RCE_EPOCHS: RCE 预热轮数（默认: 5）
#   GRPO_EPOCHS: GRPO 训练轮数（默认: 10）
#   BATCH_SIZE: 批次大小（默认: 1）
#   RCE_LR: RCE 学习率（默认: 1e-4）
#   GRPO_LR: GRPO 学习率（默认: 1e-5）
#   KL_BETA: KL 散度权重（默认: 0.1）
#   NUM_LAYERS: Cross-Attention 层数（默认: 1，与 v2 一致）
#   REWARD_MODE: Reward 模式（默认: hard_plus_soft，可选: hard_only, soft_only, legacy）
#   HARD_WEIGHT: Hard correctness 权重（默认: 1.0）
#   SOFT_WEIGHT: Soft correctness 权重（默认: 1.0）

set -e

# 解析参数
task=${1:-vqa}
dataset=${2:-okvqa_local}
gpu_id=${3:-0}
lever_lm=${4:-query_img_text_icd_img_text}
sampler=${5:-rand_sampler}
beam_model=${6:-qwen2.5_vl_3B}

# 将 sampler 转换为大驼峰格式
case "$sampler" in
    rand_sampler)
        sampler_name="RandSampler"
        ;;
    text_sim_sampler)
        sampler_name="TextSimSampler"
        ;;
    img_sim_sampler)
        sampler_name="ImgSimSampler"
        ;;
    mix_sampler)
        sampler_name="MixSampler"
        ;;
    *)
        sampler_name="${sampler}"
        ;;
esac

# 将 beam_model 映射到文件名中使用的模型名称
case "$beam_model" in
    flamingo_3B)
        model_name="flamingo_3B"
        ;;
    qwen2.5_vl_3B|qwen2_5_vl_3B)
        model_name="Qwen2_5-VL-3B-Instruct"
        ;;
    *)
        model_name=$(echo "$beam_model" | sed 's/\./_/g' | sed 's/\//_/g' | sed 's/ /_/g')
        ;;
esac

# 根据数据集自动设置参数
case "$dataset" in
    okvqa*|OKVQA*)
        sample_num=800
        dataset_name="okvqa"
        ;;
    vqav2*|VQAV2*)
        sample_num=5000
        dataset_name="vqav2"
        ;;
    *)
        sample_num=5000
        dataset_name="${dataset}"
        ;;
esac

# 构建文件路径
model_name_safe=$(echo "$model_name" | sed 's/-/_/g' | sed 's/\./_/g')
checkpoint_filename="${model_name_safe}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

# 文件路径
query_emb_path="./results/${dataset_name}/cache/query_embeddings.pt"
cand_emb_path="./results/${dataset_name}/cache/candidate_embeddings.pt"
# RL 数据路径：按 sampler 和 beam_model 分开保存
rl_data_path="./results/${dataset_name}/generated_data/rl_data_${sampler_name}_${model_name}.json"

# 读取 reward 配置（用于输出目录命名）
reward_mode=${REWARD_MODE:-hard_plus_soft}
hard_weight=${HARD_WEIGHT:-1.0}
soft_weight=${SOFT_WEIGHT:-1.0}

# 构建 reward 后缀（用于区分不同实验）
if [ "$reward_mode" == "hard_plus_soft" ]; then
    # 只有当权重不是默认值时才添加后缀
    if [ "$hard_weight" != "1.0" ] || [ "$soft_weight" != "1.0" ]; then
        reward_suffix="_h${hard_weight}_s${soft_weight}"
    else
        reward_suffix=""
    fi
elif [ "$reward_mode" == "legacy" ]; then
    reward_suffix="_legacy"
else
    reward_suffix="_${reward_mode}"
fi

# 每个采样器和模型使用独立的输出目录，避免覆盖
output_dir="./results/${dataset_name}/model_cpk/v3_${sampler_name}_${model_name}${reward_suffix}"

# 查找 v2 checkpoint
v2_ckpt_path="./results/${dataset_name}/model_cpk/v2/${checkpoint_filename}_best.ckpt"
if [ ! -f "$v2_ckpt_path" ]; then
    # 尝试查找任何匹配的 v2 checkpoint
    v2_dir="./results/${dataset_name}/model_cpk/v2"
    if [ -d "$v2_dir" ]; then
        v2_ckpt_path=$(find "$v2_dir" -name "*${sampler_name}*.ckpt" -type f | head -1)
    fi
fi

# 设置默认训练参数（可通过环境变量覆盖）
rce_epochs=${RCE_EPOCHS:-5}
grpo_epochs=${GRPO_EPOCHS:-10}
batch_size=${BATCH_SIZE:-1}
rce_lr=${RCE_LR:-1e-4}
grpo_lr=${GRPO_LR:-1e-5}
kl_beta=${KL_BETA:-0.1}
num_layers=${NUM_LAYERS:-1}
# 新的 Reward 参数
reward_mode=${REWARD_MODE:-hard_plus_soft}
hard_weight=${HARD_WEIGHT:-1.0}
soft_weight=${SOFT_WEIGHT:-1.0}

echo "=========================================="
echo "V3 训练配置"
echo "=========================================="
echo "Task: ${task}"
echo "Dataset: ${dataset} → ${dataset_name}"
echo "GPU ID: ${gpu_id}"
echo "Sampler: ${sampler} → ${sampler_name}"
echo "Beam Model: ${beam_model} → ${model_name}"
echo "=========================================="
echo "训练参数:"
echo "  RCE Epochs: ${rce_epochs}"
echo "  GRPO Epochs: ${grpo_epochs}"
echo "  Batch Size: ${batch_size}"
echo "  RCE LR: ${rce_lr}"
echo "  GRPO LR: ${grpo_lr}"
echo "  KL Beta: ${kl_beta}"
echo "  Num Layers: ${num_layers}"
echo "Reward 参数:"
echo "  Reward Mode: ${reward_mode}"
echo "  Hard Weight: ${hard_weight}"
echo "  Soft Weight: ${soft_weight}"
echo "=========================================="

# Step 0: 检查并导出 Embeddings
echo ""
echo "=========================================="
echo "Step 0: 检查 Embeddings"
echo "=========================================="

if [ ! -f "$query_emb_path" ] || [ ! -f "$cand_emb_path" ]; then
    echo "Embeddings 不存在，开始导出..."
    
    if [ -z "$v2_ckpt_path" ] || [ ! -f "$v2_ckpt_path" ]; then
        echo "错误: 未找到 v2 checkpoint，无法导出 embeddings"
        echo "请先训练 v2 模型: bash scripts/train_lever_lm.sh ${task} ${dataset} ${gpu_id} ${lever_lm} ${sampler} ${beam_model} v2"
        exit 1
    fi
    
    bash scripts/export_embeddings.sh \
        "$v2_ckpt_path" \
        "$dataset" \
        "./results/${dataset_name}/cache" \
        "cuda:${gpu_id}"
    
    echo "✓ Embeddings 导出完成"
else
    echo "✓ Embeddings 已存在，跳过导出"
    echo "  - Query: ${query_emb_path}"
    echo "  - Candidate: ${cand_emb_path}"
fi

# Step 1: 生成 RL 数据（强制重新生成，覆盖旧数据）
echo ""
echo "=========================================="
echo "Step 1: 生成 RL 数据"
echo "=========================================="

# 如果旧数据存在，先删除
if [ -f "$rl_data_path" ]; then
    echo "删除旧的 RL 数据: ${rl_data_path}"
    rm -f "$rl_data_path"
fi

echo "开始生成 RL 数据..."
bash scripts/generate_rl_data_for_sampler.sh \
    "$sampler" \
    "$beam_model" \
    "$dataset" \
    "cuda:${gpu_id}"

echo "✓ RL 数据生成完成"

# Step 2: 执行 GRPO 训练（强制重新训练，覆盖旧模型）
echo ""
echo "=========================================="
echo "Step 2: 执行 GRPO 强化学习训练"
echo "=========================================="

# 如果旧模型目录存在，先删除
if [ -d "$output_dir" ]; then
    echo "删除旧的模型目录: ${output_dir}"
    rm -rf "$output_dir"
fi

# 创建输出目录
mkdir -p "$output_dir"

echo "SFT Checkpoint: ${v2_ckpt_path}"
echo "RL Data: ${rl_data_path}"
echo "Query Embeddings: ${query_emb_path}"
echo "Output Directory: ${output_dir}"
echo ""

CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data "$rl_data_path" \
    --img_emb "$query_emb_path" \
    --sft_ckpt "$v2_ckpt_path" \
    --output_dir "$output_dir" \
    --rce_epochs ${rce_epochs} \
    --grpo_epochs ${grpo_epochs} \
    --batch_size ${batch_size} \
    --rce_lr ${rce_lr} \
    --grpo_lr ${grpo_lr} \
    --kl_beta ${kl_beta} \
    --num_layers ${num_layers} \
    --reward_mode ${reward_mode} \
    --hard_weight ${hard_weight} \
    --soft_weight ${soft_weight} \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ V3 训练完成！"
echo "=========================================="
echo "Checkpoint 保存在: ${output_dir}"
echo "  - RCE checkpoints: rce_epoch1.pt ~ rce_epoch${rce_epochs}.pt"
echo "  - GRPO checkpoints: grpo_epoch1.pt ~ grpo_epoch${grpo_epochs}.pt"
echo "  - 推荐使用: grpo_epoch${grpo_epochs}.pt"
echo ""
echo "推理命令（自动转换格式）:"
echo "  bash scripts/inference.sh ${task} ${dataset} ${gpu_id} ${lever_lm} ${sampler} ${beam_model} v3"
echo "=========================================="
