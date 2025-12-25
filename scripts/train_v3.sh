#!/bin/bash
# V3 训练脚本（统一参数格式）
# 使用方法: bash scripts/train_v3.sh <task> <dataset> <gpu_id> <lever_lm> <sampler> <beam_model>
# 示例: bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
#
# 该脚本会自动执行 v3 训练的完整流程：
#   Step 0: 导出 Embeddings（如果不存在）
#   Step 1: 生成 RL 数据（如果不存在）
#   Step 2: 执行 GRPO 强化学习训练
#   Step 3: 自动转换为 v2 格式（如果不存在，用于推理）
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
#   GRPO_EPOCHS: GRPO 训练轮数（默认: 1，文档建议先做 RCE-only baseline，设为 0）
#   BATCH_SIZE: 批次大小（默认: 1）
#   RCE_LR: RCE 学习率（默认: 1e-4）
#   GRPO_LR: GRPO 学习率（默认: 5e-6，轻量 GRPO）
#   KL_BETA: KL 散度权重（默认: 0.15，稍强的 KL 约束）
#   NUM_LAYERS: Cross-Attention 层数（默认: 1，与 v2 一致）
#   REWARD_MODE: Reward 模式（默认: hard_plus_soft）
#     - hard_plus_soft: reward = vqa_correct + vqa_acc_score，范围 [0, 2]
#     - separated: 正样本 [2,3]，负样本 [0,1]（需要数据有足够正样本）
#     - hard_plus_gtprob_plus_rel: reward = hard*hard_weight + gtprob*soft_weight + rel*(1-hard)*rel_weight
#       （推荐：负样本也有梯度信号，通过 relevance shaping）
#   HARD_WEIGHT: Hard correctness 权重（默认: 1.0）
#   SOFT_WEIGHT: Soft correctness 权重（默认: 1.0）
#   REL_WEIGHT: Relevance 权重（默认: 0.1，hard_plus_gtprob_plus_rel 模式使用）
#   USE_RANK_ADVANTAGE: 是否使用排名归一化计算 advantage（默认: false）
#   RCE_USE_RAW_REWARD: RCE 使用原始 reward（默认: true，保留正负样本的绝对差异）
#   FREEZE_BACKBONE_IN_GRPO: GRPO 时冻结 backbone（默认: false）
#   SKIP_FALLBACK_REWARD: 跳过使用 fallback 方式计算的 RL 样本（默认: true，推荐启用；传 false 可禁用）
#   REQUIRE_POSITIVE_QUERY: 只保留至少有一个正样本的 query（默认: false；传 true 可启用，用于正样本挖掘后的高质量数据训练）
#
# 实验路线（根据 LeverPlus_v3_RL_plan_cn.md）:
#   Step 3 (RCE-only baseline): export GRPO_EPOCHS=0 && bash scripts/train_v3.sh ...（默认配置，使用归一化reward）
#   Step 4 (RCE + 轻量 GRPO): export GRPO_EPOCHS=1 GRPO_LR=5e-6 KL_BETA=0.15 && bash scripts/train_v3.sh ...
#   3.4 对比实验（测试 raw reward）: export RCE_USE_RAW_REWARD=true && bash scripts/train_v3.sh ...
#   3.5.2 冻结 backbone: export FREEZE_BACKBONE_IN_GRPO=true GRPO_EPOCHS=3 && bash scripts/train_v3.sh ...

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
# RL 数据路径：可通过环境变量 RL_DATA_PATH 指定，否则按 sampler 和 beam_model 自动生成
if [ -n "${RL_DATA_PATH}" ]; then
    rl_data_path="${RL_DATA_PATH}"
    echo "使用自定义 RL 数据路径: ${rl_data_path}"
else
    rl_data_path="./results/${dataset_name}/generated_data/rl_data_${sampler_name}_${model_name}.json"
fi

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
# 根据文档实验路线，默认先做 RCE-only baseline（GRPO_EPOCHS=0）
rce_epochs=${RCE_EPOCHS:-5}
grpo_epochs=${GRPO_EPOCHS:-0}  # 默认 0，先做 RCE-only baseline
batch_size=${BATCH_SIZE:-1}
rce_lr=${RCE_LR:-1e-4}
grpo_lr=${GRPO_LR:-5e-6}  # 轻量 GRPO 使用更小的学习率
kl_beta=${KL_BETA:-0.15}  # 稍强的 KL 约束
num_layers=${NUM_LAYERS:-1}
# 新的 Reward 参数
# 注意：separated 模式需要数据中有足够的正样本（vqa_correct=1），否则 reward 全是 0
# 当前数据正确率较低，建议使用 hard_plus_soft 模式
reward_mode=${REWARD_MODE:-hard_plus_soft}
hard_weight=${HARD_WEIGHT:-1.0}
soft_weight=${SOFT_WEIGHT:-1.0}
rel_weight=${REL_WEIGHT:-0.1}  # relevance权重（hard_plus_gtprob_plus_rel模式使用）
# 3.4、3.5.2 和 3.3.3 新增参数
# 默认使用 raw reward，保留正负样本的绝对差异
rce_use_raw_reward=${RCE_USE_RAW_REWARD:-true}
freeze_backbone_in_grpo=${FREEZE_BACKBONE_IN_GRPO:-false}
skip_fallback_reward=${SKIP_FALLBACK_REWARD:-true}  # 默认启用，传 false 可禁用
# 正样本挖掘：只保留至少有一个正样本的 query
require_positive_query=${REQUIRE_POSITIVE_QUERY:-false}  # 默认禁用，传 true 可启用

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
if [ "${grpo_epochs}" -eq 0 ]; then
    echo "  → RCE-only baseline 模式（符合文档 Step 3 建议）"
else
    echo "  → RCE + GRPO 模式（符合文档 Step 4 建议）"
fi
echo "  Batch Size: ${batch_size}"
echo "  RCE LR: ${rce_lr}"
echo "  GRPO LR: ${grpo_lr}"
echo "  KL Beta: ${kl_beta}"
echo "  Num Layers: ${num_layers}"
echo "Reward 参数:"
echo "  Reward Mode: ${reward_mode}"
echo "  Hard Weight: ${hard_weight}"
echo "  Soft Weight: ${soft_weight}"
if [ "${reward_mode}" == "hard_plus_gtprob_plus_rel" ]; then
    echo "  Rel Weight: ${rel_weight}"
fi
if [ "${rce_use_raw_reward}" == "true" ]; then
    echo "  RCE Reward: 原始 reward (beam_rewards_raw) [显式指定]"
else
    echo "  RCE Reward: 归一化后的 reward (beam_rewards) [默认，与 rce_epoch5.pt 一致]"
fi
if [ "${freeze_backbone_in_grpo}" == "true" ]; then
    echo "  GRPO: 冻结 backbone，只训练 pointer head"
fi
if [ "${skip_fallback_reward}" == "true" ]; then
    echo "  Skip Fallback: 跳过 fallback 样本，只使用官方 VQA metric [默认启用]"
else
    echo "  Skip Fallback: 已禁用（使用所有样本，包括 fallback）"
fi
if [ "${require_positive_query}" == "true" ]; then
    echo "  Require Positive Query: 只保留有正样本的 query [已启用]"
fi
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

# Step 1: 检查并生成 RL 数据（如果不存在或缺少 vqa_eval_mode 字段）
echo ""
echo "=========================================="
echo "Step 1: 检查 RL 数据"
echo "=========================================="

need_regenerate=false

# 如果指定了自定义 RL 数据路径，检查文件是否存在，不存在则报错退出
if [ -n "${RL_DATA_PATH}" ]; then
    if [ ! -f "$rl_data_path" ]; then
        echo "❌ 错误：指定的 RL 数据文件不存在: ${rl_data_path}"
        exit 1
    fi
    echo "✓ 使用自定义 RL 数据路径，跳过生成"
    echo "  - RL Data: ${rl_data_path}"
elif [ ! -f "$rl_data_path" ]; then
    echo "RL 数据不存在，需要生成"
    need_regenerate=true
else
    # 检查 RL 数据是否包含 vqa_eval_mode 字段（新格式）
    echo "检查 RL 数据格式..."
    if python3 -c "
import json
import sys
try:
    with open('$rl_data_path', 'r') as f:
        data = json.load(f)
    # 检查第一个 query 的第一个 candidate 是否有 vqa_eval_mode 字段
    first_query = next(iter(data.values()))
    if 'pointer_candidates' in first_query and len(first_query['pointer_candidates']) > 0:
        first_candidate = first_query['pointer_candidates'][0]
        if 'vqa_eval_mode' not in first_candidate:
            print('缺少 vqa_eval_mode 字段')
            sys.exit(1)
        else:
            print('包含 vqa_eval_mode 字段')
            sys.exit(0)
    else:
        print('数据格式异常')
        sys.exit(1)
except Exception as e:
    print(f'检查失败: {e}')
    sys.exit(1)
" 2>/dev/null; then
        echo "✓ RL 数据格式正确（包含 vqa_eval_mode 字段），跳过生成"
        echo "  - RL Data: ${rl_data_path}"
    else
        echo "⚠️  RL 数据缺少 vqa_eval_mode 字段（旧格式），需要重新生成"
        need_regenerate=true
    fi
fi

if [ "$need_regenerate" = true ]; then
    echo "开始生成 RL 数据..."
    bash scripts/generate_rl_data_for_sampler.sh \
        "$sampler" \
        "$beam_model" \
        "$dataset" \
        "cuda:${gpu_id}"
    echo "✓ RL 数据生成完成"
fi

# Step 2: 执行 GRPO 训练
echo ""
echo "=========================================="
echo "Step 2: 执行 GRPO 强化学习训练"
echo "=========================================="

# 创建输出目录
mkdir -p "$output_dir"

echo "SFT Checkpoint: ${v2_ckpt_path}"
echo "RL Data: ${rl_data_path}"
echo "Query Embeddings: ${query_emb_path}"
echo "Output Directory: ${output_dir}"
echo ""

# 构建训练命令
train_cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
    --beam_data \"$rl_data_path\" \
    --img_emb \"$query_emb_path\" \
    --sft_ckpt \"$v2_ckpt_path\" \
    --output_dir \"$output_dir\" \
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
    --rel_weight ${rel_weight}"

# 3.4: 如果指定使用原始 reward（默认使用归一化后的 reward）
if [ "${rce_use_raw_reward}" == "true" ]; then
    train_cmd="${train_cmd} --rce_use_raw_reward"
fi

# 3.5.2: 如果指定冻结 backbone
if [ "${freeze_backbone_in_grpo}" == "true" ]; then
    train_cmd="${train_cmd} --freeze_backbone_in_grpo"
fi

# 3.3.3: skip_fallback_reward 默认启用，只有禁用时才传参数
if [ "${skip_fallback_reward}" == "false" ]; then
    train_cmd="${train_cmd} --no_skip_fallback_reward"
fi

# 正样本挖掘：只保留有正样本的 query
if [ "${require_positive_query}" == "true" ]; then
    train_cmd="${train_cmd} --require_positive_query"
fi

# 添加设备参数
train_cmd="${train_cmd} --device cuda:0"

# 执行训练命令
eval $train_cmd

# Step 3: 自动转换为 v2 格式（如果不存在）
echo ""
echo "=========================================="
echo "Step 3: 检查并转换 checkpoint 格式"
echo "=========================================="

# 确定要转换的 checkpoint 文件
if [ "${grpo_epochs}" -eq 0 ]; then
    # RCE-only baseline：使用最后一个 RCE checkpoint
    v3_pt_path="${output_dir}/rce_epoch${rce_epochs}.pt"
    recommended_ckpt="rce_epoch${rce_epochs}.pt"
else
    # RCE + GRPO：使用最后一个 GRPO checkpoint
    v3_pt_path="${output_dir}/grpo_epoch${grpo_epochs}.pt"
    recommended_ckpt="grpo_epoch${grpo_epochs}.pt"
fi

# 检查 checkpoint 是否存在
if [ ! -f "$v3_pt_path" ]; then
    echo "⚠️  警告: 推荐的 checkpoint 不存在: ${v3_pt_path}"
    echo "  尝试查找最新的 checkpoint..."
    # 查找最新的 .pt 文件
    latest_pt=$(ls -t "${output_dir}"/*.pt 2>/dev/null | head -1)
    if [ -n "$latest_pt" ] && [ -f "$latest_pt" ]; then
        v3_pt_path="$latest_pt"
        recommended_ckpt=$(basename "$latest_pt")
        echo "  ✓ 找到 checkpoint: ${recommended_ckpt}"
    else
        echo "  ✗ 未找到任何 checkpoint 文件"
        echo "=========================================="
        exit 1
    fi
fi

# 生成 v2format 文件路径
v2format_path="${v3_pt_path%.pt}_v2format.ckpt"

# 检查是否需要重新转换：如果 .pt 文件比 .ckpt 文件新，需要重新转换
if [ -f "${v2format_path}" ] && [ -f "${v3_pt_path}" ]; then
    pt_time=$(stat -c %Y "${v3_pt_path}" 2>/dev/null || echo 0)
    ckpt_time=$(stat -c %Y "${v2format_path}" 2>/dev/null || echo 0)
    if [ "${pt_time}" -gt "${ckpt_time}" ]; then
        echo "⚠️  v2format 文件已存在，但 .pt 文件更新，需要重新转换"
        echo "  删除旧的 v2format 文件..."
        rm -f "${v2format_path}"
    else
        echo "✓ v2format 文件已存在: $(basename ${v2format_path})"
        echo "  直接使用已转换的 checkpoint，跳过转换步骤"
    fi
fi

if [ ! -f "${v2format_path}" ]; then
    echo "v2format 文件不存在，开始转换..."
    echo "  v3 checkpoint: $(basename ${v3_pt_path})"
    echo "  目标路径: $(basename ${v2format_path})"
    
    if [ ! -f "scripts/convert_v3_to_v2_format.py" ]; then
        echo "✗ 错误: 转换脚本不存在: scripts/convert_v3_to_v2_format.py"
        echo "  请确保转换脚本存在"
    else
        if CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/convert_v3_to_v2_format.py --v3_ckpt "${v3_pt_path}"; then
            if [ -f "${v2format_path}" ]; then
                echo "✓ 转换成功: $(basename ${v2format_path})"
            else
                echo "✗ 警告: 转换脚本执行成功，但未找到输出文件"
                echo "  请检查转换脚本的输出"
            fi
        else
            echo "✗ 转换失败（退出码: $?），请检查错误信息"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "✓ V3 训练完成！"
echo "=========================================="
echo "Checkpoint 保存在: ${output_dir}"
echo "  - RCE checkpoints: rce_epoch1.pt ~ rce_epoch${rce_epochs}.pt"
if [ "${grpo_epochs}" -eq 0 ]; then
    echo "  - 推荐使用: ${recommended_ckpt} (RCE-only baseline)"
else
    echo "  - GRPO checkpoints: grpo_epoch1.pt ~ grpo_epoch${grpo_epochs}.pt"
    echo "  - 推荐使用: ${recommended_ckpt}"
fi
if [ -f "${v2format_path}" ]; then
    echo "  - v2format: $(basename ${v2format_path}) (可用于推理)"
fi
echo ""
echo "推理命令（自动转换格式）:"
echo "  bash scripts/inference.sh ${task} ${dataset} ${gpu_id} ${lever_lm} ${sampler} ${beam_model} v3"
echo "=========================================="
