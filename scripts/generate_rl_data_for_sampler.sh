#!/bin/bash
# 为指定采样器生成 RL 数据
# 使用方法: bash scripts/generate_rl_data_for_sampler.sh <sampler> <beam_model> <dataset> <device> [sft_ckpt]
# 示例: bash scripts/generate_rl_data_for_sampler.sh text_sim_sampler qwen2.5_vl_3B okvqa_local cuda:2

sampler=${1:-rand_sampler}
beam_model=${2:-qwen2.5_vl_3B}
dataset=${3:-okvqa_local}
device=${4:-cuda:0}
sft_ckpt=${5:-""}

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

# 根据数据集自动设置 sample_num 和 dataset_name
case "$dataset" in
    okvqa*|OKVQA*)
        sample_num=800
        dataset_name="okvqa"
        task="vqa"
        # VQA 标注文件路径（用于准确率计算）
        # questions 文件需要包含 "questions" 键，annotations 文件需要包含 "annotations" 键
        # RL 数据通常来自训练集，所以优先使用训练集文件
        train_ques_path="./datasets/okvqa/OpenEnded_mscoco_train2014_questions.json"
        train_ann_path="./datasets/okvqa/mscoco_train2014_annotations.json"
        val_ques_path="./datasets/okvqa/OpenEnded_mscoco_val2014_questions.json"
        val_ann_path="./datasets/okvqa/mscoco_val2014_annotations.json"
        ;;
    vqav2*|VQAV2*)
        sample_num=5000
        dataset_name="vqav2"
        task="vqa"
        # VQA 标注文件路径（用于准确率计算）
        val_ques_path="./datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
        val_ann_path="./datasets/vqav2/v2_mscoco_val2014_annotations.json"
        ;;
    coco2017*|COCO2017*)
        sample_num=5000
        dataset_name="coco2017"
        task="caption"
        val_ques_path=""
        val_ann_path=""
        ;;
    *)
        sample_num=5000
        dataset_name="${dataset}"
        task="vqa"
        val_ques_path=""
        val_ann_path=""
        ;;
esac

# 构建文件路径
beam_data_file="${task}-${dataset_name}-${model_name}-${sampler_name}-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:${sample_num}.json"
beam_data_path="./results/${dataset_name}/generated_data/${beam_data_file}"
# RL 数据路径：按 sampler 和 beam_model 分开保存，避免覆盖
rl_data_path="./results/${dataset_name}/generated_data/rl_data_${sampler_name}_${model_name}.json"
query_emb_path="./results/${dataset_name}/cache/query_embeddings.pt"
cand_emb_path="./results/${dataset_name}/cache/candidate_embeddings.pt"

# 如果没有提供 sft_ckpt，尝试自动查找
if [ -z "$sft_ckpt" ]; then
    # 构建 checkpoint 文件名模式
    model_name_safe=$(echo "$model_name" | sed 's/-/_/g' | sed 's/\./_/g')
    checkpoint_filename="${model_name_safe}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"
    sft_ckpt_path="./results/${dataset_name}/model_cpk/v2/${checkpoint_filename}_best.ckpt"
    
    # 检查最佳 checkpoint 是否存在
    if [ -f "$sft_ckpt_path" ]; then
        sft_ckpt="$sft_ckpt_path"
    else
        # 尝试查找任何匹配的 v2 checkpoint
        v2_dir="./results/${dataset_name}/model_cpk/v2"
        if [ -d "$v2_dir" ]; then
            found_ckpt=$(find "$v2_dir" -name "*${sampler_name}*.ckpt" -type f | head -1)
            if [ -n "$found_ckpt" ]; then
                sft_ckpt="$found_ckpt"
                echo "✓ 自动找到 checkpoint: $sft_ckpt"
            else
                echo "错误: 未找到 v2 checkpoint，请手动指定 --sft_ckpt 参数"
                echo "查找模式: $v2_dir/*${sampler_name}*.ckpt"
                exit 1
            fi
        else
            echo "错误: v2 checkpoint 目录不存在: $v2_dir"
            exit 1
        fi
    fi
fi

# 检查必要文件
if [ ! -f "$beam_data_path" ]; then
    echo "错误: Beam 数据文件不存在: $beam_data_path"
    echo "请先运行: bash scripts/generate_data.sh ${task} ${dataset} \"[0]\" ${sampler} ${beam_model}"
    exit 1
fi

if [ ! -f "$query_emb_path" ]; then
    echo "错误: Query embeddings 文件不存在: $query_emb_path"
    echo "请先运行: bash scripts/export_embeddings.sh"
    echo "注意: Embeddings 是通用的，只需要生成一次，所有采样器都可以使用"
    exit 1
fi

if [ ! -f "$cand_emb_path" ]; then
    echo "错误: Candidate embeddings 文件不存在: $cand_emb_path"
    echo "请先运行: bash scripts/export_embeddings.sh"
    echo "注意: Embeddings 是通用的，只需要生成一次，所有采样器都可以使用"
    exit 1
fi

echo "=========================================="
echo "为 ${sampler_name} 生成 RL 数据"
echo "=========================================="
echo "Sampler: ${sampler} → ${sampler_name}"
echo "Beam Model: ${beam_model} → ${model_name}"
echo "Dataset: ${dataset} → ${dataset_name}"
echo "Beam Data: ${beam_data_path}"
echo "RL Data Output: ${rl_data_path}"
echo "SFT Checkpoint: ${sft_ckpt}"
echo "Query Embeddings: ${query_emb_path}"
echo "Candidate Embeddings: ${cand_emb_path}"
echo "Device: ${device}"
echo "=========================================="

# 创建输出目录
output_dir=$(dirname "$rl_data_path")
mkdir -p "$output_dir"

# 执行生成命令
cmd="python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt \"$sft_ckpt\" \
    --beam_data \"$beam_data_path\" \
    --output_path \"$rl_data_path\" \
    --query_emb \"$query_emb_path\" \
    --cand_emb \"$cand_emb_path\" \
    --device \"$device\" \
    --vqa_model \"$beam_model\" \
    --dataset \"$dataset\""

# 如果是 VQA 任务，添加标注文件路径（重要：用于准确率计算）
# RL 数据通常来自训练集，所以优先使用训练集文件
if [ "$task" == "vqa" ]; then
    # 添加训练集文件（如果存在）
    if [ -n "$train_ques_path" ] && [ -n "$train_ann_path" ]; then
        if [ -f "$train_ques_path" ] && [ -f "$train_ann_path" ]; then
            cmd="$cmd --train_ques_path \"$train_ques_path\" --train_ann_path \"$train_ann_path\""
            echo "使用训练集 VQA 标注文件（优先）:"
            echo "  - Questions: $train_ques_path"
            echo "  - Annotations: $train_ann_path"
        fi
    fi
    
    # 添加验证集文件（作为 fallback）
    if [ -n "$val_ques_path" ] && [ -n "$val_ann_path" ]; then
        if [ -f "$val_ques_path" ] && [ -f "$val_ann_path" ]; then
            cmd="$cmd --val_ques_path \"$val_ques_path\" --val_ann_path \"$val_ann_path\""
            echo "使用验证集 VQA 标注文件（fallback）:"
            echo "  - Questions: $val_ques_path"
            echo "  - Annotations: $val_ann_path"
        else
            echo "⚠️  警告: 验证集 VQA 标注文件不存在"
            echo "  - Questions: $val_ques_path"
            echo "  - Annotations: $val_ann_path"
        fi
    fi
    
    # 如果都没有，给出警告
    if [ -z "$train_ques_path" ] && [ -z "$val_ques_path" ]; then
        echo "⚠️  警告: 未提供 VQA 标注文件，将使用 fallback 字符串匹配"
    fi
fi

eval $cmd

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✓ RL 数据生成完成！"
    echo "输出文件: $rl_data_path"
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ RL 数据生成失败"
    echo "=========================================="
    exit 1
fi
