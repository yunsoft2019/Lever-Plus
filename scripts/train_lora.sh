#!/bin/bash

# 训练 LoRA adapter 脚本
# 用途：专门用于训练 CLIP 的 LoRA adapter，生成 LoRA checkpoint 用于束搜索
# 
# 参数说明：
#   $1: task (默认: caption)
#   $2: dataset (默认: coco2017)
#   $3: gpu_id (默认: 0)
#   $4: lever_lm (默认: query_img_icd_img_text)
#   $5: sampler (默认: rand_sampler)
#   $6: beam_model (默认: flamingo_3B)
#
# 示例：
#   bash scripts/train_lora.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B
#   bash scripts/train_lora.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

task=${1:-caption}
dataset=${2:-coco2017}
gpu_id=${3:-0}
lever_lm=${4:-query_img_icd_img_text}
sampler=${5:-rand_sampler}
beam_model=${6:-flamingo_3B}

# 固定使用 v2_lora 版本（基于 v2 架构，不含排序学习）
version="v2_lora"

# 将 GPU 编号转换为 PyTorch Lightning 的 devices 格式
# 例如: 0 -> [0], 1 -> [1]
devices_arg="[${gpu_id}]"

# 将 sampler 转换为大驼峰格式（与实际生成的文件名匹配）
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
        # 如果已经是大驼峰格式，直接使用
        sampler_name="${sampler}"
        ;;
esac

# 将 beam_model 映射到文件名中使用的模型名称
# 这个映射需要与 generate_data.py 中的 model_name_safe 逻辑一致
case "$beam_model" in
    flamingo_3B)
        model_name="flamingo_3B"
        ;;
    qwen2.5_vl_3B|qwen2_5_vl_3B)
        model_name="Qwen2_5-VL-3B-Instruct"
        ;;
    *)
        # 默认情况：尝试将 beam_model 转换为文件名格式
        model_name=$(echo "$beam_model" | sed 's/\./_/g' | sed 's/\//_/g' | sed 's/ /_/g')
        echo "Warning: Unknown beam_model '$beam_model', using converted name: $model_name"
        ;;
esac

# 根据数据集自动设置 sample_num 和 dataset_name
case "$dataset" in
    okvqa*|OKVQA*)
        sample_num=800
        dataset_name="okvqa"
        ;;
    vqav2*|VQAV2*)
        sample_num=5000
        dataset_name="vqav2"
        ;;
    coco2017*|COCO2017*)
        sample_num=5000
        dataset_name="coco2017"
        ;;
    coco2014*|COCO2014*)
        sample_num=5000
        dataset_name="coco2014"
        ;;
    *)
        sample_num=5000
        dataset_name="${dataset}"
        ;;
esac

# 构建数据文件名（使用 sampler_name 大驼峰格式）
# 注意：只包含文件名，不包含路径，因为 train.py 会自己拼接路径
data_file="${task}-${dataset_name}-${model_name}-${sampler_name}-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:${sample_num}.json"

# 数据文件完整路径（仅用于显示）
data_file_path="./results/${dataset_name}/generated_data/${data_file}"

# 构建检查点保存路径和文件名
# LoRA checkpoint 会保存在 results/{dataset}/model_cpk/v2_lora/lora/ 目录下
checkpoint_dir="./results/${dataset_name}/model_cpk/${version}"
# 文件名格式：模型_采样器_scorer_construct_order_beam_size_few_shot_candidate_num_sample_num
model_name_safe=$(echo "$model_name" | sed 's/-/_/g' | sed 's/\./_/g')
checkpoint_filename="${model_name_safe}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

echo "=========================================="
echo "LoRA Training Configuration:"
echo "  Task: ${task}"
echo "  Dataset: ${dataset}"
echo "  GPU ID: ${gpu_id}"
echo "  Beam Model: ${beam_model} → ${model_name}"
echo "  Sampler: ${sampler} → ${sampler_name}"
echo "  Lever LM: ${lever_lm}"
echo "  Sample Num: ${sample_num}"
echo "  Version: ${version} (固定，用于训练 LoRA adapter)"
echo "=========================================="
echo "Data file: ${data_file_path}"
echo "Checkpoint will save to: ${checkpoint_dir}/${checkpoint_filename}_best.ckpt"
echo "LoRA checkpoint will save to: ${checkpoint_dir}/lora/"
echo "=========================================="

run_train() {
    local data_file=$1
 
    # 在 ex_name 中包含采样器和模型名称，避免不同配置互相覆盖
    local ex_name_prefix="main_${task}_${sampler_name}_${model_name_safe}"
    
    # 固定使用 v2 的 LoRA 配置文件
    # LoRA 配置文件格式：lever_lm/v2/query_img_text_icd_img_text_lever_lm_lora
    # 但 Hydra 配置路径只需要文件名（不含扩展名），所以是 query_img_text_icd_img_text_lever_lm_lora
    train_config="${lever_lm}_lever_lm_lora"

    if [ "${task}" == "vqa" ]; then
        echo "==========Begin LoRA Training: ${ex_name_prefix}-LeverLM: ${train_config}==========" 
        python train.py train="${train_config}" \
                        data_files="${data_file}" \
                        trainer_args.max_epochs=20 \
                        trainer_args.val_check_interval=0.25 \
                        ex_name="${ex_name_prefix}_${lever_lm}" \
                        dirpath="${checkpoint_dir}" \
                        +checkpoint_filename="${checkpoint_filename}" \
                        trainer_args.devices=${devices_arg} \
                        dataset=${dataset} \
                        task=${task} \
                        lr=1e-4 \
                        +use_wandb=false \
                        +use_simple_logger=true 2>&1 | grep -v "UserWarning\|FutureWarning\|DeprecationWarning\|Found.*module.*eval mode"

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin LoRA Training: ${ex_name_prefix}-LeverLM: ${train_config}==========" 
        python train.py train="${train_config}" \
                        data_files="${data_file}" \
                        trainer_args.max_epochs=20 \
                        trainer_args.val_check_interval=0.25 \
                        ex_name="${ex_name_prefix}_${lever_lm}" \
                        trainer_args.devices=${devices_arg} \
                        dataset=${dataset} \
                        task=${task} \
                        lr=1e-4 \
                        train.lever_lm.norm=false \
                        train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                        train.lever_lm.adapter=true \
                        +use_wandb=false \
                        +use_simple_logger=true 2>&1 | grep -v "UserWarning\|FutureWarning\|DeprecationWarning\|Found.*module.*eval mode"
    fi
}

run_train "${data_file}"

