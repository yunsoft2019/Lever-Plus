#!/bin/bash
# 根据参数自动加载对应的检查点并继续训练

task=${1:-vqa}
dataset=${2:-okvqa_local}
device_num=${3:-1}
lever_lm=${4:-query_img_text_icd_img_text}
sampler=${5:-rand_sampler}
version=${6:-v0}

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

# 根据数据集自动设置 sample_num、model_name 和 dataset_name
case "$dataset" in
    okvqa*|OKVQA*)
        sample_num=800
        model_name="flamingo_3B"
        dataset_name="okvqa"
        ;;
    vqav2*|VQAV2*)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="vqav2"
        ;;
    coco2017*|COCO2017*)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="coco2017"
        ;;
    coco2014*|COCO2014*)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="coco2014"
        ;;
    *)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="${dataset}"
        ;;
esac

# 构建检查点目录和文件名模式
# 添加版本目录支持：v0, v1, v2, v3, v4...
checkpoint_dir="./results/${dataset_name}/model_cpk/${version}"
checkpoint_filename_pattern="${model_name}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

# 检查目录是否存在
if [ ! -d "$checkpoint_dir" ]; then
    echo "错误: 检查点目录不存在: $checkpoint_dir"
    exit 1
fi

# 查找匹配的检查点文件（支持多种可能的文件名格式）
# 1. 查找精确匹配的文件名（可能包含 epoch 信息）
ckpt_files=($(ls -1 "${checkpoint_dir}/${checkpoint_filename_pattern}"*.ckpt 2>/dev/null))

# 如果没找到，尝试查找包含采样器名称的所有文件
if [ ${#ckpt_files[@]} -eq 0 ]; then
    ckpt_files=($(ls -1 "${checkpoint_dir}"/*${sampler_name}*.ckpt 2>/dev/null))
fi

# 如果还是没找到，报错
if [ ${#ckpt_files[@]} -eq 0 ]; then
    echo "错误: 在 $checkpoint_dir 中未找到匹配的检查点文件"
    echo "查找模式: ${checkpoint_filename_pattern}*.ckpt 或 *${sampler_name}*.ckpt"
    echo ""
    echo "可用的检查点文件:"
    ls -1 "${checkpoint_dir}"/*.ckpt 2>/dev/null | head -10
    exit 1
fi

# 如果有多个匹配的检查点，选择最新的（按修改时间排序）
if [ ${#ckpt_files[@]} -gt 1 ]; then
    echo "找到 ${#ckpt_files[@]} 个匹配的检查点文件，将使用最新的:"
    for ckpt in "${ckpt_files[@]}"; do
        echo "  - $(basename $ckpt)"
    done
    # 按修改时间排序，取最新的
    ckpt_path=$(ls -t "${ckpt_files[@]}" 2>/dev/null | head -1)
else
    ckpt_path="${ckpt_files[0]}"
fi

ckpt_filename=$(basename "$ckpt_path")

# 构建数据文件名
data_file="${task}-${dataset_name}-${model_name}-${sampler_name}-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:${sample_num}.json"

# 构建检查点保存路径和文件名
# 添加版本目录支持：v0, v1, v2, v3, v4...
checkpoint_save_dir="./results/${dataset_name}/model_cpk/${version}"
checkpoint_filename="${checkpoint_filename_pattern}_resume"

echo "=========================================="
echo "从检查点继续训练"
echo "=========================================="
echo "任务: ${task}"
echo "数据集: ${dataset} (${dataset_name})"
echo "采样器: ${sampler} (${sampler_name})"
echo "检查点: ${ckpt_filename}"
echo "检查点路径: ${ckpt_path}"
echo "数据文件: ${data_file}"
echo "保存目录: ${checkpoint_save_dir}"
echo "新检查点文件名: ${checkpoint_filename}"
echo "版本: ${version}"
echo "=========================================="

# 在 ex_name 中包含采样器名称
ex_name_prefix="resume_${task}_${sampler_name}"

run_train() {
    # 使用环境变量传递检查点路径，避免 Hydra 解析路径中的特殊字符
    export RESUME_CKPT_PATH="${ckpt_path}"
    
    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${lever_lm}==========" 
        python train.py train="${lever_lm}" \
                        data_files="${data_file}" \
                        trainer_args.max_epochs=20 \
                        trainer_args.val_check_interval=0.25 \
                        ex_name="${ex_name_prefix}_${lever_lm}" \
                        dirpath="${checkpoint_save_dir}" \
                        +checkpoint_filename="${checkpoint_filename}" \
                        trainer_args.devices=${device_num} \
                        dataset=${dataset} \
                        task=${task} \
                        +use_wandb=false \
                        +use_simple_logger=true \
                        2>&1 | grep -v "UserWarning\|FutureWarning\|DeprecationWarning\|Found.*module.*eval mode"

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${lever_lm}==========" 
        python train.py train="${lever_lm}" \
                        data_files="${data_file}" \
                        trainer_args.max_epochs=20 \
                        trainer_args.val_check_interval=0.25 \
                        ex_name="${ex_name_prefix}_${lever_lm}" \
                        dirpath="${checkpoint_save_dir}" \
                        +checkpoint_filename="${checkpoint_filename}" \
                        trainer_args.devices=${device_num} \
                        dataset=${dataset} \
                        task=${task} \
                        train.lever_lm.norm=false \
                        train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                        train.lever_lm.adapter=true \
                        +use_wandb=false \
                        +use_simple_logger=true \
                        2>&1 | grep -v "UserWarning\|FutureWarning\|DeprecationWarning\|Found.*module.*eval mode"
    fi
    
    # 清理环境变量
    unset RESUME_CKPT_PATH
}

run_train

