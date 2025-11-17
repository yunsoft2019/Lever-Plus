# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device=${3:-0}
lever_lm=${4:-query_img_icd_img_text}
sampler=${5:-rand_sampler}
inference_bs=${6:-4}

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
checkpoint_dir="./results/${dataset_name}/model_cpk"
checkpoint_filename_pattern="${model_name}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

# 查找匹配的检查点文件
# 1. 优先在当前数据集目录查找 resume 版本
ckpt_files=($(ls -t "${checkpoint_dir}/${checkpoint_filename_pattern}"*resume*.ckpt 2>/dev/null))

# 2. 如果没找到，在当前数据集目录查找原始版本
if [ ${#ckpt_files[@]} -eq 0 ]; then
    ckpt_files=($(ls -t "${checkpoint_dir}/${checkpoint_filename_pattern}"*.ckpt 2>/dev/null))
fi

# 3. 如果还是没找到，在当前数据集目录查找包含采样器名称的所有文件
if [ ${#ckpt_files[@]} -eq 0 ]; then
    ckpt_files=($(ls -t "${checkpoint_dir}"/*${sampler_name}*.ckpt 2>/dev/null))
fi

# 4. 如果还是没找到，在所有数据集目录中搜索匹配的采样器检查点（跨数据集查找）
if [ ${#ckpt_files[@]} -eq 0 ]; then
    echo "在当前数据集目录未找到检查点，尝试在所有数据集目录中搜索..."
    # 搜索所有数据集目录
    for dir in ./results/*/model_cpk; do
        if [ -d "$dir" ]; then
            found_files=($(ls -t "$dir"/*${sampler_name}*.ckpt 2>/dev/null))
            if [ ${#found_files[@]} -gt 0 ]; then
                ckpt_files+=("${found_files[@]}")
                echo "  在 $dir 中找到 ${#found_files[@]} 个匹配的检查点"
            fi
        fi
    done
fi

# 如果找到了检查点，使用最新的
if [ ${#ckpt_files[@]} -gt 0 ]; then
    ckpt_path=$(ls -t "${ckpt_files[@]}" 2>/dev/null | head -1)
    ckpt_filename=$(basename "$ckpt_path")
    echo "找到检查点: ${ckpt_filename}"
    echo "检查点路径: ${ckpt_path}"
    # 使用环境变量传递检查点路径，避免 Hydra 解析路径中的特殊字符
    export LEVER_LM_CHECKPOINT_PATH="${ckpt_path}"
else
    echo "警告: 在所有数据集目录中未找到匹配的检查点文件"
    echo "查找模式: ${checkpoint_filename_pattern}*.ckpt 或 *${sampler_name}*.ckpt"
    echo "将使用默认的检查点查找逻辑（基于 ex_name）"
    unset LEVER_LM_CHECKPOINT_PATH
fi

run_inference() {
    # 在 ex_name 中包含采样器名称，与训练脚本保持一致
    local ex_name_prefix="main_${task}_${sampler_name}"
    
    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${lever_lm}-LeverLM==========" 
        # 使用环境变量传递检查点路径，避免 Hydra 解析路径中的特殊字符
        # 使用 flamingo_3B 模型（OpenFlamingo-3B-vitl-mpt1b）
        # 如果本地没有模型文件，允许从网络下载（覆盖 load_from_local 设置）
        python icl_inference.py train="${lever_lm}" \
                                ex_name="${ex_name_prefix}_${lever_lm}" \
                                dataset=${dataset} \
                                task=${task} \
                                inference_bs=${inference_bs} \
                                test_lever_lm=true \
                                infer_model=flamingo_3B \
                                infer_model.load_from_local=false

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_freeze_adapter_non_norm_${lever_lm}-LeverLM==========" 
        # 使用环境变量传递检查点路径，避免 Hydra 解析路径中的特殊字符
        # 使用 flamingo_3B 模型（OpenFlamingo-3B-vitl-mpt1b）
        # 如果本地没有模型文件，允许从网络下载（覆盖 load_from_local 设置）
        python icl_inference.py train="${lever_lm}" \
                                ex_name="${ex_name_prefix}_freeze_adapter_non_norm_${lever_lm}" \
                                dataset=${dataset} \
                                task=${task} \
                                inference_bs=${inference_bs} \
                                test_lever_lm=true \
                                train.lever_lm.norm=false \
                                train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                                train.lever_lm.adapter=true \
                                infer_model=flamingo_3B \
                                infer_model.load_from_local=false
    fi
    
    # 清理环境变量
    unset LEVER_LM_CHECKPOINT_PATH
}

run_inference