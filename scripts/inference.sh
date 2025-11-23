# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device=${3:-0}
lever_lm=${4:-query_img_icd_img_text}
sampler=${5:-rand_sampler}
beam_model=${6:-flamingo_3B}
version=${7:-v0}

# 固定批量大小为 1，避免批处理时的图像数量不匹配问题
inference_bs=1

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
# 这个映射需要与 train_lever_lm.sh 中的逻辑一致
case "$beam_model" in
    flamingo_3B)
        # flamingo_3B.yaml 中 name: flamingo_3B
        model_name="flamingo_3B"
        infer_model="flamingo_3B"
        ;;
    qwen2.5_vl_3B|qwen2_5_vl_3B)
        # qwen2.5_vl_3B.yaml 中 name: Qwen2.5-VL-3B-Instruct
        # 文件名中会变成 Qwen2_5-VL-3B-Instruct (点替换为下划线)
        model_name="Qwen2_5-VL-3B-Instruct"
        infer_model="qwen2.5_vl_3B"
        ;;
    *)
        # 默认情况：尝试将 beam_model 转换为文件名格式
        # 替换 . 为 _，/ 为 _，空格为 _
        model_name=$(echo "$beam_model" | sed 's/\./_/g' | sed 's/\//_/g' | sed 's/ /_/g')
        infer_model="$beam_model"
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

# 构建检查点目录和文件名模式
# 添加版本目录支持：v0, v1, v2, v3, v4...
checkpoint_dir="./results/${dataset_name}/model_cpk/${version}"
# 将模型名称中的特殊字符替换为下划线，用于文件名匹配（与训练脚本一致）
model_name_safe=$(echo "$model_name" | sed 's/-/_/g' | sed 's/\./_/g')
checkpoint_filename_pattern="${model_name_safe}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

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

# 4. 如果还是没找到，在当前版本目录的父目录中搜索（兼容旧文件）
if [ ${#ckpt_files[@]} -eq 0 ]; then
    parent_checkpoint_dir="./results/${dataset_name}/model_cpk"
    echo "在当前版本目录未找到检查点，尝试在父目录中搜索..."
    if [ -d "$parent_checkpoint_dir" ]; then
        found_files=($(ls -t "${parent_checkpoint_dir}"/*${sampler_name}*.ckpt 2>/dev/null))
        if [ ${#found_files[@]} -gt 0 ]; then
            ckpt_files+=("${found_files[@]}")
            echo "  在 $parent_checkpoint_dir 中找到 ${#found_files[@]} 个匹配的检查点"
        fi
    fi
fi

# 5. 如果还是没找到，在所有数据集目录的版本目录中搜索匹配的采样器检查点（跨数据集查找）
if [ ${#ckpt_files[@]} -eq 0 ]; then
    echo "在当前数据集目录未找到检查点，尝试在所有数据集目录的版本目录中搜索..."
    # 搜索所有数据集目录的版本目录
    for dir in ./results/*/model_cpk/${version}; do
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
    # 在 ex_name 中包含采样器和模型名称，与训练脚本保持一致
    local ex_name_prefix="main_${task}_${sampler_name}_${model_name_safe}"
    
    # 根据版本选择不同的配置文件（与训练脚本保持一致）
    local train_config="${lever_lm}"
    if [ "${version}" != "v0" ]; then
        # v1, v2, v3, v4 使用对应的配置文件
        train_config="${lever_lm}_${version}"
    fi
    
    # 将 GPU 编号转换为 device 格式（cuda:0, cuda:1 等）
    if [[ "$device" =~ ^[0-9]+$ ]]; then
        device_arg="cuda:${device}"
    else
        # 如果已经是 cuda:0 格式，直接使用
        device_arg="${device}"
    fi
    
    echo "=========================================="
    echo "Inference Configuration:"
    echo "  Task: ${task}"
    echo "  Dataset: ${dataset}"
    echo "  GPU ID: ${device} → ${device_arg}"
    echo "  Beam Model: ${beam_model} → ${model_name} (infer_model: ${infer_model})"
    echo "  Sampler: ${sampler} → ${sampler_name}"
    echo "  Lever LM: ${lever_lm}"
    echo "  Train Config: ${train_config} (version: ${version})"
    echo "  Sample Num: ${sample_num}"
    echo "  Version: ${version}"
    echo "=========================================="
    
    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${lever_lm}-LeverLM (version: ${version}, config: ${train_config})==========" 
        # 使用环境变量传递检查点路径，避免 Hydra 解析路径中的特殊字符
        # 根据 beam_model 选择对应的推理模型
        python icl_inference.py train="${train_config}" \
                                ex_name="${ex_name_prefix}_${lever_lm}" \
                                dataset=${dataset} \
                                task=${task} \
                                device=${device_arg} \
                                inference_bs=${inference_bs} \
                                test_lever_lm=true \
                                infer_model=${infer_model} \
                                infer_model.load_from_local=false

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_freeze_adapter_non_norm_${lever_lm}-LeverLM (version: ${version}, config: ${train_config})==========" 
        # 使用环境变量传递检查点路径，避免 Hydra 解析路径中的特殊字符
        # 根据 beam_model 选择对应的推理模型
        python icl_inference.py train="${train_config}" \
                                ex_name="${ex_name_prefix}_freeze_adapter_non_norm_${lever_lm}" \
                                dataset=${dataset} \
                                task=${task} \
                                device=${device_arg} \
                                inference_bs=${inference_bs} \
                                test_lever_lm=true \
                                train.lever_lm.norm=false \
                                train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                                train.lever_lm.adapter=true \
                                infer_model=${infer_model} \
                                infer_model.load_from_local=false
    fi
    
    # 清理环境变量
    unset LEVER_LM_CHECKPOINT_PATH
}

run_inference