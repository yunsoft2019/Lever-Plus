# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device=${3:-0}
lever_lm=${4:-query_img_icd_img_text}
sampler=${5:-rand_sampler}
beam_model=${6:-flamingo_3B}
version=${7:-v0}
test_data_num=${8:-100}  # 默认 100 条数据，设置为 -1 表示使用全部数据

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
# 添加版本目录支持：v0, v1, v2, v3...
# v3 使用采样器和模型特定的目录（如 v3_RandSampler_Qwen2_5-VL-3B-Instruct）

# ========================================
# 优先使用用户设置的 LEVER_LM_CHECKPOINT_PATH
# ========================================
USER_SET_CHECKPOINT=""
if [ -n "${LEVER_LM_CHECKPOINT_PATH}" ]; then
    echo "=========================================="
    echo "使用用户设置的 checkpoint 路径:"
    echo "  LEVER_LM_CHECKPOINT_PATH=${LEVER_LM_CHECKPOINT_PATH}"
    if [ -f "${LEVER_LM_CHECKPOINT_PATH}" ]; then
        echo "  ✓ 文件存在"
    else
        echo "  ✗ 警告: 文件不存在!"
    fi
    echo "=========================================="
    # 标记用户已设置 checkpoint，跳过自动查找
    USER_SET_CHECKPOINT="true"
    ckpt_path="${LEVER_LM_CHECKPOINT_PATH}"
fi

if [ "${version}" == "v3" ]; then
    # 优先查找新格式目录（包含 model_name）
    checkpoint_dir="./results/${dataset_name}/model_cpk/v3_${sampler_name}_${model_name}"
    # 如果新格式目录不存在，回退到旧格式目录（仅包含 sampler_name）
    if [ ! -d "$checkpoint_dir" ]; then
        checkpoint_dir_old="./results/${dataset_name}/model_cpk/v3_${sampler_name}"
        if [ -d "$checkpoint_dir_old" ]; then
            echo "注意: 使用旧格式目录 ${checkpoint_dir_old}"
            checkpoint_dir="$checkpoint_dir_old"
        fi
    fi
else
    checkpoint_dir="./results/${dataset_name}/model_cpk/${version}"
fi
# 将模型名称中的特殊字符替换为下划线，用于文件名匹配（与训练脚本一致）
model_name_safe=$(echo "$model_name" | sed 's/-/_/g' | sed 's/\./_/g')
checkpoint_filename_pattern="${model_name_safe}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

# v3 使用 .pt 格式的 checkpoint（GRPO checkpoint）
# 只有在用户没有设置 LEVER_LM_CHECKPOINT_PATH 时才自动查找
if [ -n "${USER_SET_CHECKPOINT}" ]; then
    # 用户已设置 checkpoint，跳过自动查找
    echo "跳过自动查找 checkpoint（用户已设置）"
elif [ "${version}" == "v3" ]; then
    # v3 checkpoint 格式：优先使用 v2format.ckpt（通过 v2 推理流程），其次是 grpo_epoch*.pt
    echo "=========================================="
    echo "查找 v3 GRPO checkpoint..."
    echo "查找目录: ${checkpoint_dir}"
    echo "=========================================="
    
    # 1. 先查找最新的 grpo_epoch*.pt 文件
    grpo_files=($(ls -t "${checkpoint_dir}"/grpo_epoch*.pt 2>/dev/null))
    
    # 2. 如果没找到 grpo，查找 rce_epoch*.pt 文件
    if [ ${#grpo_files[@]} -eq 0 ]; then
        echo "未找到 grpo checkpoint，尝试查找 rce checkpoint..."
        rce_files=($(ls -t "${checkpoint_dir}"/rce_epoch*.pt 2>/dev/null))
        if [ ${#rce_files[@]} -gt 0 ]; then
            grpo_files=("${rce_files[@]}")
        fi
    fi
    
    # 3. 如果还是没找到，查找所有 .pt 文件
    if [ ${#grpo_files[@]} -eq 0 ]; then
        echo "未找到 grpo/rce checkpoint，查找所有 .pt 文件..."
        grpo_files=($(ls -t "${checkpoint_dir}"/*.pt 2>/dev/null))
    fi
    
    # 4. 跨数据集查找（优先新格式目录，然后旧格式目录）
    if [ ${#grpo_files[@]} -eq 0 ]; then
        echo "在当前数据集目录未找到 checkpoint，尝试在所有数据集目录的 v3 目录中搜索..."
        # 先查找新格式目录（包含 model_name）
        for dir in ./results/*/model_cpk/v3_${sampler_name}_${model_name}; do
            if [ -d "$dir" ]; then
                found_files=($(ls -t "$dir"/grpo_epoch*.pt 2>/dev/null))
                if [ ${#found_files[@]} -gt 0 ]; then
                    grpo_files+=("${found_files[@]}")
                    echo "  在 $dir 中找到 ${#found_files[@]} 个 grpo checkpoint"
                fi
            fi
        done
        # 如果新格式目录没找到，查找旧格式目录
        if [ ${#grpo_files[@]} -eq 0 ]; then
            for dir in ./results/*/model_cpk/v3_${sampler_name}; do
                if [ -d "$dir" ]; then
                    found_files=($(ls -t "$dir"/grpo_epoch*.pt 2>/dev/null))
                    if [ ${#found_files[@]} -gt 0 ]; then
                        grpo_files+=("${found_files[@]}")
                        echo "  在 $dir (旧格式) 中找到 ${#found_files[@]} 个 grpo checkpoint"
                    fi
                fi
            done
        fi
    fi
    
    # 使用最新的 .pt checkpoint
    if [ ${#grpo_files[@]} -gt 0 ]; then
        v3_pt_path=$(ls -t "${grpo_files[@]}" 2>/dev/null | head -1)
        v3_pt_filename=$(basename "$v3_pt_path")
        echo "✓ 找到 v3 checkpoint: ${v3_pt_filename}"
        echo "  检查点路径: ${v3_pt_path}"
        
        # 检查是否已有对应的 v2format.ckpt 文件
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
            ckpt_path="${v2format_path}"
            fi
        fi
        
        if [ ! -f "${v2format_path}" ]; then
            echo "=========================================="
            echo "自动转换 v3 checkpoint 为 v2 格式..."
            echo "=========================================="
            echo "  v3 checkpoint: ${v3_pt_path}"
            echo "  目标路径: ${v2format_path}"
            
            # 检查转换脚本是否存在
            if [ ! -f "scripts/convert_v3_to_v2_format.py" ]; then
                echo "✗ 错误: 转换脚本不存在: scripts/convert_v3_to_v2_format.py"
                echo "  使用原始 .pt 文件（可能无法正常推理）"
                ckpt_path="${v3_pt_path}"
                export LEVER_LM_CHECKPOINT_VERSION="v3"
            else
                # 执行转换
                if python scripts/convert_v3_to_v2_format.py --v3_ckpt "${v3_pt_path}"; then
                    # 检查转换是否成功
            if [ -f "${v2format_path}" ]; then
                echo "✓ 转换成功: $(basename ${v2format_path})"
                ckpt_path="${v2format_path}"
            else
                        echo "✗ 警告: 转换脚本执行成功，但未找到输出文件"
                        echo "  使用原始 .pt 文件（可能无法正常推理）"
                        ckpt_path="${v3_pt_path}"
                        export LEVER_LM_CHECKPOINT_VERSION="v3"
                    fi
                else
                    echo "✗ 转换失败（退出码: $?），使用原始 .pt 文件（可能无法正常推理）"
                ckpt_path="${v3_pt_path}"
                export LEVER_LM_CHECKPOINT_VERSION="v3"
                fi
            fi
        fi
        
        export LEVER_LM_CHECKPOINT_PATH="${ckpt_path}"
    else
        echo "警告: 未找到 v3 checkpoint (.pt 文件)"
        echo "查找目录: ${checkpoint_dir}"
        echo "查找模式: grpo_epoch*.pt 或 rce_epoch*.pt"
        unset LEVER_LM_CHECKPOINT_PATH
        unset LEVER_LM_CHECKPOINT_VERSION
    fi
else
    # v0, v1, v2, v2_lora 使用 .ckpt 格式的 checkpoint
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
        echo "✓ 找到检查点: ${ckpt_filename}"
        echo "  检查点路径: ${ckpt_path}"
        # 使用环境变量传递检查点路径，避免 Hydra 解析路径中的特殊字符
        export LEVER_LM_CHECKPOINT_PATH="${ckpt_path}"
    else
        echo "警告: 在所有数据集目录中未找到匹配的检查点文件"
        echo "查找模式: ${checkpoint_filename_pattern}*.ckpt 或 *${sampler_name}*.ckpt"
        echo "将使用默认的检查点查找逻辑（基于 ex_name）"
        unset LEVER_LM_CHECKPOINT_PATH
    fi
fi

run_inference() {
    # 在 ex_name 中包含采样器和模型名称，与训练脚本保持一致
    local ex_name_prefix="main_${task}_${sampler_name}_${model_name_safe}"
    
    # 根据版本选择不同的配置文件（与训练脚本保持一致）
    local train_config="${lever_lm}"
    if [ "${version}" != "v0" ]; then
        # 如果版本包含 _lora，使用对应的 LoRA 配置文件
        if [[ "${version}" == *"_lora" ]]; then
            # LoRA 配置文件格式：query_img_text_icd_img_text_lever_lm_lora
            train_config="${lever_lm}_lever_lm_lora"
        elif [ "${version}" == "v3" ]; then
            # v3 使用 v2 的配置文件（因为转换后的 checkpoint 是 v2 格式）
            train_config="${lever_lm}_v2"
        else
            # v1, v2 使用对应的配置文件
            train_config="${lever_lm}_${version}"
        fi
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
    echo "  Test Data Num: ${test_data_num} (${test_data_num} == -1 表示使用全部数据)"
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
                                test_data_num=${test_data_num} \
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
                                test_data_num=${test_data_num} \
                                test_lever_lm=true \
                                train.lever_lm.norm=false \
                                train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                                train.lever_lm.adapter=true \
                                infer_model="${infer_model}" \
                                infer_model.load_from_local=false
    fi
    
    # 清理环境变量
    unset LEVER_LM_CHECKPOINT_PATH
    unset LEVER_LM_CHECKPOINT_VERSION
}

run_inference