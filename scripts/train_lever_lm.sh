task=${1:-caption}
dataset=${2:-coco2017}
gpu_id=${3:-0}
lever_lm=${4:-query_img_icd_img_text}
sampler=${5:-rand_sampler}
beam_model=${6:-flamingo_3B}
version=${7:-v0}

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
# generate_data.py: model_name_safe = cfg.infer_model.name.replace(".", "_").replace("/", "_").replace(" ", "_")
case "$beam_model" in
    flamingo_3B)
        # flamingo_3B.yaml 中 name: flamingo_3B
        model_name="flamingo_3B"
        ;;
    qwen2.5_vl_3B|qwen2_5_vl_3B)
        # qwen2.5_vl_3B.yaml 中 name: Qwen2.5-VL-3B-Instruct
        # 文件名中会变成 Qwen2_5-VL-3B-Instruct (点替换为下划线)
        model_name="Qwen2_5-VL-3B-Instruct"
        ;;
    *)
        # 默认情况：尝试将 beam_model 转换为文件名格式
        # 替换 . 为 _，/ 为 _，空格为 _
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
# 注意：实际生成的文件使用 max_shot 而不是 few_shot
data_file="${task}-${dataset_name}-${model_name}-${sampler_name}-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:${sample_num}.json"

# 数据文件完整路径（仅用于显示）
data_file_path="./results/${dataset_name}/generated_data/${data_file}"

# 构建检查点保存路径和文件名
# 完整的绝对路径（相对于项目根目录）
# 添加版本目录支持：v0, v1, v2, v3...
checkpoint_dir="./results/${dataset_name}/model_cpk/${version}"
# 文件名格式：模型_采样器_scorer_construct_order_beam_size_few_shot_candidate_num_sample_num
# 将模型名称中的特殊字符替换为下划线，用于文件名
model_name_safe=$(echo "$model_name" | sed 's/-/_/g' | sed 's/\./_/g')
checkpoint_filename="${model_name_safe}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

echo "=========================================="
echo "Training Configuration:"
echo "  Task: ${task}"
echo "  Dataset: ${dataset}"
echo "  GPU ID: ${gpu_id}"
echo "  Beam Model: ${beam_model} → ${model_name}"
echo "  Sampler: ${sampler} → ${sampler_name}"
echo "  Lever LM: ${lever_lm}"
echo "  Sample Num: ${sample_num}"
echo "  Version: ${version}"
if [ "${version}" == "v3" ]; then
    echo "  Training Type: GRPO Reinforcement Learning"
    echo "  RL Data: results/${dataset_name}/generated_data/rl_data_${sampler_name}.json"
    echo "  Query Embeddings: results/${dataset_name}/cache/query_embeddings.pt"
else
    echo "  Training Type: Supervised Fine-Tuning (SFT)"
    echo "  Data file: ${data_file_path}"
fi
echo "=========================================="
if [ "${version}" != "v3" ]; then
    echo "Checkpoint will save to: ${checkpoint_dir}/${checkpoint_filename}_best.ckpt"
    echo "=========================================="
fi

run_train() {
    local data_file=$1
 
    # v3 使用 GRPO 强化学习训练
    if [ "${version}" == "v3" ]; then
        echo "==========Begin: V3 GRPO Reinforcement Learning Training=========="
        
        # 构建 RL 数据路径
        local rl_data_path="./results/${dataset_name}/generated_data/rl_data_${sampler_name}.json"
        local query_emb_path="./results/${dataset_name}/cache/query_embeddings.pt"
        local sft_ckpt_path="./results/${dataset_name}/model_cpk/v2/${checkpoint_filename}_best.ckpt"
        
        # 检查必要文件是否存在
        if [ ! -f "$rl_data_path" ]; then
            echo "错误: RL 数据文件不存在: $rl_data_path"
            echo ""
            echo "说明:"
            echo "  - Embeddings (query_embeddings.pt 和 candidate_embeddings.pt) 是通用的，"
            echo "    只需要生成一次，所有采样器都可以使用"
            echo "  - RL 数据 (rl_data_${sampler_name}.json) 是采样器特定的，"
            echo "    需要为每个采样器分别生成"
            echo ""
            echo "解决方案:"
            echo "  1. 如果 embeddings 还未生成，先运行:"
            echo "     bash scripts/export_embeddings.sh"
            echo ""
            echo "  2. 为当前采样器生成 RL 数据:"
            echo "     bash scripts/generate_rl_data_for_sampler.sh ${sampler} ${beam_model} ${dataset} cuda:${gpu_id}"
            echo ""
            echo "  或者手动运行:"
            echo "     bash scripts/generate_rl_data.sh <sft_ckpt> <beam_data> <output_path> <query_emb> <cand_emb> <device>"
            exit 1
        fi
        
        if [ ! -f "$query_emb_path" ]; then
            echo "错误: Query embeddings 文件不存在: $query_emb_path"
            echo ""
            echo "说明: Embeddings 是通用的，只需要生成一次，所有采样器都可以使用"
            echo ""
            echo "解决方案:"
            echo "  bash scripts/export_embeddings.sh"
            exit 1
        fi
        
        if [ ! -f "./results/${dataset_name}/cache/candidate_embeddings.pt" ]; then
            echo "错误: Candidate embeddings 文件不存在: ./results/${dataset_name}/cache/candidate_embeddings.pt"
            echo ""
            echo "说明: Embeddings 是通用的，只需要生成一次，所有采样器都可以使用"
            echo ""
            echo "解决方案:"
            echo "  bash scripts/export_embeddings.sh"
            exit 1
        fi
        
        if [ ! -f "$sft_ckpt_path" ]; then
            echo "提示: 最佳 checkpoint 不存在: $sft_ckpt_path"
            echo "正在查找其他可用的 v2 checkpoint..."
            # 尝试查找任何 v2 checkpoint
            local v2_dir="./results/${dataset_name}/model_cpk/v2"
            if [ -d "$v2_dir" ]; then
                # 查找最新的 checkpoint（按修改时间排序）
                sft_ckpt_path=$(find "$v2_dir" -name "*${sampler_name}*.ckpt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
                if [ -z "$sft_ckpt_path" ]; then
                    echo "错误: 未找到 v2 checkpoint，请先训练 v2 模型"
                    exit 1
                fi
                echo "✓ 找到可用的 checkpoint: $sft_ckpt_path"
            else
                echo "错误: v2 checkpoint 目录不存在，请先训练 v2 模型"
                exit 1
            fi
        else
            echo "✓ 使用最佳 checkpoint: $sft_ckpt_path"
        fi
        
        # 设置默认 GRPO 参数（可通过环境变量覆盖）
        local rce_epochs=${RCE_EPOCHS:-25}
        local grpo_epochs=${GRPO_EPOCHS:-25}
        local batch_size=${BATCH_SIZE:-1}
        local rce_lr=${RCE_LR:-5e-4}
        local grpo_lr=${GRPO_LR:-5e-6}
        local kl_beta=${KL_BETA:-0.3}
        local reward_alpha=${REWARD_ALPHA:-0.2}
        local reward_beta=${REWARD_BETA:-1.0}
        
        echo "RL Data: $rl_data_path"
        echo "Query Embeddings: $query_emb_path"
        echo "SFT Checkpoint: $sft_ckpt_path"
        echo "Output Directory: $checkpoint_dir"
        echo "=========================================="
        
        # 调用 GRPO 训练
        # 注意：使用 CUDA_VISIBLE_DEVICES 后，PyTorch 只能看到 cuda:0
        # 所以 device 参数应该设置为 cuda:0
        CUDA_VISIBLE_DEVICES=${gpu_id} python -m lever_lm.workflows.grpo_post_train \
            --beam_data "$rl_data_path" \
            --img_emb "$query_emb_path" \
            --sft_ckpt "$sft_ckpt_path" \
            --output_dir "$checkpoint_dir" \
            --rce_epochs ${rce_epochs} \
            --grpo_epochs ${grpo_epochs} \
            --batch_size ${batch_size} \
            --rce_lr ${rce_lr} \
            --grpo_lr ${grpo_lr} \
            --kl_beta ${kl_beta} \
            --reward_alpha ${reward_alpha} \
            --reward_beta ${reward_beta} \
            --device cuda:0
        
        echo "==========End: V3 GRPO Training=========="
        echo ""
        echo "✓ 训练完成！Checkpoint 保存在: $checkpoint_dir"
        echo "  - RCE checkpoints: rce_epoch1.pt ~ rce_epoch25.pt"
        echo "  - GRPO checkpoints: grpo_epoch1.pt ~ grpo_epoch25.pt"
        echo "  - 推荐使用: grpo_epoch25.pt（最终模型）"
        return 0
    fi
    
    # v0, v1, v2, v2_lora 使用 SFT 训练
    # 在 ex_name 中包含采样器和模型名称，避免不同配置互相覆盖
    local ex_name_prefix="main_${task}_${sampler_name}_${model_name_safe}"
    
    # 根据版本选择不同的配置文件
    local train_config="${lever_lm}"
    if [ "${version}" != "v0" ]; then
        # v1, v2 使用对应的配置文件
        # 如果版本包含 _lora，使用对应的 LoRA 配置文件
        if [[ "${version}" == *"_lora" ]]; then
            # 提取基础版本号（如 v2_lora -> v2）
            base_version="${version%_lora}"
            # LoRA 配置文件在对应版本的目录下，格式：lever_lm/v{version}/query_img_text_icd_img_text_lever_lm_lora
            # 但 Hydra 配置路径只需要文件名（不含扩展名），所以是 query_img_text_icd_img_text_lever_lm_lora
            train_config="${lever_lm}_lever_lm_lora"
        else
        train_config="${lever_lm}_${version}"
        fi
    fi

    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${train_config} (version: ${version})==========" 
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
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${train_config} (version: ${version})==========" 
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
