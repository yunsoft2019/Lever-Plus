task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}
sampler=${4:-rand_sampler}
beam_model=${5:-flamingo_3B}
use_lora=${6:-false}  # 是否使用LoRA，默认为false
lora_checkpoint_path=${7:-""}  # LoRA checkpoint 路径（可选）

# 解析 gpu_ids：将 "[0]" 或 "[0,1]" 转换为 Hydra 列表格式 [0] 或 [0,1]
# 移除外层引号和方括号，然后重新格式化为 Hydra 列表格式
if [[ "$gpu_ids" =~ ^\"?\[.*\]\"?$ ]]; then
    # 移除外层引号和方括号
    gpu_ids_clean=$(echo "$gpu_ids" | sed 's/^"*\[//' | sed 's/\]"*$//')
    # 转换为 Hydra 列表格式 [0] 或 [0,1]
    gpu_ids_hydra="[${gpu_ids_clean}]"
else
    # 如果格式不对，使用默认值
    gpu_ids_hydra="[0]"
fi

# 根据数据集自动设置 sample_num
case "$dataset" in
    okvqa*|OKVQA*)
        sample_num=800  # OKVQA 数据集较小(9k)，使用 800
        ;;
    vqav2*|VQAV2*|vqa*)
        sample_num=5000  # VQAv2 数据集大(443k)，使用 5000
        ;;
    coco*|COCO*)
        sample_num=5000  # COCO 数据集使用 5000
        ;;
    *)
        sample_num=5000  # 默认使用 5000
        ;;
esac

echo "Dataset: $dataset, Sampler: $sampler, Beam Model: $beam_model, Sample num: $sample_num"
echo "GPU IDs: $gpu_ids_hydra"
echo "Use LoRA: $use_lora"

# 如果 use_lora=true 但没有提供 lora_checkpoint_path，使用默认路径
if [ "$use_lora" = "true" ] && [ -z "$lora_checkpoint_path" ]; then
    # 处理数据集名称，与 train_lora.sh 保持一致
    # 去掉 _local 后缀（如果存在）
    dataset_name=$(echo "$dataset" | sed 's/_local$//')
    # 默认 LoRA checkpoint 路径：./results/{dataset_name}/model_cpk/v2_lora/lora
    lora_checkpoint_path="./results/${dataset_name}/model_cpk/v2_lora/lora"
    echo "LoRA enabled but path not provided, using default: $lora_checkpoint_path"
fi

if [ -n "$lora_checkpoint_path" ]; then
    echo "LoRA Checkpoint Path: $lora_checkpoint_path"
fi

# 支持通过环境变量或参数覆盖 sample_num（用于测试）
test_sample_num=${TEST_SAMPLE_NUM:-${sample_num}}

# 注意：gpu_ids 参数不能加引号，否则 Hydra 会将其解析为字符串而不是列表
python generate_data.py beam_size=5 \
                        cand_num=64 \
                        sample_num=${test_sample_num} \
                        gpu_ids=${gpu_ids_hydra} \
                        task=${task} \
                        dataset=${dataset} \
                        sampler=${sampler} \
                        infer_model=${beam_model} \
                        few_shot_num=2 \
                        use_lora=${use_lora} \
                        lora_checkpoint_path="${lora_checkpoint_path}"