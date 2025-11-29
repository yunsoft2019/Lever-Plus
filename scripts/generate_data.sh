task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}
sampler=${4:-rand_sampler}
beam_model=${5:-flamingo_3B}
# 注意：束搜索不使用 LoRA，LoRA 只在训练时使用（见 train_lever_lm.sh v2_lora 版本）

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
echo "Note: Beam search uses original VLM model (no LoRA). LoRA is only used during training (v2_lora version)."

# 支持通过环境变量或参数覆盖 sample_num（用于测试）
test_sample_num=${TEST_SAMPLE_NUM:-${sample_num}}

# 注意：gpu_ids 参数不能加引号，否则 Hydra 会将其解析为字符串而不是列表
# 束搜索不使用 LoRA，所以 use_lora 固定为 false
python generate_data.py beam_size=5 \
                        cand_num=64 \
                        sample_num=${test_sample_num} \
                        gpu_ids=${gpu_ids_hydra} \
                        task=${task} \
                        dataset=${dataset} \
                        sampler=${sampler} \
                        infer_model=${beam_model} \
                        few_shot_num=4 \
                        use_lora=false \
                        lora_checkpoint_path=""