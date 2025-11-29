#!/bin/bash
# 批量从所有 checkpoint 文件中提取 LoRA 权重

BASE_DIR="/mnt/share/yiyun/Projects/Lever-Plus/results/okvqa/model_cpk/v2_lora"
EXTRACT_SCRIPT="/mnt/share/yiyun/Projects/Lever-Plus/extract_lora_from_checkpoint.py"

# 定义所有需要提取的配置
declare -a configs=(
    "flamingo_3B_RandSampler"
    "flamingo_3B_TextSimSampler"
    "flamingo_3B_ImgSimSampler"
    "flamingo_3B_MixSampler"
    "Qwen2_5_VL_3B_Instruct_RandSampler"
    "Qwen2_5_VL_3B_Instruct_TextSimSampler"
    "Qwen2_5_VL_3B_Instruct_ImgSimSampler"
    "Qwen2_5_VL_3B_Instruct_MixSampler"
)

echo "=========================================="
echo "批量提取 LoRA checkpoint"
echo "=========================================="

for config in "${configs[@]}"; do
    echo ""
    echo "处理配置: $config"
    
    # 查找该配置的最佳 checkpoint（优先选择 val loss 最低的）
    checkpoint=$(find "$BASE_DIR" -name "${config}_*.ckpt" -type f | grep -v "\-v1.ckpt" | sort | head -1)
    
    if [ -z "$checkpoint" ]; then
        echo "  ⚠ 未找到 checkpoint 文件"
        continue
    fi
    
    # 从文件名提取配置名称（用于输出目录）
    # 例如: flamingo_3B_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=18_train=28.59430_val=22.39737.ckpt
    # 提取: flamingo_3B_RandSampler
    config_name=$(basename "$checkpoint" | sed 's/_infoscore.*//')
    
    # 输出目录
    output_dir="${BASE_DIR}/lora/${config_name}"
    
    echo "  Checkpoint: $(basename $checkpoint)"
    echo "  输出目录: $output_dir"
    
    # 执行提取
    python "$EXTRACT_SCRIPT" "$checkpoint" "$output_dir"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 提取成功"
    else
        echo "  ✗ 提取失败"
    fi
done

echo ""
echo "=========================================="
echo "批量提取完成！"
echo "=========================================="
echo ""
echo "提取的 LoRA checkpoint 保存在:"
echo "  ${BASE_DIR}/lora/<配置名称>/"
echo ""

