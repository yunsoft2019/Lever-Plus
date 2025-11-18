#!/bin/bash
# 批量执行推理脚本
# 用法: bash scripts/batch_inference.sh [task] [dataset] [device] [lever_lm] [inference_bs]
# 示例: bash scripts/batch_inference.sh vqa okvqa_local 0 query_img_text_icd_img_text 64

# 设置默认值
task=${1:-vqa}
dataset=${2:-okvqa_local}
device=${3:-0}
lever_lm=${4:-query_img_text_icd_img_text}
inference_bs=${5:-64}  # 默认批量大小改为64

# 定义所有要执行的采样器
samplers=(
    "rand_sampler"
    "text_sim_sampler"
    "img_sim_sampler"
    "mix_sampler"
)

echo "=========================================="
echo "开始批量推理任务"
echo "任务: ${task}"
echo "数据集: ${dataset}"
echo "设备: ${device}"
echo "LeverLM配置: ${lever_lm}"
echo "推理批次大小: ${inference_bs}"
echo "采样器列表: ${samplers[@]}"
echo "=========================================="
echo ""

# 记录开始时间
start_time=$(date +%s)

# 遍历所有采样器并执行推理
for sampler in "${samplers[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始执行: ${sampler}"
    echo "命令: bash scripts/inference.sh ${task} ${dataset} ${device} ${lever_lm} ${sampler} ${inference_bs}"
    echo "=========================================="
    
    # 执行推理脚本
    bash scripts/inference.sh "${task}" "${dataset}" "${device}" "${lever_lm}" "${sampler}" "${inference_bs}"
    
    # 检查上一个命令的退出状态
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ ${sampler} 执行成功"
    else
        echo ""
        echo "✗ ${sampler} 执行失败，退出码: ${exit_code}"
        echo "是否继续执行下一个采样器？(y/n)"
        read -r continue
        if [ "$continue" != "y" ] && [ "$continue" != "Y" ]; then
            echo "用户选择停止执行"
            exit $exit_code
        fi
    fi
    
    echo ""
done

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "=========================================="
echo "所有推理任务完成！"
echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
echo "=========================================="

