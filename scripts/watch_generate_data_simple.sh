#!/bin/bash
# 简洁版：只显示进度条，不显示其他日志

# 找到最新的 Hydra 日志文件
hydra_log=$(find results/vqav2/hydra_output/generate_data/vqa -name "generate_data.log" -type f 2>/dev/null | sort -t/ -k7 -r | head -1)

if [ -z "$hydra_log" ]; then
    echo "错误: 找不到 Hydra 日志文件"
    exit 1
fi

echo "=========================================="
echo "实时进度监控（只显示进度条）"
echo "=========================================="
echo "日志文件: $hydra_log"
echo "按 Ctrl+C 退出"
echo "=========================================="
echo ""

# 只显示进度条相关的行
tail -f "$hydra_log" | grep --line-buffered -E "(it \[|%\|)" | while IFS= read -r line; do
    # 清理输出，只显示进度条
    echo "$line" | sed 's/.*DEBUG.*//' | grep -v "^$"
done



