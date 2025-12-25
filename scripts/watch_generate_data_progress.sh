#!/bin/bash
# 实时查看数据生成进度，过滤掉 DEBUG 日志

# 找到最新的 Hydra 日志文件
hydra_log=$(find results/vqav2/hydra_output/generate_data/vqa -name "generate_data.log" -type f 2>/dev/null | sort -t/ -k7 -r | head -1)

if [ -z "$hydra_log" ]; then
    echo "错误: 找不到 Hydra 日志文件"
    exit 1
fi

echo "=========================================="
echo "实时查看数据生成进度"
echo "=========================================="
echo "日志文件: $hydra_log"
echo "=========================================="
echo ""
echo "提示: 按 Ctrl+C 退出"
echo ""

# 使用 tail -f 实时跟踪，过滤掉 DEBUG 日志，只显示进度条和 INFO 信息
tail -f "$hydra_log" | grep --line-buffered -v "DEBUG" | grep --line-buffered -E "(it \[|%\|processing|remaining|skipping|INFO|WARNING|ERROR|进度|完成)" | while IFS= read -r line; do
    # 提取进度条信息
    if echo "$line" | grep -qE "(it \[|%\|)"; then
        echo "$line"
    # 提取 INFO 级别的关键信息
    elif echo "$line" | grep -qE "(processing|remaining|skipping|INFO.*gen_data)"; then
        echo "$line"
    fi
done



