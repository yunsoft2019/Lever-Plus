#!/bin/bash
# 监控 RL 数据生成进度

LOG_FILE="results/okvqa/generated_data/rl_data_v4_50queries.log"
OUTPUT_FILE="results/okvqa/generated_data/rl_data_v4_50queries.json"

echo "=========================================="
echo "RL 数据生成监控"
echo "=========================================="
echo "日志文件: $LOG_FILE"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 检查进程是否在运行
if ps aux | grep -v grep | grep "generate_rl_data" > /dev/null; then
    echo "✓ 进程正在运行"
else
    echo "✗ 进程未运行"
fi

echo ""
echo "【GPU 使用情况】"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | grep "^3,"

echo ""
echo "【最新日志（最后 20 行）】"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "日志文件尚未生成"
fi

echo ""
echo "【输出文件状态】"
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "✓ 文件已生成，大小: $FILE_SIZE"
    # 检查文件中的 query 数量
    if command -v python3 &> /dev/null; then
        QUERY_COUNT=$(python3 -c "import json; data=json.load(open('$OUTPUT_FILE')); print(len(data))" 2>/dev/null)
        if [ ! -z "$QUERY_COUNT" ]; then
            echo "  已生成 query 数量: $QUERY_COUNT / 50"
        fi
    fi
else
    echo "✗ 文件尚未生成"
fi

echo ""
echo "=========================================="

