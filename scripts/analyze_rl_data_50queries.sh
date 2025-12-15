#!/bin/bash
# 分析 50 个 query 的 RL 数据质量

cd /mnt/share/yiyun/Projects/Lever-Plus

RL_DATA="results/okvqa/generated_data/rl_data_v4_50queries.json"
OUTPUT_CSV="results/okvqa/generated_data/rl_data_v4_50queries_analysis.csv"

echo "=========================================="
echo "RL 数据体检脚本"
echo "=========================================="
echo "RL 数据文件: $RL_DATA"
echo "输出 CSV: $OUTPUT_CSV"
echo ""

# 检查文件是否存在
if [ ! -f "$RL_DATA" ]; then
    echo "✗ 错误：RL 数据文件不存在: $RL_DATA"
    echo "请等待 RL 数据生成完成"
    exit 1
fi

# 检查文件大小
FILE_SIZE=$(du -h "$RL_DATA" | cut -f1)
echo "文件大小: $FILE_SIZE"

# 运行体检脚本
echo ""
echo "开始分析..."
conda activate lever_env && python -m lever_lm.models.v3.analyze_rl_data_v4 \
    --rl_data "$RL_DATA" \
    --output_csv "$OUTPUT_CSV"

echo ""
echo "=========================================="
echo "✓ 体检完成！"
echo "=========================================="
echo ""
echo "查看详细结果："
echo "  - Query 级别统计: ${OUTPUT_CSV%.csv}_query_level.csv"
echo "  - Candidate 级别统计: ${OUTPUT_CSV%.csv}_candidate_level.csv"
echo ""

