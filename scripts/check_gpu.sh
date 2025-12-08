#!/bin/bash
# 检查 GPU 使用情况
echo "=========================================="
echo "GPU 使用情况"
echo "=========================================="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,processes.name --format=csv,noheader | awk -F',' '{printf "GPU %s: %s/%s MB (%.1f%%), 利用率: %s%%, 进程: %s\n", $1, $2, $3, ($2/$3)*100, $4, $5}'
echo "=========================================="
echo ""
echo "推荐使用内存使用率 < 50% 的 GPU"

