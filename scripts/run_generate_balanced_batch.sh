#!/bin/bash
# 批量启动 balanced RL 数据生成任务
# 每个 GPU 跑 2 个进程，每个进程处理 1000 条

# 起始索引（从上次结束的地方继续）
START=${1:-43000}

echo "=========================================="
echo "批量启动 Balanced RL 数据生成"
echo "=========================================="
echo "起始索引: ${START}"
echo "每个 GPU 跑 2 个进程"
echo "=========================================="

# GPU 0-7，每个 GPU 跑 2 个任务
for gpu in 0 1 2 3 4 5 6 7; do
    idx1=$((START + gpu * 2000))
    idx2=$((idx1 + 1000))
    idx3=$((idx2 + 1000))
    
    echo "GPU ${gpu}: ${idx1}-${idx2}, ${idx2}-${idx3}"
    
    nohup bash scripts/run_generate_balanced_all_gpus.sh ${gpu} ${idx1} ${idx2} > logs/balanced_${idx1}_${idx2}.log 2>&1 &
    nohup bash scripts/run_generate_balanced_all_gpus.sh ${gpu} ${idx2} ${idx3} > logs/balanced_${idx2}_${idx3}.log 2>&1 &
done

echo ""
echo "✓ 已启动 16 个任务"
echo "查看进度: ps aux | grep generate_balanced"
echo "查看日志: tail -f logs/balanced_*.log"
