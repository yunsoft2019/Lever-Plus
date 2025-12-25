#!/bin/bash
# 完整评估 KL_BETA=0.12 的 GRPO 模型（所有样本数和 epoch）
# 用于与 2025-12-20正确率.md 中的 KL_BETA=0.15 结果对比

set -e

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# GRPO epochs 列表
GRPO_EPOCHS=(1 2)

# 测试样本数列表
TEST_NUMS=(100 200 400 800)

# GPU 分配（可以根据实际情况调整）
# GPU 0: 100, 200 samples
# GPU 1: 400, 800 samples
declare -A GPU_MAP
GPU_MAP[100]=0
GPU_MAP[200]=0
GPU_MAP[400]=1
GPU_MAP[800]=1

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "完整评估 KL_BETA=0.12 的 GRPO 模型"
echo "对比基准: 2025-12-20正确率.md (KL_BETA=0.15)"
echo "=========================================="
echo ""

# 遍历每个 GRPO epoch
for epoch in "${GRPO_EPOCHS[@]}"; do
    echo "=========================================="
    echo "GRPO Epoch ${epoch}"
    echo "=========================================="
    
    # 遍历每个测试样本数
    for test_num in "${TEST_NUMS[@]}"; do
        gpu_id=${GPU_MAP[$test_num]}
        
        echo ""
        echo ">>> 评估 ${test_num} 样本 (GPU ${gpu_id})..."
        
        # 调用评估脚本
        bash scripts/eval_grpo_kl012.sh ${epoch} ${gpu_id} ${test_num}
        
        echo ""
        echo "等待 5 秒后继续..."
        sleep 5
    done
    
    echo ""
    echo "=========================================="
    echo "✓ GRPO Epoch ${epoch} 评估完成"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "✓ 所有评估完成！"
echo "=========================================="
echo ""
echo "结果汇总："
echo "  - GRPO Epoch 1: 100, 200, 400, 800 样本"
echo "  - GRPO Epoch 2: 100, 200, 400, 800 样本"
echo ""
echo "请查看推理输出，并与 2025-12-20正确率.md 中的结果对比"

