#!/bin/bash
# 用 GPU 0/1/2 分别评估 GRPO epoch1 模型在 100/400/800 样本上的效果
# inference.sh 会自动跑 shot 1-4

set -e

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
CKPT_PATH="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_k64_grpo/grpo_epoch1_v2format.ckpt"

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

cd "${PROJECT_ROOT}"

export LEVER_LM_CHECKPOINT_PATH="${CKPT_PATH}"

# 根据参数决定跑哪个GPU
GPU_ID=${1:-0}

case $GPU_ID in
    0)
        TEST_NUM=100
        ;;
    1)
        TEST_NUM=400
        ;;
    2)
        TEST_NUM=800
        ;;
    *)
        echo "Usage: $0 <GPU_ID: 0|1|2>"
        echo "  GPU 0 -> 100 samples"
        echo "  GPU 1 -> 400 samples"
        echo "  GPU 2 -> 800 samples"
        exit 1
        ;;
esac

echo "=========================================="
echo "GRPO Epoch1 评估"
echo "GPU ${GPU_ID}: ${TEST_NUM} 样本, shot 1-4"
echo "Checkpoint: ${CKPT_PATH}"
echo "=========================================="

bash scripts/inference.sh vqa okvqa_local ${GPU_ID} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM}

echo ""
echo "✓ GPU ${GPU_ID} 完成: ${TEST_NUM} 样本评估"
