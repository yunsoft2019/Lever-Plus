#!/bin/bash
# 评估 GRPO epoch2 模型
# 用 GPU 0/1/2/3 分别评估 100/200/400/800 样本

set -e

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
GRPO_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_k64_grpo/grpo_epoch2.pt"
V2FORMAT_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_k64_grpo/grpo_epoch2_v2format.ckpt"

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

cd "${PROJECT_ROOT}"

# 第一步：转换 checkpoint 格式（只需执行一次）
if [ ! -f "${V2FORMAT_CKPT}" ]; then
    echo "转换 grpo_epoch2.pt -> grpo_epoch2_v2format.ckpt ..."
    python scripts/convert_v3_to_v2_format.py \
        --v3_ckpt "${GRPO_CKPT}" \
        --output "${V2FORMAT_CKPT}"
    echo "✓ 转换完成"
fi

export LEVER_LM_CHECKPOINT_PATH="${V2FORMAT_CKPT}"

# 根据参数决定跑哪个GPU和样本量
GPU_ID=${1:-0}
TEST_NUM=${2:-100}

# 如果只传了一个参数，使用默认映射
if [ -z "$2" ]; then
    case $GPU_ID in
        0)
            TEST_NUM=100
            ;;
        1)
            TEST_NUM=200
            ;;
        3)
            TEST_NUM=400
            ;;
        4)
            TEST_NUM=800
            ;;
        *)
            echo "Usage: $0 <GPU_ID> [TEST_NUM]"
            echo "  默认映射: GPU 0->100, 1->200, 3->400, 4->800"
            exit 1
            ;;
    esac
fi

echo "=========================================="
echo "GRPO Epoch2 评估"
echo "GPU ${GPU_ID}: ${TEST_NUM} 样本, shot 1-4"
echo "Checkpoint: ${V2FORMAT_CKPT}"
echo "=========================================="

bash scripts/inference.sh vqa okvqa_local ${GPU_ID} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM}

echo ""
echo "✓ GPU ${GPU_ID} 完成: ${TEST_NUM} 样本评估"
