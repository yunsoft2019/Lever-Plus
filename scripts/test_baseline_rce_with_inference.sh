#!/bin/bash
# 使用inference.sh测试Baseline RCE模型（与文档中12-12测试方法一致）
# 使用方法: bash scripts/test_baseline_rce_with_inference.sh [GPU_ID] [TEST_NUM]
#
# 注意: inference.sh会自动测试多个shot_num (1,2,3,4)，不需要指定shot_num
#
# 示例:
#   bash scripts/test_baseline_rce_with_inference.sh 4 200

set -e

# 参数
GPU_ID=${1:-4}
TEST_NUM=${2:-200}

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa_local"
DATASET_NAME="okvqa"

# Baseline RCE模型checkpoint（v2format格式，与文档一致）
BASELINE_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/rce_epoch5_v2format.ckpt"

echo "=========================================="
echo "使用inference.sh测试Baseline RCE模型"
echo "（与文档中12-12测试方法一致）"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Baseline Checkpoint: ${BASELINE_CKPT}"
echo "Test Num: ${TEST_NUM}"
echo "Shot Num: 自动测试 1,2,3,4 (inference.sh默认行为)"
echo "=========================================="

# 检查checkpoint是否存在
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "错误: Baseline checkpoint不存在: $BASELINE_CKPT"
    exit 1
fi

echo ""
echo "开始测试..."
echo ""

cd "${PROJECT_ROOT}"

# 设置checkpoint路径（inference.sh会使用这个环境变量）
export LEVER_LM_CHECKPOINT_PATH="${BASELINE_CKPT}"

# 使用inference.sh进行推理（与文档中12-12测试方法一致）
# 参数格式: task dataset device lever_lm sampler beam_model version test_data_num
bash scripts/inference.sh \
    vqa \
    ${DATASET} \
    ${GPU_ID} \
    query_img_text_icd_img_text \
    rand_sampler \
    qwen2.5_vl_3B \
    v3 \
    ${TEST_NUM}

echo ""
echo "=========================================="
echo "✓ Baseline RCE模型测试完成！"
echo "（使用inference.sh，与文档中12-12测试方法一致）"
echo "=========================================="

