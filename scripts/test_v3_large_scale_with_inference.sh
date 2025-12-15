#!/bin/bash
# 使用 inference.sh 测试 v3_large_scale 模型效果（RCE和GRPO）
# 使用方法: bash scripts/test_v3_large_scale_with_inference.sh [GPU_ID] [TEST_NUM] [MODEL_TYPE]
#
# MODEL_TYPE: rce 或 grpo (默认: rce)
#
# 示例:
#   bash scripts/test_v3_large_scale_with_inference.sh 4 200 rce
#   bash scripts/test_v3_large_scale_with_inference.sh 4 200 grpo

set -e

# 参数
GPU_ID=${1:-4}
TEST_NUM=${2:-200}
MODEL_TYPE=${3:-rce}  # rce 或 grpo

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa_local" # inference.sh 使用 okvqa_local
MODEL_NAME="qwen2.5_vl_3B" # inference.sh 使用 qwen2.5_vl_3B
SAMPLER="rand_sampler" # 文档中 Baseline 使用 RandSampler
LEVER_LM_CONFIG="query_img_text_icd_img_text" # 对应 train.lever_lm 配置
VERSION="v3" # 评估 v3 模型

# v3_large_scale 模型检查点路径
if [ "$MODEL_TYPE" == "rce" ]; then
    V3_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_large_scale/rce_epoch5.pt"
    MODEL_DESC="RCE (epoch5)"
elif [ "$MODEL_TYPE" == "grpo" ]; then
    V3_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_large_scale/grpo_epoch1.pt"
    MODEL_DESC="GRPO (epoch1)"
else
    echo "错误: MODEL_TYPE 必须是 'rce' 或 'grpo'"
    exit 1
fi

# 检查checkpoint是否存在
if [ ! -f "${V3_CKPT}" ]; then
    echo "错误: v3_large_scale checkpoint不存在: ${V3_CKPT}"
    echo ""
    echo "可用的checkpoint:"
    ls -lh "${PROJECT_ROOT}/results/okvqa/model_cpk/v3_large_scale/"*.pt 2>/dev/null | head -10
    exit 1
fi

echo "=========================================="
echo "测试 v3_large_scale 模型效果（使用inference.sh）"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Model Type: ${MODEL_DESC}"
echo "Checkpoint: ${V3_CKPT}"
echo "Test Num: ${TEST_NUM}"
echo "=========================================="

# 设置环境变量，让 inference.sh 使用指定的 checkpoint
export LEVER_LM_CHECKPOINT_PATH="${V3_CKPT}"
export LEVER_LM_CHECKPOINT_VERSION="${VERSION}" # 告知 inference.sh 这是 v3 模型

# 运行 inference.sh 脚本
# inference.sh 参数顺序: task dataset device lever_lm sampler beam_model version test_data_num
# 注意: 当使用 CUDA_VISIBLE_DEVICES 时，device 参数应该设置为 0（因为设备会被重新映射）
# inference.sh 会自动处理 test_data_num 和 shot_num_list
CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/inference.sh \
    "vqa" \
    "${DATASET}" \
    "0" \
    "${LEVER_LM_CONFIG}" \
    "${SAMPLER}" \
    "${MODEL_NAME}" \
    "${VERSION}" \
    "${TEST_NUM}"

# 清理环境变量
unset LEVER_LM_CHECKPOINT_PATH
unset LEVER_LM_CHECKPOINT_VERSION

echo "=========================================="
echo "✓ v3_large_scale ${MODEL_DESC} 模型测试完成！"
echo "请查看日志输出中的VQA准确率。"
echo "=========================================="

