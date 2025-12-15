#!/bin/bash
# 测试Baseline RCE模型效果（用于对比）
# 使用方法: bash scripts/test_baseline_rce_model.sh [GPU_ID] [TEST_NUM] [SHOT_NUM]
#
# 示例:
#   bash scripts/test_baseline_rce_model.sh 4 200 2

set -e

# 参数
GPU_ID=${1:-4}
TEST_NUM=${2:-200}
SHOT_NUM=${3:-2}

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa"

# Baseline RCE模型checkpoint（根据2025-12-13正确率.md，使用v2format格式）
# 优先使用v2format，如果不存在则使用.pt格式
BASELINE_CKPT_V2FORMAT="${PROJECT_ROOT}/results/${DATASET}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/rce_epoch5_v2format.ckpt"
BASELINE_CKPT_PT="${PROJECT_ROOT}/results/${DATASET}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/rce_epoch5.pt"

if [ -f "$BASELINE_CKPT_V2FORMAT" ]; then
    BASELINE_CKPT="$BASELINE_CKPT_V2FORMAT"
    echo "✓ 使用v2format checkpoint（与文档一致）"
else
    BASELINE_CKPT="$BASELINE_CKPT_PT"
    echo "⚠️  使用.pt checkpoint（v2format不存在）"
fi
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET}/cache/query_embeddings.pt"

# 使用原始beam_data（与文档中12-12测试一致）
# 注意：文档中的56.3%可能使用的是原始beam_data，而非RL数据
BEAM_DATA="${PROJECT_ROOT}/results/${DATASET}/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json"
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET}/eval/v3_baseline_rce"

# VQA模型配置
VQA_MODEL="Qwen2.5-VL-3B-Instruct"
DATASET_NAME="okvqa"

echo "=========================================="
echo "测试Baseline RCE模型效果（对比用）"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Baseline Checkpoint: ${BASELINE_CKPT}"
echo "Query Embeddings: ${QUERY_EMB}"
echo "Beam Data: ${BEAM_DATA}"
echo "Test Num: ${TEST_NUM}"
echo "Shot Num: ${SHOT_NUM}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="

# 检查checkpoint是否存在
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "错误: Baseline checkpoint不存在: $BASELINE_CKPT"
    echo ""
    echo "可用的checkpoint:"
    ls -lh "${PROJECT_ROOT}/results/${DATASET}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/"*.pt 2>/dev/null | head -10
    exit 1
fi

# 检查其他文件
if [ ! -f "$QUERY_EMB" ]; then
    echo "错误: Query embeddings不存在: $QUERY_EMB"
    exit 1
fi

if [ ! -f "$BEAM_DATA" ]; then
    echo "错误: Beam data不存在: $BEAM_DATA"
    exit 1
fi

echo ""
echo "开始测试..."
echo ""

cd "${PROJECT_ROOT}"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行评估
# 使用 --full_train_set 参数，使用完整训练集作为候选池（与文档中测试一致）
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt "${BASELINE_CKPT}" \
    --img_emb "${QUERY_EMB}" \
    --beam_data "${BEAM_DATA}" \
    --dataset "${DATASET_NAME}" \
    --model_name "${VQA_MODEL}" \
    --shot_num ${SHOT_NUM} \
    --test_num ${TEST_NUM} \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 32 \
    --device cuda:0 \
    --full_train_set

echo ""
echo "=========================================="
echo "✓ Baseline RCE模型测试完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  cat ${OUTPUT_DIR}/results.json"
echo "  cat ${OUTPUT_DIR}/vqa_accuracy.txt"

