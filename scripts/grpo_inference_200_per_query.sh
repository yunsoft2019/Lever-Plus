#!/bin/bash
# GRPO V3模型推理脚本 - 200条测试，使用Per-Query候选池训练的checkpoint
# 使用方法: bash scripts/grpo_inference_200_per_query.sh [GPU_ID]

set -e

GPU_ID=${1:-2}

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

# 加载.env文件（如果存在）
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
    echo "✓ 已加载.env文件"
fi

# 数据集配置
DATASET_NAME="okvqa"

# 文件路径（使用Per-Query候选池训练的checkpoint）
GRPO_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_per_query_candidate_pool/grpo_epoch8.pt"
IMG_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
BEAM_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_RandSampler_v4_strictEval_merged_fixed_candidate_pool.json"
OUTPUT_DIR="${PROJECT_ROOT}/results/v3_eval_per_query_candidate_pool"

# 检查文件是否存在
if [ ! -f "${GRPO_CKPT}" ]; then
    echo "错误: GRPO checkpoint文件不存在: ${GRPO_CKPT}"
    echo "请先完成训练: bash scripts/train_v3_with_per_query_candidate_pool.sh ${GPU_ID}"
    exit 1
fi

if [ ! -f "${IMG_EMB}" ]; then
    echo "错误: Image embedding文件不存在: ${IMG_EMB}"
    exit 1
fi

if [ ! -f "${BEAM_DATA}" ]; then
    echo "错误: Beam data文件不存在: ${BEAM_DATA}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "GRPO V3模型推理 - 200条测试（Per-Query候选池）"
echo "=========================================="
echo "GPU ID: ${GPU_ID}"
echo "GRPO检查点: ${GRPO_CKPT}"
echo "图像Embedding: ${IMG_EMB}"
echo "束搜索数据: ${BEAM_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "测试数量: 200"
echo "候选池: Per-Query候选池（每个query使用自己独立的64个候选）"
echo "=========================================="
echo ""

# 运行推理（确保输出不被重定向，能看到实时结果）
echo "开始推理..."
echo "✓ 使用Per-Query候选池训练的checkpoint，推理时也使用Per-Query候选池"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt "${GRPO_CKPT}" \
    --img_emb "${IMG_EMB}" \
    --beam_data "${BEAM_DATA}" \
    --dataset okvqa \
    --model_name "Qwen2.5-VL-3B-Instruct" \
    --shot_num 2 \
    --test_num 200 \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 32 \
    --device cuda:0 2>&1 | tee "${OUTPUT_DIR}/inference.log"

echo ""
echo "=========================================="
echo "✓ 推理完成！"
echo "=========================================="
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${OUTPUT_DIR}/inference.log"
echo ""

# 尝试从结果文件中读取准确率
RESULT_FILE="${OUTPUT_DIR}/v3_vqa_result.json"
if [ -f "${RESULT_FILE}" ]; then
    echo "📊 推理结果："
    python3 << EOF
import json
try:
    with open("${RESULT_FILE}", 'r') as f:
        result = json.load(f)
    accuracy = result.get('accuracy', 0.0)
    print(f"  VQA准确率: {accuracy*100:.2f}%")
    print(f"  模型: {result.get('model', 'N/A')}")
    print(f"  数据集: {result.get('dataset', 'N/A')}")
    print(f"  Shot数: {result.get('shot_num', 'N/A')}")
except Exception as e:
    print(f"  读取结果文件失败: {e}")
EOF
else
    echo "⚠️  结果文件不存在: ${RESULT_FILE}"
    echo "   请检查日志文件: ${OUTPUT_DIR}/inference.log"
fi

echo ""
echo "=========================================="


