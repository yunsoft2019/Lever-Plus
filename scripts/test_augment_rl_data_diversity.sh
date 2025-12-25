#!/bin/bash
# 测试RL数据多样性补全脚本
# 使用GPU6，测试10个query

set -e

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=6
MAX_QUERIES=10  # 先测试10个query

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

# 加载.env文件（如果存在）
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
    echo "✓ 已加载.env文件"
    # 将相对路径转换为绝对路径
    if [ -n "$OKVQA_PATH" ] && [[ "$OKVQA_PATH" != /* ]]; then
        export OKVQA_PATH="${PROJECT_ROOT}/${OKVQA_PATH}"
    fi
    if [ -n "$COCO_PATH" ] && [[ "$COCO_PATH" != /* ]]; then
        export COCO_PATH="${PROJECT_ROOT}/${COCO_PATH}"
    fi
fi

INPUT_RL_DATA="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval.json"
QUERY_EMB="${PROJECT_ROOT}/results/okvqa/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/okvqa/cache/candidate_embeddings.pt"
OUTPUT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_diverse_test.json"
REPORT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_diverse_test_report.json"

# 模型配置
SFT_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_RandSampler_v2data/rce_epoch5_v2format.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 数据集配置
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/okvqa_local.yaml"

# 设置环境变量（如果未设置）
if [ -z "$OKVQA_PATH" ]; then
    # 尝试从常见路径推断
    if [ -d "/mnt/share/yiyun/datasets/okvqa" ]; then
        export OKVQA_PATH="/mnt/share/yiyun/datasets/okvqa"
    elif [ -d "${PROJECT_ROOT}/datasets/okvqa" ]; then
        export OKVQA_PATH="${PROJECT_ROOT}/datasets/okvqa"
    elif [ -d "$HOME/datasets/okvqa" ]; then
        export OKVQA_PATH="$HOME/datasets/okvqa"
    else
        echo "警告: 未设置OKVQA_PATH环境变量，且无法自动推断"
        echo "      将跳过dataset加载，candidate_pool可能无法使用"
        # 不退出，继续执行（candidate_pool可以为None）
    fi
fi

if [ -z "$COCO_PATH" ]; then
    # 尝试从常见路径推断
    if [ -d "/mnt/share/yiyun/datasets/mscoco" ]; then
        export COCO_PATH="/mnt/share/yiyun/datasets/mscoco"
    elif [ -d "${PROJECT_ROOT}/datasets/mscoco" ]; then
        export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"
    elif [ -d "$HOME/datasets/mscoco" ]; then
        export COCO_PATH="$HOME/datasets/mscoco"
    else
        echo "警告: 未设置COCO_PATH环境变量，且无法自动推断"
        echo "      将跳过dataset加载，candidate_pool可能无法使用"
        # 不退出，继续执行（candidate_pool可以为None）
    fi
fi

echo "环境变量:"
echo "  OKVQA_PATH=${OKVQA_PATH:-未设置}"
echo "  COCO_PATH=${COCO_PATH:-未设置}"
echo ""

# VQA评测文件路径（从环境变量获取）
VAL_QUES_PATH="${OKVQA_PATH}/OpenEnded_mscoco_val2014_questions.json"
VAL_ANN_PATH="${OKVQA_PATH}/mscoco_val2014_annotations.json"
TRAIN_QUES_PATH="${OKVQA_PATH}/OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN_PATH="${OKVQA_PATH}/mscoco_train2014_annotations.json"

echo "=========================================="
echo "RL数据多样性补全测试（GPU${GPU_ID}）"
echo "=========================================="
echo "输入数据: ${INPUT_RL_DATA}"
echo "输出数据: ${OUTPUT_PATH}"
echo "测试query数量: ${MAX_QUERIES}"
echo ""

# 检查文件是否存在
if [ ! -f "${INPUT_RL_DATA}" ]; then
    echo "错误: 输入RL数据文件不存在: ${INPUT_RL_DATA}"
    exit 1
fi

if [ ! -f "${QUERY_EMB}" ]; then
    echo "错误: Query embeddings文件不存在: ${QUERY_EMB}"
    exit 1
fi

if [ ! -f "${CAND_EMB}" ]; then
    echo "错误: Candidate embeddings文件不存在: ${CAND_EMB}"
    exit 1
fi

# 运行补全脚本
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.models.v3.augment_rl_data_diversity \
    --input_rl_data "${INPUT_RL_DATA}" \
    --query_embeddings "${QUERY_EMB}" \
    --candidate_embeddings "${CAND_EMB}" \
    --output_path "${OUTPUT_PATH}" \
    --report_path "${REPORT_PATH}" \
    --add_timestamp \
    --sft_ckpt "${SFT_CKPT}" \
    --vqa_model "${VQA_MODEL}" \
    --device "cuda:0" \
    --val_ques_path "${VAL_QUES_PATH}" \
    --val_ann_path "${VAL_ANN_PATH}" \
    --train_ques_path "${TRAIN_QUES_PATH}" \
    --train_ann_path "${TRAIN_ANN_PATH}" \
    --dataset_config "${DATASET_CONFIG}" \
    --max_queries ${MAX_QUERIES} \
    --max_candidates_per_query 20 \
    --max_eval_budget_all0 12 \
    --max_eval_budget_all1 6 \
    --max_eval_budget_all06 6

echo ""
echo "=========================================="
echo "✓ 测试完成"
echo "=========================================="
echo "输出文件: ${OUTPUT_PATH}"
echo "报告文件: ${REPORT_PATH}"

