#!/bin/bash
# RL数据多样性补全脚本 - 专门用于补全All-0.6查询（使用增强策略）

set -e

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=7

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

INPUT_RL_DATA="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all06_only.json"
QUERY_EMB="${PROJECT_ROOT}/results/okvqa/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/okvqa/cache/candidate_embeddings.pt"
OUTPUT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all06_only_diverse.json"
REPORT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all06_only_diverse_report.json"

# 模型配置
SFT_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_RandSampler_v2data/rce_epoch5_v2format.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 数据集配置
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/okvqa_local.yaml"

# 设置环境变量（如果未设置）
if [ -z "$OKVQA_PATH" ]; then
    if [ -d "/mnt/share/yiyun/datasets/okvqa" ]; then
        export OKVQA_PATH="/mnt/share/yiyun/datasets/okvqa"
    elif [ -d "${PROJECT_ROOT}/datasets/okvqa" ]; then
        export OKVQA_PATH="${PROJECT_ROOT}/datasets/okvqa"
    elif [ -d "$HOME/datasets/okvqa" ]; then
        export OKVQA_PATH="$HOME/datasets/okvqa"
    fi
fi

if [ -z "$COCO_PATH" ]; then
    if [ -d "/mnt/share/yiyun/datasets/mscoco" ]; then
        export COCO_PATH="/mnt/share/yiyun/datasets/mscoco"
    elif [ -d "${PROJECT_ROOT}/datasets/mscoco" ]; then
        export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"
    elif [ -d "$HOME/datasets/mscoco" ]; then
        export COCO_PATH="$HOME/datasets/mscoco"
    fi
fi

echo "环境变量:"
echo "  OKVQA_PATH=${OKVQA_PATH:-未设置}"
echo "  COCO_PATH=${COCO_PATH:-未设置}"
echo ""

# VQA评测文件路径
VAL_QUES_PATH="${OKVQA_PATH}/OpenEnded_mscoco_val2014_questions.json"
VAL_ANN_PATH="${OKVQA_PATH}/mscoco_val2014_annotations.json"
TRAIN_QUES_PATH="${OKVQA_PATH}/OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN_PATH="${OKVQA_PATH}/mscoco_train2014_annotations.json"

echo "=========================================="
echo "RL数据多样性补全 - All-0.6查询专用（GPU${GPU_ID}）"
echo "=========================================="
echo "输入数据: ${INPUT_RL_DATA}"
echo "输出数据: ${OUTPUT_PATH} (将自动添加时间戳)"
echo "处理query数量: 全部All-0.6查询 (约21个)"
echo ""
echo "【补全策略】"
echo "  - All-0.6查询需要补出0或1.0的候选"
echo "  - 使用bottom pair和top pair策略"
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
echo "开始补全All-0.6查询..."
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
    --max_queries -1 \
    --max_candidates_per_query 24 \
    --max_eval_budget_all0 50 \
    --max_eval_budget_all1 8 \
    --max_eval_budget_all06 8

echo ""
echo "=========================================="
echo "✓ All-0.6查询补全完成"
echo "=========================================="
echo "输出文件: ${OUTPUT_PATH} (带时间戳)"
echo "报告文件: ${REPORT_PATH} (带时间戳)"


