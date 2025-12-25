#!/bin/bash
# RL数据多样性补全脚本 - All-Zero查询专用（超激进策略）
# 目标：对剩余152个all0 query使用更大的候选池和更多采样次数

set -e

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-2}  # 默认使用GPU2，可通过参数指定

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

# 加载.env文件（如果存在）
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
    echo "✓ 已加载.env文件"
    if [ -n "$OKVQA_PATH" ] && [[ "$OKVQA_PATH" != /* ]]; then
        export OKVQA_PATH="${PROJECT_ROOT}/${OKVQA_PATH}"
    fi
    if [ -n "$COCO_PATH" ] && [[ "$COCO_PATH" != /* ]]; then
        export COCO_PATH="${PROJECT_ROOT}/${COCO_PATH}"
    fi
fi

# 使用最新的all0数据作为输入（包含之前补全失败的query）
INPUT_RL_DATA="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all0_only_diverse_20251219_072948.json"
QUERY_EMB="${PROJECT_ROOT}/results/okvqa/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/okvqa/cache/candidate_embeddings.pt"
OUTPUT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all0_aggressive.json"
REPORT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all0_aggressive_report.json"

# 模型配置
SFT_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_RandSampler_v2data/rce_epoch5_v2format.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 数据集配置
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/okvqa_local.yaml"

# 设置环境变量
if [ -z "$OKVQA_PATH" ]; then
    if [ -d "/mnt/share/yiyun/datasets/okvqa" ]; then
        export OKVQA_PATH="/mnt/share/yiyun/datasets/okvqa"
    elif [ -d "${PROJECT_ROOT}/datasets/okvqa" ]; then
        export OKVQA_PATH="${PROJECT_ROOT}/datasets/okvqa"
    fi
fi

if [ -z "$COCO_PATH" ]; then
    if [ -d "/mnt/share/yiyun/datasets/mscoco" ]; then
        export COCO_PATH="/mnt/share/yiyun/datasets/mscoco"
    elif [ -d "${PROJECT_ROOT}/datasets/mscoco" ]; then
        export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"
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
echo "RL数据多样性补全 - All-Zero查询【超激进策略】"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "输入数据: ${INPUT_RL_DATA}"
echo "输出数据: ${OUTPUT_PATH} (将自动添加时间戳)"
echo ""
echo "【超激进策略参数】"
echo "  - max_eval_budget_all0: 150 (从50增加到150)"
echo "  - max_candidates_per_query: 48 (从24增加到48)"
echo "  - use_full_candidate_pool: 启用（使用64个最相似候选，而不是现有的15-19个）"
echo "  - 更多的L值范围和采样轮次"
echo ""
echo "预计处理: ~152个仍为all0的query"
echo "预计耗时: 较长（每个query最多150次VQA评测）"
echo ""

# 检查文件是否存在
if [ ! -f "${INPUT_RL_DATA}" ]; then
    echo "错误: 输入RL数据文件不存在: ${INPUT_RL_DATA}"
    echo "请先运行 augment_rl_data_diversity_all0_only.sh"
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

echo "开始超激进补全All-Zero查询..."
echo "开始时间: $(date)"
echo ""

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
    --max_candidates_per_query 48 \
    --max_eval_budget_all0 150 \
    --max_eval_budget_all1 40 \
    --max_eval_budget_all06 40 \
    --use_full_candidate_pool

echo ""
echo "=========================================="
echo "✓ All-Zero查询超激进补全完成"
echo "=========================================="
echo "结束时间: $(date)"
echo "输出文件: ${OUTPUT_PATH} (带时间戳)"
echo "报告文件: ${REPORT_PATH} (带时间戳)"
