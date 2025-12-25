#!/bin/bash
# 测试VQA生成过程，只处理1个查询用于调试

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
fi

# 使用只包含1个All-Zero查询的测试数据
INPUT_RL_DATA="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_test_single_all0.json"
QUERY_EMB="${PROJECT_ROOT}/results/okvqa/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/okvqa/cache/candidate_embeddings.pt"
OUTPUT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_test_vqa_debug.json"
REPORT_PATH="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_test_vqa_debug_report.json"

# 模型配置
SFT_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_RandSampler_v2data/rce_epoch5_v2format.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 数据集配置
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/okvqa_local.yaml"

# VQA评测文件路径
VAL_QUES_PATH="${OKVQA_PATH}/OpenEnded_mscoco_val2014_questions.json"
VAL_ANN_PATH="${OKVQA_PATH}/mscoco_val2014_annotations.json"
TRAIN_QUES_PATH="${OKVQA_PATH}/OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN_PATH="${OKVQA_PATH}/mscoco_train2014_annotations.json"

echo "=========================================="
echo "VQA生成调试测试（只处理1个查询）"
echo "=========================================="

# 运行补全脚本，只处理1个查询
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
    --max_queries 1 \
    --max_candidates_per_query 24 \
    --max_eval_budget_all0 5 \
    --max_eval_budget_all1 8 \
    --max_eval_budget_all06 8

echo ""
echo "=========================================="
echo "✓ 调试测试完成"
echo "=========================================="

