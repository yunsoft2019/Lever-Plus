#!/bin/bash

# RL数据多样性补全脚本（修复候选池大小问题）
# 方案1：限制候选池大小为64，每个query只使用自己的候选索引

set -e

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "$PROJECT_ROOT"

# GPU设置
GPU_ID=7

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 输入输出文件
INPUT_RL_DATA="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_merged.json"
OUTPUT_RL_DATA="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_fixed_candidate_pool.json"
REPORT_FILE="${PROJECT_ROOT}/results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_fixed_candidate_pool_report.json"

# Embeddings路径
QUERY_EMB="${PROJECT_ROOT}/results/okvqa/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/okvqa/cache/candidate_embeddings.pt"

# 模型路径
SFT_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 数据集路径
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/okvqa_local.yaml"
VAL_QUES_PATH="${PROJECT_ROOT}/data/okvqa/OpenEnded_mscoco_val2014_questions.json"
VAL_ANN_PATH="${PROJECT_ROOT}/data/okvqa/mscoco_val2014_annotations.json"
TRAIN_QUES_PATH="${PROJECT_ROOT}/data/okvqa/OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN_PATH="${PROJECT_ROOT}/data/okvqa/mscoco_train2014_annotations.json"

# 补全参数
MAX_CANDIDATES_PER_QUERY=24
MAX_EVAL_BUDGET_ALL0=100
MAX_EVAL_BUDGET_ALL1=8
MAX_EVAL_BUDGET_ALL06=8
MAX_QUERIES=-1  # -1表示处理所有query

echo "=========================================="
echo "RL数据多样性补全（修复候选池大小问题）"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "输入文件: ${INPUT_RL_DATA}"
echo "输出文件: ${OUTPUT_RL_DATA}"
echo "报告文件: ${REPORT_FILE}"
echo "最大query数: ${MAX_QUERIES}"
echo "=========================================="
echo ""

# 运行补全脚本
python -m lever_lm.models.v3.augment_rl_data_diversity \
    --input_rl_data "${INPUT_RL_DATA}" \
    --query_embeddings "${QUERY_EMB}" \
    --candidate_embeddings "${CAND_EMB}" \
    --output_path "${OUTPUT_RL_DATA}" \
    --report_path "${REPORT_FILE}" \
    --sft_ckpt "${SFT_CKPT}" \
    --vqa_model "${VQA_MODEL}" \
    --device "cuda:${GPU_ID}" \
    --dataset_config "${DATASET_CONFIG}" \
    --val_ques_path "${VAL_QUES_PATH}" \
    --val_ann_path "${VAL_ANN_PATH}" \
    --train_ques_path "${TRAIN_QUES_PATH}" \
    --train_ann_path "${TRAIN_ANN_PATH}" \
    --max_candidates_per_query "${MAX_CANDIDATES_PER_QUERY}" \
    --max_eval_budget_all0 "${MAX_EVAL_BUDGET_ALL0}" \
    --max_eval_budget_all1 "${MAX_EVAL_BUDGET_ALL1}" \
    --max_eval_budget_all06 "${MAX_EVAL_BUDGET_ALL06}" \
    --max_queries "${MAX_QUERIES}" \
    --add_timestamp

echo ""
echo "=========================================="
echo "补全任务完成！"
echo "=========================================="
echo "输出文件: ${OUTPUT_RL_DATA}"
echo "报告文件: ${REPORT_FILE}"
echo ""
