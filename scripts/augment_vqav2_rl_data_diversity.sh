#!/bin/bash
# VQAv2 RL数据多样性补全脚本
# 参考 OKVQA 的数据补全方法
# 
# 核心思路：
# - 对全0的query补出至少一个0.6或1.0（正样本）
# - 对全1.0的query补出至少一个0或0.6（负样本）
# - 对全0.6的query补出至少一个0或1.0
# - 最终让每个query的候选集合至少有2档reward

set -e

# 激活conda环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-0}

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

# 加载.env文件
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
    echo "✓ 已加载.env文件"
fi

# VQAv2 路径配置
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"

# 输入输出配置
# 使用原始 merged 数据进行补全（包含 270 个 All-Zero 和 1954 个 All-One query）
INPUT_RL_DATA="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged.json"
QUERY_EMB="${PROJECT_ROOT}/results/vqav2/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/vqav2/cache/candidate_embeddings.pt"
OUTPUT_PATH="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_diverse.json"
REPORT_PATH="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_diverse_report.json"

# 模型配置
SFT_CKPT="${PROJECT_ROOT}/results/vqav2/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample5000_epoch=17_train=19.10844_val=22.25372.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 数据集配置
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/vqav2_local.yaml"

# VQA评测文件路径
VAL_QUES_PATH="${VQAV2_PATH}/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_ANN_PATH="${VQAV2_PATH}/v2_mscoco_val2014_annotations.json"
TRAIN_QUES_PATH="${VQAV2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN_PATH="${VQAV2_PATH}/v2_mscoco_train2014_annotations.json"

echo "=========================================="
echo "VQAv2 RL数据多样性补全 (GPU${GPU_ID})"
echo "=========================================="
echo "输入数据: ${INPUT_RL_DATA}"
echo "输出数据: ${OUTPUT_PATH}"
echo ""
echo "【数据分布】"
echo "  - All-Zero query: 270 (5.4%) - 需要补正样本"
echo "  - All-One query: 1954 (39.1%) - 需要补负样本"
echo "  - All-0.6 query: 36 (0.7%)"
echo "  - Diverse query: 2740 (54.8%) - 已多样"
echo ""
echo "【补全策略】"
echo "  - 对全0 query: 补出正样本 (0.6/1.0)"
echo "  - 对全1 query: 补出负样本 (0/0.6)"
echo "  - 对全0.6 query: 补出 0 或 1.0"
echo "  - max_eval_budget_all0: 50"
echo "  - max_eval_budget_all1: 40"
echo "  - max_eval_budget_all06: 40"
echo ""

# 检查文件是否存在
if [ ! -f "${INPUT_RL_DATA}" ]; then
    echo "错误: 输入RL数据文件不存在: ${INPUT_RL_DATA}"
    echo "请确认文件路径正确"
    exit 1
fi

if [ ! -f "${QUERY_EMB}" ]; then
    echo "错误: Query embeddings文件不存在: ${QUERY_EMB}"
    echo "需要先生成 query embeddings"
    exit 1
fi

if [ ! -f "${CAND_EMB}" ]; then
    echo "错误: Candidate embeddings文件不存在: ${CAND_EMB}"
    echo "需要先生成 candidate embeddings"
    exit 1
fi

# 运行补全脚本
echo "开始补全..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.models.v3.augment_rl_data_diversity \
    --input_rl_data "${INPUT_RL_DATA}" \
    --query_embeddings "${QUERY_EMB}" \
    --candidate_embeddings "${CAND_EMB}" \
    --output_path "${OUTPUT_PATH}" \
    --report_path "${REPORT_PATH}" \
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
    --max_eval_budget_all1 40 \
    --max_eval_budget_all06 40 \
    --use_full_candidate_pool

echo ""
echo "=========================================="
echo "✓ VQAv2 数据补全完成"
echo "=========================================="
echo "输出文件: ${OUTPUT_PATH}"
echo "报告文件: ${REPORT_PATH}"
