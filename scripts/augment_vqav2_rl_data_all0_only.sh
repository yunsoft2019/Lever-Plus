#!/bin/bash
# VQAv2 RL数据补全 - 只补全 All-Zero query
# 
# All-Zero query: 270 个 (5.4%)
# 这些 query 的所有候选都是错误的，需要补出正样本

set -e

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-0}

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
fi

export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"

# 先提取 All-Zero query
echo "=========================================="
echo "Step 1: 提取 All-Zero query"
echo "=========================================="

python3 -c "
import json

with open('results/vqav2/generated_data/rl_data_merged.json', 'r') as f:
    data = json.load(f)

all0_data = {}
for qid, qdata in data.items():
    candidates = qdata.get('pointer_candidates', [])
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    unique_scores = set(scores)
    
    if len(unique_scores) == 1 and abs(list(unique_scores)[0] - 0.0) < 1e-6:
        all0_data[qid] = qdata

print(f'提取了 {len(all0_data)} 个 All-Zero query')

with open('results/vqav2/generated_data/rl_data_merged_all0_only.json', 'w') as f:
    json.dump(all0_data, f, indent=2, ensure_ascii=False)
print('已保存到 results/vqav2/generated_data/rl_data_merged_all0_only.json')
"

INPUT_RL_DATA="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_all0_only.json"
QUERY_EMB="${PROJECT_ROOT}/results/vqav2/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/vqav2/cache/candidate_embeddings.pt"
OUTPUT_PATH="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_all0_only_diverse.json"
REPORT_PATH="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_all0_only_diverse_report.json"

SFT_CKPT="${PROJECT_ROOT}/results/vqav2/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample5000_epoch=17_train=19.10844_val=22.25372.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/vqav2_local.yaml"

VAL_QUES_PATH="${VQAV2_PATH}/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_ANN_PATH="${VQAV2_PATH}/v2_mscoco_val2014_annotations.json"
TRAIN_QUES_PATH="${VQAV2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN_PATH="${VQAV2_PATH}/v2_mscoco_train2014_annotations.json"

echo ""
echo "=========================================="
echo "Step 2: 补全 All-Zero query (GPU${GPU_ID})"
echo "=========================================="
echo "输入: ${INPUT_RL_DATA}"
echo "输出: ${OUTPUT_PATH}"
echo ""

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
    --max_eval_budget_all1 0 \
    --max_eval_budget_all06 0 \
    --use_full_candidate_pool

echo ""
echo "=========================================="
echo "✓ All-Zero query 补全完成"
echo "=========================================="
