#!/bin/bash
# VQAv2 RL数据补全 - 只补全 All-One query（补负样本）
# All-One query: 1954 个 (39.1%) - 所有候选都正确，需要补负样本

set -e
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-0}
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

[ -f "${PROJECT_ROOT}/.env" ] && source "${PROJECT_ROOT}/.env"
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"

echo "Step 1: 提取 All-One query"
python3 -c "
import json
with open('results/vqav2/generated_data/rl_data_merged.json', 'r') as f:
    data = json.load(f)
all1_data = {}
for qid, qdata in data.items():
    candidates = qdata.get('pointer_candidates', [])
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    unique_scores = set(scores)
    if len(unique_scores) == 1 and abs(list(unique_scores)[0] - 1.0) < 1e-6:
        all1_data[qid] = qdata
print(f'提取了 {len(all1_data)} 个 All-One query')
with open('results/vqav2/generated_data/rl_data_merged_all1_only.json', 'w') as f:
    json.dump(all1_data, f, indent=2, ensure_ascii=False)
"

INPUT="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_all1_only.json"
OUTPUT="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_all1_only_diverse.json"
REPORT="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_merged_all1_only_diverse_report.json"

SFT_CKPT="${PROJECT_ROOT}/results/vqav2/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample5000_epoch=17_train=19.10844_val=22.25372.ckpt"

echo "Step 2: 补全 All-One query (GPU${GPU_ID})"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.models.v3.augment_rl_data_diversity \
    --input_rl_data "${INPUT}" \
    --query_embeddings "${PROJECT_ROOT}/results/vqav2/cache/query_embeddings.pt" \
    --candidate_embeddings "${PROJECT_ROOT}/results/vqav2/cache/candidate_embeddings.pt" \
    --output_path "${OUTPUT}" \
    --report_path "${REPORT}" \
    --sft_ckpt "${SFT_CKPT}" \
    --vqa_model "Qwen/Qwen2.5-VL-3B-Instruct" \
    --device "cuda:0" \
    --val_ques_path "${VQAV2_PATH}/v2_OpenEnded_mscoco_val2014_questions.json" \
    --val_ann_path "${VQAV2_PATH}/v2_mscoco_val2014_annotations.json" \
    --train_ques_path "${VQAV2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json" \
    --train_ann_path "${VQAV2_PATH}/v2_mscoco_train2014_annotations.json" \
    --dataset_config "${PROJECT_ROOT}/configs/dataset/vqav2_local.yaml" \
    --max_queries -1 \
    --max_candidates_per_query 24 \
    --max_eval_budget_all0 0 \
    --max_eval_budget_all1 40 \
    --max_eval_budget_all06 0 \
    --use_full_candidate_pool

echo "✓ All-One query 补全完成"
