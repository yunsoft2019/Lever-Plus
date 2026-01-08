#!/bin/bash
# VQAv2 RL数据激进补全 - 增加eval budget，尝试补全剩余的All-Zero和All-One query
# 目标：进一步降低正样本比例，提高数据多样性

set -e
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

GPU_ID=${1:-0}
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
cd "${PROJECT_ROOT}"

[ -f "${PROJECT_ROOT}/.env" ] && source "${PROJECT_ROOT}/.env"
export VQAV2_PATH="${PROJECT_ROOT}/datasets/vqav2"
export COCO_PATH="${PROJECT_ROOT}/datasets/mscoco"

echo "=========================================="
echo "VQAv2 RL数据激进补全"
echo "=========================================="
echo "当前数据状态:"
python3 -c "
import json
with open('results/vqav2/generated_data/rl_data_merged_full_diverse.json', 'r') as f:
    data = json.load(f)
all0 = all1 = diverse = 0
total_cand = pos_cand = 0
for qid, qdata in data.items():
    candidates = qdata.get('pointer_candidates', [])
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    total_cand += len(candidates)
    pos_cand += sum(1 for s in scores if s > 0)
    unique = set(scores)
    if len(unique) == 1:
        if abs(list(unique)[0]) < 1e-6: all0 += 1
        elif abs(list(unique)[0] - 1.0) < 1e-6: all1 += 1
        else: diverse += 1
    else: diverse += 1
print(f'  All-Zero: {all0} ({100*all0/len(data):.1f}%)')
print(f'  All-One: {all1} ({100*all1/len(data):.1f}%)')
print(f'  Diverse: {diverse} ({100*diverse/len(data):.1f}%)')
print(f'  正样本比例: {100*pos_cand/total_cand:.1f}%')
"
echo "=========================================="

# Step 1: 提取仍然是 All-Zero 的 query
echo "Step 1: 提取仍然是 All-Zero 的 query"
python3 -c "
import json
with open('results/vqav2/generated_data/rl_data_merged_full_diverse.json', 'r') as f:
    data = json.load(f)
all0_data = {}
for qid, qdata in data.items():
    candidates = qdata.get('pointer_candidates', [])
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    unique_scores = set(scores)
    if len(unique_scores) == 1 and abs(list(unique_scores)[0]) < 1e-6:
        all0_data[qid] = qdata
print(f'提取了 {len(all0_data)} 个 All-Zero query')
with open('results/vqav2/generated_data/rl_data_remaining_all0.json', 'w') as f:
    json.dump(all0_data, f, indent=2, ensure_ascii=False)
"

# Step 2: 提取仍然是 All-One 的 query
echo "Step 2: 提取仍然是 All-One 的 query"
python3 -c "
import json
with open('results/vqav2/generated_data/rl_data_merged_full_diverse.json', 'r') as f:
    data = json.load(f)
all1_data = {}
for qid, qdata in data.items():
    candidates = qdata.get('pointer_candidates', [])
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    unique_scores = set(scores)
    if len(unique_scores) == 1 and abs(list(unique_scores)[0] - 1.0) < 1e-6:
        all1_data[qid] = qdata
print(f'提取了 {len(all1_data)} 个 All-One query')
with open('results/vqav2/generated_data/rl_data_remaining_all1.json', 'w') as f:
    json.dump(all1_data, f, indent=2, ensure_ascii=False)
"

INPUT_ALL0="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_remaining_all0.json"
OUTPUT_ALL0="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_remaining_all0_augmented.json"
REPORT_ALL0="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_remaining_all0_augmented_report.json"

INPUT_ALL1="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_remaining_all1.json"
OUTPUT_ALL1="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_remaining_all1_augmented.json"
REPORT_ALL1="${PROJECT_ROOT}/results/vqav2/generated_data/rl_data_remaining_all1_augmented_report.json"

SFT_CKPT="${PROJECT_ROOT}/results/vqav2/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample5000_epoch=17_train=19.10844_val=22.25372.ckpt"

# Step 3: 激进补全 All-Zero query (增加eval budget到60)
echo "Step 3: 激进补全 All-Zero query (eval_budget=60)"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.models.v3.augment_rl_data_diversity \
    --input_rl_data "${INPUT_ALL0}" \
    --query_embeddings "${PROJECT_ROOT}/results/vqav2/cache/query_embeddings.pt" \
    --candidate_embeddings "${PROJECT_ROOT}/results/vqav2/cache/candidate_embeddings.pt" \
    --output_path "${OUTPUT_ALL0}" \
    --report_path "${REPORT_ALL0}" \
    --sft_ckpt "${SFT_CKPT}" \
    --vqa_model "Qwen/Qwen2.5-VL-3B-Instruct" \
    --device "cuda:0" \
    --val_ques_path "${VQAV2_PATH}/v2_OpenEnded_mscoco_val2014_questions.json" \
    --val_ann_path "${VQAV2_PATH}/v2_mscoco_val2014_annotations.json" \
    --train_ques_path "${VQAV2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json" \
    --train_ann_path "${VQAV2_PATH}/v2_mscoco_train2014_annotations.json" \
    --dataset_config "${PROJECT_ROOT}/configs/dataset/vqav2_local.yaml" \
    --max_queries -1 \
    --max_candidates_per_query 30 \
    --max_eval_budget_all0 60 \
    --max_eval_budget_all1 0 \
    --max_eval_budget_all06 0 \
    --use_full_candidate_pool

echo "✓ All-Zero query 激进补全完成"

# Step 4: 激进补全 All-One query (增加eval budget到80)
echo "Step 4: 激进补全 All-One query (eval_budget=80)"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.models.v3.augment_rl_data_diversity \
    --input_rl_data "${INPUT_ALL1}" \
    --query_embeddings "${PROJECT_ROOT}/results/vqav2/cache/query_embeddings.pt" \
    --candidate_embeddings "${PROJECT_ROOT}/results/vqav2/cache/candidate_embeddings.pt" \
    --output_path "${OUTPUT_ALL1}" \
    --report_path "${REPORT_ALL1}" \
    --sft_ckpt "${SFT_CKPT}" \
    --vqa_model "Qwen/Qwen2.5-VL-3B-Instruct" \
    --device "cuda:0" \
    --val_ques_path "${VQAV2_PATH}/v2_OpenEnded_mscoco_val2014_questions.json" \
    --val_ann_path "${VQAV2_PATH}/v2_mscoco_val2014_annotations.json" \
    --train_ques_path "${VQAV2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json" \
    --train_ann_path "${VQAV2_PATH}/v2_mscoco_train2014_annotations.json" \
    --dataset_config "${PROJECT_ROOT}/configs/dataset/vqav2_local.yaml" \
    --max_queries -1 \
    --max_candidates_per_query 80 \
    --max_eval_budget_all0 0 \
    --max_eval_budget_all1 80 \
    --max_eval_budget_all06 0 \
    --use_full_candidate_pool

echo "✓ All-One query 激进补全完成"

# Step 5: 合并数据
echo "Step 5: 合并增强后的数据"
python3 << 'EOF'
import json

# 读取原始数据
with open('results/vqav2/generated_data/rl_data_merged_full_diverse.json', 'r') as f:
    merged_data = json.load(f)

# 读取 All-Zero 增强数据
try:
    with open('results/vqav2/generated_data/rl_data_remaining_all0_augmented.json', 'r') as f:
        all0_aug = json.load(f)
    for qid, qdata in all0_aug.items():
        if qid in merged_data:
            merged_data[qid] = qdata
    print(f"合并了 {len(all0_aug)} 个 All-Zero 增强数据")
except FileNotFoundError:
    print("All-Zero 增强数据文件不存在")

# 读取 All-One 增强数据
try:
    with open('results/vqav2/generated_data/rl_data_remaining_all1_augmented.json', 'r') as f:
        all1_aug = json.load(f)
    for qid, qdata in all1_aug.items():
        if qid in merged_data:
            merged_data[qid] = qdata
    print(f"合并了 {len(all1_aug)} 个 All-One 增强数据")
except FileNotFoundError:
    print("All-One 增强数据文件不存在")

# 保存合并后的数据
with open('results/vqav2/generated_data/rl_data_merged_aggressive.json', 'w') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

# 统计
all0 = all1 = diverse = 0
total_cand = pos_cand = 0
for qid, qdata in merged_data.items():
    candidates = qdata.get('pointer_candidates', [])
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    total_cand += len(candidates)
    pos_cand += sum(1 for s in scores if s > 0)
    unique = set(scores)
    if len(unique) == 1:
        if abs(list(unique)[0]) < 1e-6: all0 += 1
        elif abs(list(unique)[0] - 1.0) < 1e-6: all1 += 1
        else: diverse += 1
    else: diverse += 1

print(f"\n=== 激进补全后数据状态 ===")
print(f"总 query 数: {len(merged_data)}")
print(f"All-Zero: {all0} ({100*all0/len(merged_data):.1f}%)")
print(f"All-One: {all1} ({100*all1/len(merged_data):.1f}%)")
print(f"Diverse: {diverse} ({100*diverse/len(merged_data):.1f}%)")
print(f"总候选数: {total_cand}")
print(f"正样本比例: {100*pos_cand/total_cand:.1f}%")
EOF

echo "=========================================="
echo "✓ VQAv2 RL数据激进补全完成！"
echo "输出文件: results/vqav2/generated_data/rl_data_merged_aggressive.json"
echo "=========================================="
