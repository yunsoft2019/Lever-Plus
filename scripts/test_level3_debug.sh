#!/bin/bash
# Level-3策略调试脚本（GPU7）
# 用于测试Level-3策略的执行情况，并查看调试日志

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
    if [ -n "$OKVQA_PATH" ] && [[ ! "$OKVQA_PATH" = /* ]]; then
        export OKVQA_PATH="${PROJECT_ROOT}/${OKVQA_PATH}"
    fi
    if [ -n "$COCO_PATH" ] && [[ ! "$COCO_PATH" = /* ]]; then
        export COCO_PATH="${PROJECT_ROOT}/${COCO_PATH}"
    fi
fi

# 设置GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "=========================================="
echo "Level-3策略调试测试（GPU7）"
echo "=========================================="
echo ""
echo "环境变量:"
echo "  OKVQA_PATH=${OKVQA_PATH:-未设置}"
echo "  COCO_PATH=${COCO_PATH:-未设置}"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# 输入文件（使用all0_only数据）
INPUT_FILE="results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all0_only.json"
OUTPUT_FILE="results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all0_only_diverse_debug.json"
REPORT_FILE="results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_all0_only_diverse_debug_report.json"

echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "报告文件: $REPORT_FILE"
echo ""

# 设置路径
QUERY_EMB="${PROJECT_ROOT}/results/okvqa/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/okvqa/cache/candidate_embeddings.pt"
SFT_CKPT="${PROJECT_ROOT}/results/okvqa/model_cpk/v3_RandSampler_v2data/rce_epoch5_v2format.ckpt"
VQA_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset/okvqa_local.yaml"

# 运行Python脚本（只处理前10个query用于调试）
python3 -m lever_lm.models.v3.augment_rl_data_diversity \
    --input_rl_data "$INPUT_FILE" \
    --query_embeddings "$QUERY_EMB" \
    --candidate_embeddings "$CAND_EMB" \
    --output_path "$OUTPUT_FILE" \
    --report_path "$REPORT_FILE" \
    --sft_ckpt "$SFT_CKPT" \
    --vqa_model "$VQA_MODEL" \
    --dataset_config "$DATASET_CONFIG" \
    --max_queries 10 \
    --max_candidates_per_query 24 \
    --max_eval_budget_all0 50 \
    --max_eval_budget_all1 8 \
    --max_eval_budget_all06 8 \
    --add_timestamp 2>&1 | tee /tmp/level3_debug.log

echo ""
echo "=========================================="
echo "调试日志已保存到: /tmp/level3_debug.log"
echo "=========================================="
echo ""
echo "查看Level-3策略相关日志："
echo "grep 'DEBUG Level-3' /tmp/level3_debug.log"
