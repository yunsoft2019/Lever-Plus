#!/bin/bash
# 生成 RL 数据 v4（Reward 与最终评测严格一致 + 一次生成可复用）
# 
# 使用方法: 
#   bash scripts/generate_rl_data_v4.sh [GPU_ID] [SAMPLER] [OUTPUT_SUFFIX]
#
# 参数说明:
#   GPU_ID: GPU设备ID（默认: 4）
#   SAMPLER: sampler名称（默认: RandSampler）
#   OUTPUT_SUFFIX: 输出文件名后缀（默认: v4_strictEval）
#
# 示例:
#   bash scripts/generate_rl_data_v4.sh 4 RandSampler v4_strictEval
#   bash scripts/generate_rl_data_v4.sh 4 RandSampler v4_800queries_strictEval

set -e

# ========== 参数配置 ==========
GPU_ID=${1:-4}
SAMPLER=${2:-RandSampler}
OUTPUT_SUFFIX=${3:-v4_strictEval}

# ========== 路径配置 ==========
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa_local"
DATASET_NAME="okvqa"

# SFT模型checkpoint（v2格式）
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_${SAMPLER}_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt"

# Beam数据（输入）
BEAM_DATA="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-${SAMPLER}-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json"

# RL数据输出路径
OUTPUT_PATH="${PROJECT_ROOT}/results/${DATASET_NAME}/generated_data/rl_data_${SAMPLER}_${OUTPUT_SUFFIX}.json"

# Embeddings路径
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/${DATASET_NAME}/cache/candidate_embeddings.pt"

# VQA评测文件路径（用于strict_eval）
TRAIN_QUES_PATH="${PROJECT_ROOT}/datasets/okvqa/OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN_PATH="${PROJECT_ROOT}/datasets/okvqa/mscoco_train2014_annotations.json"
VAL_QUES_PATH="${PROJECT_ROOT}/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json"
VAL_ANN_PATH="${PROJECT_ROOT}/datasets/okvqa/mscoco_val2014_annotations.json"

# ========== 显示配置信息 ==========
echo "=========================================="
echo "生成 RL 数据 v4（严格一致 + 可复用）"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Sampler: ${SAMPLER}"
echo "SFT Checkpoint: ${SFT_CKPT}"
echo "Beam Data: ${BEAM_DATA}"
echo "Output Path: ${OUTPUT_PATH}"
echo "Query Embeddings: ${QUERY_EMB}"
echo "Candidate Embeddings: ${CAND_EMB}"
echo ""
echo "VQA评测文件："
echo "  Train Questions: ${TRAIN_QUES_PATH}"
echo "  Train Annotations: ${TRAIN_ANN_PATH}"
echo "  Val Questions: ${VAL_QUES_PATH}"
echo "  Val Annotations: ${VAL_ANN_PATH}"
echo ""
echo "数据生成配置："
echo "  - Beam候选数: 5"
echo "  - 温度采样: τ=1.0 x2 + τ=1.3 x2 = 4条"
echo "  - 随机组合: 1条"
echo "  - Retrieval: 5条"
echo "  - 每个query总计: ~15条候选"
echo ""
echo "新功能："
echo "  ✅ strict_eval: True（禁用fallback，确保与最终评测一致）"
echo "  ✅ 保存raw_generation（postprocess前）"
echo "  ✅ 保存gt_answers_raw/norm（query级别）"
echo "  ✅ 保存relevance指标（token_f1, edit_sim, rel_score）"
echo "  ✅ 保存pointer_pos和pointer(global)"
echo "  ✅ 保存_meta信息（创建时间、gen_args等）"
echo "  ⚠️  save_prompts: False（默认不保存prompt文本，可添加--save_prompts启用）"
echo "=========================================="

# ========== 检查文件是否存在 ==========
echo ""
echo "检查必需文件..."

if [ ! -f "$SFT_CKPT" ]; then
    echo "错误: SFT checkpoint 不存在: $SFT_CKPT"
    echo ""
    echo "尝试查找其他可用的 checkpoint..."
    FOUND_CKPT=$(find "${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v2" -name "*${SAMPLER}*.ckpt" -type f | head -1)
    if [ -n "$FOUND_CKPT" ]; then
        echo "找到: $FOUND_CKPT"
        SFT_CKPT="$FOUND_CKPT"
    else
        echo "未找到任何 v2 checkpoint，请先训练 v2 模型"
        exit 1
    fi
fi

if [ ! -f "$BEAM_DATA" ]; then
    echo "错误: Beam 数据文件不存在: $BEAM_DATA"
    exit 1
fi

if [ ! -f "$QUERY_EMB" ]; then
    echo "错误: Query embeddings 不存在: $QUERY_EMB"
    exit 1
fi

if [ ! -f "$CAND_EMB" ]; then
    echo "错误: Candidate embeddings 不存在: $CAND_EMB"
    exit 1
fi

# 检查VQA评测文件（strict_eval需要）
if [ ! -f "$TRAIN_QUES_PATH" ] && [ ! -f "$VAL_QUES_PATH" ]; then
    echo "警告: 未找到VQA评测文件（questions.json）"
    echo "  strict_eval模式需要至少一个评测文件"
    echo "  请检查路径是否正确"
fi

if [ ! -f "$TRAIN_ANN_PATH" ] && [ ! -f "$VAL_ANN_PATH" ]; then
    echo "警告: 未找到VQA评测文件（annotations.json）"
    echo "  strict_eval模式需要至少一个评测文件"
    echo "  请检查路径是否正确"
fi

echo "✓ 所有必需文件检查通过"
echo ""

# ========== 创建输出目录 ==========
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# ========== 执行生成 ==========
echo "开始生成RL数据..."
echo ""

cd "${PROJECT_ROOT}"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 激活conda环境（如果需要）
if [ -f "/mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh" ]; then
    source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
    conda activate lever_env 2>/dev/null || true
fi

# 运行RL数据生成
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt "${SFT_CKPT}" \
    --beam_data "${BEAM_DATA}" \
    --output_path "${OUTPUT_PATH}" \
    --query_emb "${QUERY_EMB}" \
    --cand_emb "${CAND_EMB}" \
    --vqa_model "qwen2.5_vl_3B" \
    --dataset "${DATASET}" \
    --num_beams 5 \
    --temps 1.0 1.3 \
    --num_samples_per_temp 2 \
    --num_random 1 \
    --num_retrieval 5 \
    --device cuda:0 \
    --train_ques_path "${TRAIN_QUES_PATH}" \
    --train_ann_path "${TRAIN_ANN_PATH}" \
    --val_ques_path "${VAL_QUES_PATH}" \
    --val_ann_path "${VAL_ANN_PATH}" \
    --strict_eval \
    > "${OUTPUT_PATH%.json}.log" 2>&1

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ RL数据生成完成！"
    echo "=========================================="
    echo "输出文件: ${OUTPUT_PATH}"
    echo "日志文件: ${OUTPUT_PATH%.json}.log"
    echo ""
    echo "数据格式（v4）："
    echo "  - _meta: 包含创建时间、gen_args、vqa_model等"
    echo "  - query: 包含gt_answers_raw/norm"
    echo "  - pointer_candidates: 包含"
    echo "    * pointer_pos 和 pointer(global)"
    echo "    * vqa_raw_generation（postprocess前）"
    echo "    * vqa_pred_answer（postprocess后）"
    echo "    * vqa_acc_score, vqa_correct, vqa_gt_prob"
    echo "    * vqa_rel_token_f1, vqa_rel_edit_sim, vqa_rel_score"
    echo "    * vqa_eval_mode, eval_split_used, eval_failed"
    echo ""
    echo "下一步："
    echo "  1. 检查日志文件确认生成成功"
    echo "  2. 使用新数据训练v3模型："
    echo "     bash scripts/train_v3_with_new_rl_data.sh ${GPU_ID} ${SAMPLER} ${OUTPUT_PATH}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ RL数据生成失败！"
    echo "=========================================="
    echo "请查看日志文件: ${OUTPUT_PATH%.json}.log"
    echo "=========================================="
    exit 1
fi

