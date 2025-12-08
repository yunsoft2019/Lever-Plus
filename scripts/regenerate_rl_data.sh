#!/bin/bash
# 重新生成RL数据脚本
# 
# 修改说明：
# 1. 增加了真正的beam search，生成5个beam候选（之前只有1个greedy候选）
# 2. 保持温度采样4条（τ=1.0两条 + τ=1.3两条）
# 3. 保持随机组合1条
# 4. 每个query总共约10条候选（去重后可能略少）
#
# 使用方法：
#   bash scripts/regenerate_rl_data.sh [GPU_ID] [SAMPLER]
#
# 示例：
#   bash scripts/regenerate_rl_data.sh 0 rand_sampler
#   bash scripts/regenerate_rl_data.sh 1 text_sim_sampler

set -e

# 参数
GPU_ID=${1:-0}
SAMPLER=${2:-rand_sampler}

# 将 sampler 转换为大驼峰格式
case "$SAMPLER" in
    rand_sampler)
        SAMPLER_NAME="RandSampler"
        ;;
    text_sim_sampler)
        SAMPLER_NAME="TextSimSampler"
        ;;
    img_sim_sampler)
        SAMPLER_NAME="ImgSimSampler"
        ;;
    mix_sampler)
        SAMPLER_NAME="MixSampler"
        ;;
    *)
        SAMPLER_NAME="${SAMPLER}"
        ;;
esac

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa"
MODEL_NAME="Qwen2_5-VL-3B-Instruct"

# 文件路径
SFT_CKPT="${PROJECT_ROOT}/results/${DATASET}/model_cpk/v2/Qwen2_5_VL_3B_Instruct_${SAMPLER_NAME}_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt"
BEAM_DATA="${PROJECT_ROOT}/results/${DATASET}/generated_data/vqa-${DATASET}-${MODEL_NAME}-${SAMPLER_NAME}-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json"
OUTPUT_PATH="${PROJECT_ROOT}/results/${DATASET}/generated_data/rl_data_${SAMPLER_NAME}_v2.json"
QUERY_EMB="${PROJECT_ROOT}/results/${DATASET}/cache/query_embeddings.pt"
CAND_EMB="${PROJECT_ROOT}/results/${DATASET}/cache/candidate_embeddings.pt"

echo "=========================================="
echo "重新生成RL数据"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Sampler: ${SAMPLER} -> ${SAMPLER_NAME}"
echo "SFT Checkpoint: ${SFT_CKPT}"
echo "Beam Data: ${BEAM_DATA}"
echo "Output: ${OUTPUT_PATH}"
echo "Query Embeddings: ${QUERY_EMB}"
echo "Candidate Embeddings: ${CAND_EMB}"
echo "=========================================="
echo ""
echo "数据生成配置："
echo "  - Beam候选数: 5 (之前是1)"
echo "  - 温度采样: τ=1.0 x2 + τ=1.3 x2 = 4条"
echo "  - 随机组合: 1条"
echo "  - 每个query总计: ~10条候选"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "$SFT_CKPT" ]; then
    echo "错误: SFT checkpoint 不存在: $SFT_CKPT"
    echo ""
    echo "尝试查找其他可用的 checkpoint..."
    FOUND_CKPT=$(find "${PROJECT_ROOT}/results/${DATASET}/model_cpk/v2" -name "*${SAMPLER_NAME}*.ckpt" -type f | head -1)
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

echo ""
echo "开始生成RL数据..."
echo ""

# 运行数据生成
cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt "${SFT_CKPT}" \
    --beam_data "${BEAM_DATA}" \
    --output_path "${OUTPUT_PATH}" \
    --query_emb "${QUERY_EMB}" \
    --cand_emb "${CAND_EMB}" \
    --vqa_model "qwen2.5_vl_3B" \
    --dataset "okvqa_local" \
    --num_beams 5 \
    --temps 1.0 1.3 \
    --num_samples_per_temp 2 \
    --num_random 1 \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ RL数据生成完成！"
echo "输出文件: ${OUTPUT_PATH}"
echo "=========================================="
echo ""
echo "下一步：使用新数据训练v3模型"
echo "  bash scripts/train_v3_with_new_rl_data.sh ${GPU_ID} ${SAMPLER}"
