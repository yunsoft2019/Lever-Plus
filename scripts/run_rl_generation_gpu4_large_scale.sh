#!/bin/bash
# 在 GPU4 上大规模生成 RL 数据（使用所有最新修复）

cd /mnt/share/yiyun/Projects/Lever-Plus

export CUDA_VISIBLE_DEVICES=4

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 大规模生成 RL 数据（使用retrieval方法 + system_prompt + max_new_tokens=25）
# 使用800个query的完整beam_data（RandSampler）
BEAM_DATA="results/okvqa/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json"
if [ ! -f "$BEAM_DATA" ]; then
    echo "错误: beam_data文件不存在: $BEAM_DATA"
    exit 1
fi
echo "使用800个query的完整beam_data: $BEAM_DATA"

python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt \
    --beam_data "$BEAM_DATA" \
    --output_path results/okvqa/generated_data/rl_data_v4_large_scale.json \
    --query_emb results/okvqa/cache/query_embeddings.pt \
    --cand_emb results/okvqa/cache/candidate_embeddings.pt \
    --vqa_model qwen2.5_vl_3B \
    --dataset okvqa_local \
    --num_beams 5 \
    --temps 1.0 1.3 \
    --num_samples_per_temp 2 \
    --num_random 1 \
    --num_retrieval 5 \
    --device cuda:0 \
    --train_ques_path datasets/okvqa/OpenEnded_mscoco_train2014_questions.json \
    --train_ann_path datasets/okvqa/mscoco_train2014_annotations.json \
    > results/okvqa/generated_data/rl_data_v4_large_scale.log 2>&1

