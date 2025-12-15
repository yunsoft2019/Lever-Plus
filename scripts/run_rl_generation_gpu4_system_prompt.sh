#!/bin/bash
# 在 GPU4 上运行 RL 数据生成（修复system_prompt版本）

cd /mnt/share/yiyun/Projects/Lever-Plus

export CUDA_VISIBLE_DEVICES=4

# 激活 conda 环境
source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

# 运行 RL 数据生成（使用修复后的代码：system_prompt + max_new_tokens=25）
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=12_train=21.93316_val=21.99296.ckpt \
    --beam_data results/okvqa/generated_data/beam_data_50queries.json \
    --output_path results/okvqa/generated_data/rl_data_v4_50queries_system_prompt.json \
    --query_emb results/okvqa/cache/query_embeddings.pt \
    --cand_emb results/okvqa/cache/candidate_embeddings.pt \
    --vqa_model qwen2.5_vl_3B \
    --dataset okvqa_local \
    --num_beams 5 \
    --temps 1.0 1.3 \
    --num_samples_per_temp 2 \
    --num_random 1 \
    --device cuda:0 \
    --train_ques_path datasets/okvqa/OpenEnded_mscoco_train2014_questions.json \
    --train_ann_path datasets/okvqa/mscoco_train2014_annotations.json \
    > results/okvqa/generated_data/rl_data_v4_50queries_system_prompt.log 2>&1

