#!/bin/bash
# 用Qwen2.5-VL评估V2模型（与V3公平对比）

set -e

GPU=0
V2_CKPT="results/okvqa/model_cpk/v2_lora/flamingo_3B_ImgSimSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=18_train=23.38194_val=23.63491-v1.ckpt"
# 使用512维embedding（与V2检查点匹配）
IMG_EMB="results/okvqa/cache/vqa-okvqa-clip-vit-base-patch32-ImgFeatures.pth"
BEAM_DATA="results/okvqa/generated_data/vqa-okvqa-flamingo_3B-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800.json"

echo "======================================================"
echo "用Qwen2.5-VL评估V2模型（与V3公平对比）"
echo "======================================================"
echo "V2检查点: $V2_CKPT"
echo "======================================================"

CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt "$V2_CKPT" \
    --img_emb "$IMG_EMB" \
    --beam_data "$BEAM_DATA" \
    --dataset okvqa \
    --device cuda:0 \
    --test_num 500

echo "======================================================"
echo "✓ V2评估完成！"
echo "======================================================"
