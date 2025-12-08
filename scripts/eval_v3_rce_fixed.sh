#!/bin/bash
# 评估V3 RCE修复版模型

set -e

GPU=0
CKPT="results/okvqa/model_cpk/v3_rce_fixed/rce_epoch10.pt"
IMG_EMB="results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-ImgFeatures.pth"
BEAM_DATA="results/okvqa/generated_data/vqa-okvqa-flamingo_3B-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800.json"

echo "======================================================"
echo "评估V3 RCE修复版模型"
echo "======================================================"
echo "检查点: $CKPT"
echo "======================================================"

CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt "$CKPT" \
    --img_emb "$IMG_EMB" \
    --beam_data "$BEAM_DATA" \
    --dataset okvqa \
    --device cuda:0 \
    --test_num 500

echo "======================================================"
echo "✓ 评估完成！"
echo "======================================================"
