#!/bin/bash
# 快速对比评估脚本（100条）

set -e
GPU=0
IMG_EMB="results/okvqa/cache/vqa-okvqa-clip-vit-base-patch32-ImgFeatures.pth"
BEAM_DATA="results/okvqa/generated_data/vqa-okvqa-flamingo_3B-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800.json"

echo "======================================================"
echo "快速对比评估（100条）"
echo "======================================================"

# V2 基线
echo ""
echo ">>> 评估 V2 基线..."
CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt "results/okvqa/model_cpk/v2_lora/flamingo_3B_ImgSimSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=18_train=23.38194_val=23.63491-v1.ckpt" \
    --img_emb "$IMG_EMB" \
    --beam_data "$BEAM_DATA" \
    --dataset okvqa \
    --device cuda:0 \
    --test_num 100 2>&1 | grep -E "VQA准确率"

# V3 Top-1 Only
echo ""
echo ">>> 评估 V3 Top-1 Only..."
CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt "results/okvqa/model_cpk/v3_top1_512dim/rce_epoch10.pt" \
    --img_emb "$IMG_EMB" \
    --beam_data "$BEAM_DATA" \
    --dataset okvqa \
    --device cuda:0 \
    --test_num 100 2>&1 | grep -E "VQA准确率"

# V3 多beam
echo ""
echo ">>> 评估 V3 多beam..."
CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.evaluate_v3 \
    --grpo_ckpt "results/okvqa/model_cpk/v3_multibeam_512dim/rce_epoch10.pt" \
    --img_emb "$IMG_EMB" \
    --beam_data "$BEAM_DATA" \
    --dataset okvqa \
    --device cuda:0 \
    --test_num 100 2>&1 | grep -E "VQA准确率"

echo ""
echo "======================================================"
echo "✓ 对比完成！"
echo "======================================================"
