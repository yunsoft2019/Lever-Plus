#!/bin/bash
# 公平对比V2和V3（100条, 1-shot）
# 使用相同的评估逻辑

set -e
GPU=0
TEST_NUM=100
SHOT_NUM=1

echo "======================================================"
echo "V2 vs V3 公平对比"
echo "测试数量: $TEST_NUM, Shot数: $SHOT_NUM"
echo "======================================================"

# V2 使用 inference.sh
echo ""
echo ">>> 评估 V2 (使用inference.sh)..."
export LEVER_LM_CHECKPOINT_PATH="results/okvqa/model_cpk/v2_lora/flamingo_3B_ImgSimSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=18_train=23.38194_val=23.63491-v1.ckpt"
CUDA_VISIBLE_DEVICES=$GPU python icl_inference.py \
    train=query_img_icd_img_text_v2 \
    ex_name=v2_eval \
    dataset=okvqa \
    task=vqa \
    device=cuda:0 \
    inference_bs=1 \
    test_data_num=$TEST_NUM \
    test_lever_lm=true \
    infer_model=qwen2.5_vl_3B \
    infer_model.load_from_local=false \
    train.lever_lm.shot_num=$SHOT_NUM 2>&1 | grep -E "VQA|accuracy|准确率" | tail -5
unset LEVER_LM_CHECKPOINT_PATH

# V3 使用 inference_v3.py
echo ""
echo ">>> 评估 V3 Top-1 Only (使用inference_v3.py)..."
CUDA_VISIBLE_DEVICES=$GPU python scripts/inference_v3.py \
    --v3_ckpt "results/okvqa/model_cpk/v3_top1_512dim/rce_epoch10.pt" \
    --img_emb "results/okvqa/cache/vqa-okvqa-clip-vit-base-patch32-ImgFeatures.pth" \
    --dataset okvqa \
    --model_name "Qwen2.5-VL-3B-Instruct" \
    --shot_num $SHOT_NUM \
    --device cuda:0 \
    --test_num $TEST_NUM 2>&1 | grep -E "VQA准确率"

echo ""
echo "======================================================"
echo "✓ 对比完成！"
echo "======================================================"
