#!/bin/bash
# V3 RCE训练脚本（使用排名归一化修复）
# 
# 修复内容：RCE损失使用排名归一化，解决InfoScore分数差异过小的问题
# 评分标准：与V1/V2完全相同（InfoScore）

set -e

# ========== 配置 ==========
GPU=0
SFT_CKPT="results/okvqa/model_cpk/v2_lora/flamingo_3B_ImgSimSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=18_train=23.38194_val=23.63491-v1.ckpt"
BEAM_DATA="results/okvqa/generated_data/vqa-okvqa-flamingo_3B-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800.json"
IMG_EMB="results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-ImgFeatures.pth"
TEXT_EMB="results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-TextFeatures.pth"
OUTPUT_DIR="results/okvqa/model_cpk/v3_rce_fixed"

# 训练参数
RCE_EPOCHS=10
BATCH_SIZE=32
RCE_LR=5e-5

# ========== 运行 ==========
echo "======================================================"
echo "V3 RCE训练（排名归一化修复版）"
echo "======================================================"
echo "SFT检查点: $SFT_CKPT"
echo "Beam数据: $BEAM_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "RCE epochs: $RCE_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "======================================================"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.grpo_post_train \
    --sft_ckpt "$SFT_CKPT" \
    --beam_data "$BEAM_DATA" \
    --img_emb "$IMG_EMB" \
    --text_emb "$TEXT_EMB" \
    --output_dir "$OUTPUT_DIR" \
    --rce_epochs $RCE_EPOCHS \
    --grpo_epochs 0 \
    --batch_size $BATCH_SIZE \
    --rce_lr $RCE_LR \
    --device cuda:0

echo "======================================================"
echo "✓ 训练完成！"
echo "  检查点保存至: $OUTPUT_DIR"
echo "======================================================"
