#!/bin/bash
# V3 Top-1 Only训练脚本（回归V2监督学习方式）
# 
# 方案：只使用Top-1 beam作为标签，与V2完全相同的监督学习
# 这应该能获得与V2相当的效果

set -e

# ========== 配置 ==========
GPU=0
SFT_CKPT="results/okvqa/model_cpk/v2_lora/flamingo_3B_ImgSimSampler_infoscore_left_beam5_shot2_cand64_sample800_epoch=18_train=23.38194_val=23.63491-v1.ckpt"
BEAM_DATA="results/okvqa/generated_data/vqa-okvqa-flamingo_3B-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800.json"
# 使用512维embedding（与V2检查点匹配）
IMG_EMB="results/okvqa/cache/vqa-okvqa-clip-vit-base-patch32-ImgFeatures.pth"
TEXT_EMB=""
OUTPUT_DIR="results/okvqa/model_cpk/v3_top1_512dim"

# 训练参数
RCE_EPOCHS=10
BATCH_SIZE=32
RCE_LR=1e-4

# ========== 运行 ==========
echo "======================================================"
echo "V3 Top-1 Only训练（回归V2监督学习方式）"
echo "======================================================"
echo "SFT检查点: $SFT_CKPT"
echo "Beam数据: $BEAM_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "RCE epochs: $RCE_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "模式: 只使用Top-1 beam"
echo "======================================================"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -m lever_lm.workflows.grpo_post_train \
    --sft_ckpt "$SFT_CKPT" \
    --beam_data "$BEAM_DATA" \
    --img_emb "$IMG_EMB" \
    --output_dir "$OUTPUT_DIR" \
    --rce_epochs $RCE_EPOCHS \
    --grpo_epochs 0 \
    --batch_size $BATCH_SIZE \
    --rce_lr $RCE_LR \
    --device cuda:0 \
    --use_top1_only

echo "======================================================"
echo "✓ 训练完成！"
echo "  检查点保存至: $OUTPUT_DIR"
echo "======================================================"
