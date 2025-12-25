#!/bin/bash
# 使用 Epoch 2 checkpoint 在 GPU 0 上运行 800 个样本的推理
# 使用方法: bash scripts/inference_epoch2_800samples.sh

# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH=./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch2_v2format.ckpt

# 检查 checkpoint 是否存在
if [ ! -f "$LEVER_LM_CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint 文件不存在: $LEVER_LM_CHECKPOINT_PATH"
    echo "正在转换 checkpoint..."
    python scripts/convert_v3_to_v2_format.py \
        --v3_ckpt ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch2.pt
    if [ ! -f "$LEVER_LM_CHECKPOINT_PATH" ]; then
        echo "转换失败，请检查错误信息"
        exit 1
    fi
    echo "✓ Checkpoint 转换完成"
fi

echo "=========================================="
echo "使用 Epoch 2 checkpoint 进行推理"
echo "=========================================="
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo "GPU: 0"
echo "测试样本数: 800"
echo "=========================================="

# 运行推理（800 个样本）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 800

echo ""
echo "=========================================="
echo "推理完成！"
echo "=========================================="

