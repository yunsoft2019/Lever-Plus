#!/bin/bash
# V4-1 方案：使用 GRPO Epoch 2 checkpoint 进行推理（Val Loss 最小）
# 使用方法: bash scripts/inference_v4_1_epoch2_800samples.sh [gpu_id]

set -e

# 解析参数
gpu_id=${1:-0}

# 项目根目录
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# V4-1 checkpoint 路径
V3_CKPT="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_1/grpo_epoch2.pt"
V2_CKPT="${PROJECT_DIR}/results/okvqa/model_cpk/v3_plan_v4_1/grpo_epoch2_v2format.ckpt"

echo "=========================================="
echo "V4-1 方案推理（GRPO Epoch 2）"
echo "=========================================="
echo "Val Loss: 6.89825（最小）"
echo "GPU: ${gpu_id}"
echo "测试样本数: 800"
echo "=========================================="

# 检查 v3 checkpoint 是否存在
if [ ! -f "$V3_CKPT" ]; then
    echo "错误: V3 Checkpoint 文件不存在: $V3_CKPT"
    exit 1
fi

# 转换为 v2 格式（如果不存在）
if [ ! -f "$V2_CKPT" ]; then
    echo "正在转换 checkpoint 为 v2 格式..."
    python ${PROJECT_DIR}/scripts/convert_v3_to_v2_format.py \
        --v3_ckpt "$V3_CKPT" \
        --output "$V2_CKPT"
    if [ ! -f "$V2_CKPT" ]; then
        echo "转换失败，请检查错误信息"
        exit 1
    fi
    echo "✓ Checkpoint 转换完成"
fi

# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH="$V2_CKPT"

echo ""
echo "Checkpoint: $LEVER_LM_CHECKPOINT_PATH"
echo ""

# 运行推理（800 个样本，shot 1-4）
# 注意：shot_num_list 默认为 [1,2,3,4]，Python 会自动遍历所有 shot
# 不需要外层 shell 循环，否则会重复执行 4 次
cd ${PROJECT_DIR}

echo "=========================================="
echo "Running Shot 1-4..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=${gpu_id} python icl_inference.py \
    train="query_img_text_icd_img_text_v2" \
    ex_name="main_vqa_RandSampler_Qwen2_5_VL_3B_Instruct_query_img_text_icd_img_text" \
    dataset=okvqa_local \
    task=vqa \
    device=cuda:0 \
    inference_bs=1 \
    test_data_num=800 \
    test_lever_lm=true \
    infer_model=qwen2.5_vl_3B \
    infer_model.load_from_local=false

echo ""
echo "=========================================="
echo "✓ V4-1 推理完成！"
echo "=========================================="
echo "请查看结果并与方案五对比"
echo "=========================================="
