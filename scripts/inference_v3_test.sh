#!/bin/bash
# 推理脚本：测试v3训练结果
# 使用方法: bash scripts/inference_v3_test.sh [GPU_ID] [TEST_DATA_NUM]
#
# 参数:
#   GPU_ID: GPU编号（默认: 4）
#   TEST_DATA_NUM: 测试数据数量（默认: 100，-1表示全部）

set -e

GPU_ID=${1:-4}
TEST_DATA_NUM=${2:-100}

# 路径配置
PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET="okvqa_local"
DATASET_NAME="okvqa"
SAMPLER="rand_sampler"
SAMPLER_NAME="RandSampler"
BEAM_MODEL="qwen2.5_vl_3B"
MODEL_NAME="Qwen2_5-VL-3B-Instruct"
LEVER_LM="query_img_text_icd_img_text"

# Checkpoint路径（刚才训练的模型）
V3_CKPT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_test_v4_debug"
V3_PT_PATH="${V3_CKPT_DIR}/rce_epoch1.pt"
V2FORMAT_PATH="${V3_PT_PATH%.pt}_v2format.ckpt"

echo "=========================================="
echo "V3模型推理测试"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "测试数据数量: ${TEST_DATA_NUM}"
echo "Checkpoint目录: ${V3_CKPT_DIR}"
echo ""

# Step 1: 检查checkpoint是否存在
if [ ! -f "${V3_PT_PATH}" ]; then
    echo "✗ 错误: checkpoint不存在: ${V3_PT_PATH}"
    exit 1
fi
echo "✓ 找到v3 checkpoint: $(basename ${V3_PT_PATH})"

# Step 2: 转换checkpoint格式（如果需要）
if [ ! -f "${V2FORMAT_PATH}" ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: 转换checkpoint格式"
    echo "=========================================="
    echo "v2format文件不存在，开始转换..."
    echo "  源文件: $(basename ${V3_PT_PATH})"
    echo "  目标文件: $(basename ${V2FORMAT_PATH})"
    
    if [ ! -f "scripts/convert_v3_to_v2_format.py" ]; then
        echo "✗ 错误: 转换脚本不存在: scripts/convert_v3_to_v2_format.py"
        exit 1
    fi
    
    if CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/convert_v3_to_v2_format.py --v3_ckpt "${V3_PT_PATH}"; then
        if [ -f "${V2FORMAT_PATH}" ]; then
            echo "✓ 转换成功: $(basename ${V2FORMAT_PATH})"
        else
            echo "✗ 错误: 转换失败，文件不存在"
            exit 1
        fi
    else
        echo "✗ 错误: 转换失败"
        exit 1
    fi
else
    echo "✓ v2format文件已存在: $(basename ${V2FORMAT_PATH})"
    
    # 检查是否需要重新转换（如果.pt文件更新）
    pt_time=$(stat -c %Y "${V3_PT_PATH}" 2>/dev/null || echo 0)
    ckpt_time=$(stat -c %Y "${V2FORMAT_PATH}" 2>/dev/null || echo 0)
    if [ "${pt_time}" -gt "${ckpt_time}" ]; then
        echo "⚠️  .pt文件更新，重新转换..."
        rm -f "${V2FORMAT_PATH}"
        if CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/convert_v3_to_v2_format.py --v3_ckpt "${V3_PT_PATH}"; then
            echo "✓ 重新转换成功"
        else
            echo "✗ 错误: 重新转换失败"
            exit 1
        fi
    fi
fi

# Step 3: 运行推理
echo ""
echo "=========================================="
echo "Step 2: 运行推理"
echo "=========================================="

# 设置环境变量，让inference.sh使用我们的checkpoint
export LEVER_LM_CHECKPOINT_PATH="${V2FORMAT_PATH}"

# 运行推理（使用v3版本，但checkpoint路径已通过环境变量指定）
bash scripts/inference.sh \
    vqa \
    ${DATASET} \
    ${GPU_ID} \
    ${LEVER_LM} \
    ${SAMPLER} \
    ${BEAM_MODEL} \
    v3 \
    ${TEST_DATA_NUM}

echo ""
echo "=========================================="
echo "✓ 推理完成"
echo "=========================================="
echo "Checkpoint: ${V2FORMAT_PATH}"
echo "结果文件保存在: ./results/${DATASET_NAME}/results/"



