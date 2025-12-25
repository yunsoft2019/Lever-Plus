#!/bin/bash
# 评估 KL_BETA=0.15 解冻版本的 GRPO 模型

set -e

PROJECT_ROOT="/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME="okvqa"

# 根据参数决定评估哪个 checkpoint 和样本数
GRPO_EPOCH=${1:-1}  # GRPO epoch (1, 2, 3)
GPU_ID=${2:-0}      # GPU ID
TEST_NUM=${3:-100}  # 测试样本数 (100, 200, 400, 800)

# KL_BETA=0.15 解冻版本的 checkpoint 路径
CKPT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl015_unfrozen"
PT_PATH="${CKPT_DIR}/grpo_epoch${GRPO_EPOCH}.pt"
CKPT_PATH="${CKPT_DIR}/grpo_epoch${GRPO_EPOCH}_v2format.ckpt"

# 检查 .pt 文件是否存在
if [ ! -f "$PT_PATH" ]; then
    echo "错误: GRPO checkpoint 不存在: ${PT_PATH}"
    echo "请先完成 GRPO 训练（KL_BETA=0.15，解冻版本）"
    exit 1
fi

# 如果 v2format 文件不存在，自动转换
if [ ! -f "$CKPT_PATH" ]; then
    echo "=========================================="
    echo "自动转换 v3 checkpoint 为 v2 格式..."
    echo "=========================================="
    echo "  v3 checkpoint: $(basename ${PT_PATH})"
    echo "  目标路径: $(basename ${CKPT_PATH})"
    
    if [ ! -f "${PROJECT_ROOT}/scripts/convert_v3_to_v2_format.py" ]; then
        echo "✗ 错误: 转换脚本不存在: scripts/convert_v3_to_v2_format.py"
        exit 1
    fi
    
    cd "${PROJECT_ROOT}"
    source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
    conda activate lever_env
    
    if python scripts/convert_v3_to_v2_format.py --v3_ckpt "${PT_PATH}"; then
        if [ -f "${CKPT_PATH}" ]; then
            echo "✓ 转换成功: $(basename ${CKPT_PATH})"
        else
            echo "✗ 警告: 转换脚本执行成功，但未找到输出文件"
            exit 1
        fi
    else
        echo "✗ 转换失败（退出码: $?），请检查错误信息"
        exit 1
    fi
fi

source /mnt/share/yiyun/anaconda3/etc/profile.d/conda.sh
conda activate lever_env

cd "${PROJECT_ROOT}"

export LEVER_LM_CHECKPOINT_PATH="${CKPT_PATH}"

echo "=========================================="
echo "GRPO Epoch ${GRPO_EPOCH} 评估 (KL_BETA=0.15，解冻版本)"
echo "GPU ${GPU_ID}: ${TEST_NUM} 样本, shot 1-4"
echo "Checkpoint: ${CKPT_PATH}"
echo "=========================================="

# 使用 inference.sh 进行评估（会自动测试 shot 1-4）
bash scripts/inference.sh vqa okvqa_local ${GPU_ID} query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3 ${TEST_NUM}

echo ""
echo "✓ GPU ${GPU_ID} 完成: ${TEST_NUM} 样本评估 (GRPO Epoch ${GRPO_EPOCH}, KL_BETA=0.15，解冻版本)"

