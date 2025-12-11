#!/bin/bash
# V3 后台训练脚本
# 使用方法: bash scripts/train_v3_background.sh
#
# 该脚本会在后台执行4个sampler的训练任务：
#   1. RandSampler (GPU 0)
#   2. TextSimSampler (GPU 1)
#   3. ImgSimSampler (GPU 2)
#   4. MixSampler (GPU 3)
#
# 日志文件保存在: ./logs/train_v3_<sampler>_<timestamp>.log

set -e

# 创建日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "V3 后台训练任务启动"
echo "=========================================="
echo "日志目录: ${LOG_DIR}"
echo "时间戳: ${TIMESTAMP}"
echo ""

# 1. RandSampler (GPU 0)
echo "启动 RandSampler 训练任务（GPU 0）..."
LOG_FILE_RAND="${LOG_DIR}/train_v3_rand_sampler_${TIMESTAMP}.log"
nohup bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B > "$LOG_FILE_RAND" 2>&1 &
PID_RAND=$!
echo "  PID: ${PID_RAND}"
echo "  日志: ${LOG_FILE_RAND}"
echo ""

# 2. TextSimSampler (GPU 1)
echo "启动 TextSimSampler 训练任务（GPU 1）..."
LOG_FILE_TEXT="${LOG_DIR}/train_v3_text_sim_sampler_${TIMESTAMP}.log"
nohup bash scripts/train_v3.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler qwen2.5_vl_3B > "$LOG_FILE_TEXT" 2>&1 &
PID_TEXT=$!
echo "  PID: ${PID_TEXT}"
echo "  日志: ${LOG_FILE_TEXT}"
echo ""

# 3. ImgSimSampler (GPU 2)
echo "启动 ImgSimSampler 训练任务（GPU 2）..."
LOG_FILE_IMG="${LOG_DIR}/train_v3_img_sim_sampler_${TIMESTAMP}.log"
nohup bash scripts/train_v3.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler qwen2.5_vl_3B > "$LOG_FILE_IMG" 2>&1 &
PID_IMG=$!
echo "  PID: ${PID_IMG}"
echo "  日志: ${LOG_FILE_IMG}"
echo ""

# 4. MixSampler (GPU 3)
echo "启动 MixSampler 训练任务（GPU 3）..."
LOG_FILE_MIX="${LOG_DIR}/train_v3_mix_sampler_${TIMESTAMP}.log"
nohup bash scripts/train_v3.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler qwen2.5_vl_3B > "$LOG_FILE_MIX" 2>&1 &
PID_MIX=$!
echo "  PID: ${PID_MIX}"
echo "  日志: ${LOG_FILE_MIX}"
echo ""

echo "=========================================="
echo "所有训练任务已启动"
echo "=========================================="
echo ""
echo "任务信息："
echo "  1. RandSampler    - PID: ${PID_RAND} - GPU: 0 - 日志: ${LOG_FILE_RAND}"
echo "  2. TextSimSampler - PID: ${PID_TEXT} - GPU: 1 - 日志: ${LOG_FILE_TEXT}"
echo "  3. ImgSimSampler  - PID: ${PID_IMG}  - GPU: 2 - 日志: ${LOG_FILE_IMG}"
echo "  4. MixSampler     - PID: ${PID_MIX}  - GPU: 3 - 日志: ${LOG_FILE_MIX}"
echo ""
echo "查看日志命令："
echo "  # 查看所有日志（实时）"
echo "  tail -f ${LOG_DIR}/train_v3_*_${TIMESTAMP}.log"
echo ""
echo "  # 查看特定sampler的日志"
echo "  tail -f ${LOG_FILE_RAND}"
echo "  tail -f ${LOG_FILE_TEXT}"
echo "  tail -f ${LOG_FILE_IMG}"
echo "  tail -f ${LOG_FILE_MIX}"
echo ""
echo "  # 查看所有训练进程"
echo "  ps aux | grep train_v3.sh | grep -v grep"
echo ""
echo "  # 查看GPU使用情况"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "=========================================="
echo "提示：可以使用以下命令管理任务"
echo "=========================================="
echo "  # 查看所有训练进程"
echo "  ps aux | grep train_v3.sh | grep -v grep"
echo ""
echo "  # 停止所有训练任务（谨慎使用）"
echo "  pkill -f 'train_v3.sh.*okvqa_local'"
echo ""
echo "  # 停止特定GPU的训练任务"
echo "  pkill -f 'train_v3.sh.*okvqa_local.*[0-9].*rand_sampler'  # GPU 0"
echo "  pkill -f 'train_v3.sh.*okvqa_local.*[0-9].*text_sim_sampler'  # GPU 1"
echo "  pkill -f 'train_v3.sh.*okvqa_local.*[0-9].*img_sim_sampler'  # GPU 2"
echo "  pkill -f 'train_v3.sh.*okvqa_local.*[0-9].*mix_sampler'  # GPU 3"
echo ""

# 保存PID到文件（方便后续管理）
PID_FILE="${LOG_DIR}/train_v3_pids_${TIMESTAMP}.txt"
cat > "$PID_FILE" << EOF
# V3 训练任务 PID 记录
# 时间戳: ${TIMESTAMP}
RandSampler=${PID_RAND}
TextSimSampler=${PID_TEXT}
ImgSimSampler=${PID_IMG}
MixSampler=${PID_MIX}
EOF

echo "PID 已保存到: ${PID_FILE}"
echo ""
echo "✓ 所有任务已在后台启动，可以安全关闭终端"

