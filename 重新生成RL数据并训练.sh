#!/bin/bash
# 重新生成RL数据并重新训练
# 使用方法: bash 重新生成RL数据并训练.sh

echo "=========================================="
echo "重新生成 RL 数据并重新训练"
echo "=========================================="

# 停止所有正在运行的训练任务
echo "1. 停止所有训练任务..."
pkill -f "train_v3.sh.*okvqa_local" 2>/dev/null || true
sleep 2
echo "✓ 已停止所有训练任务"
echo ""

# 删除旧的RL数据文件（缺少 vqa_eval_mode 字段）
echo "2. 删除旧的 RL 数据文件..."
rm -f ./results/okvqa/generated_data/rl_data_RandSampler_Qwen2_5-VL-3B-Instruct.json
rm -f ./results/okvqa/generated_data/rl_data_TextSimSampler_Qwen2_5-VL-3B-Instruct.json
rm -f ./results/okvqa/generated_data/rl_data_ImgSimSampler_Qwen2_5-VL-3B-Instruct.json
rm -f ./results/okvqa/generated_data/rl_data_MixSampler_Qwen2_5-VL-3B-Instruct.json
echo "✓ 已删除旧的 RL 数据文件"
echo ""

# 重新启动训练（会自动重新生成RL数据）
echo "3. 重新启动训练任务..."
echo ""

# 创建日志目录
mkdir -p logs

# 1. RandSampler (GPU 0)
echo "启动 RandSampler 训练任务（GPU 0）..."
nohup bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B > logs/train_rand.log 2>&1 &
echo "  PID: $!"
echo "  日志: logs/train_rand.log"
echo ""

# 2. TextSimSampler (GPU 1)
echo "启动 TextSimSampler 训练任务（GPU 1）..."
nohup bash scripts/train_v3.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler qwen2.5_vl_3B > logs/train_text.log 2>&1 &
echo "  PID: $!"
echo "  日志: logs/train_text.log"
echo ""

# 3. ImgSimSampler (GPU 2)
echo "启动 ImgSimSampler 训练任务（GPU 2）..."
nohup bash scripts/train_v3.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler qwen2.5_vl_3B > logs/train_img.log 2>&1 &
echo "  PID: $!"
echo "  日志: logs/train_img.log"
echo ""

# 4. MixSampler (GPU 3)
echo "启动 MixSampler 训练任务（GPU 3）..."
nohup bash scripts/train_v3.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler qwen2.5_vl_3B > logs/train_mix.log 2>&1 &
echo "  PID: $!"
echo "  日志: logs/train_mix.log"
echo ""

echo "=========================================="
echo "✓ 所有训练任务已重新启动"
echo "=========================================="
echo ""
echo "查看日志："
echo "  tail -f logs/train_rand.log"
echo "  tail -f logs/train_text.log"
echo "  tail -f logs/train_img.log"
echo "  tail -f logs/train_mix.log"
echo ""





