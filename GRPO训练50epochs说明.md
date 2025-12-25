# GRPO 训练 50 Epochs 说明

> 日期：2025-12-23  
> 目标：训练 50 个 GRPO epochs，找到最优 checkpoint 进行推理

---

## 一、训练配置

### 训练参数

```bash
export USE_RANK_ADVANTAGE=false  # 方案五：关闭 Rank Normalization
export GRPO_EPOCHS=50            # 训练 50 个 epochs
export KL_BETA=0.1               # KL 权重
```

### 训练命令

```bash
bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
```

### 输出目录

```
./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/
├── rce_epoch1.pt
├── rce_epoch2.pt
├── ...
├── rce_epoch5.pt
├── grpo_epoch1.pt
├── grpo_epoch2.pt
├── ...
└── grpo_epoch50.pt
```

---

## 二、监控训练进度

### 方法 1：查看训练日志（推荐）

训练日志会实时输出到终端，包含每个 epoch 的指标：

```
阶段3：GRPO训练
Epoch Train Loss    Val Loss    PPO Loss      KL    Adv Std   Adv Max     Beta
--------------------------------------------------------------------------------
    1    0.01234    0.01123    0.00123   0.08123   0.5234    1.2345   0.1000
    2    0.01198    0.01098    0.00115   0.07543   0.5432    1.3456   0.1000
...
```

### 方法 2：检查 checkpoint 文件

```bash
# 查看已保存的 checkpoint
ls -lh ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch*.pt

# 查看最新的 checkpoint
ls -t ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch*.pt | head -1
```

### 方法 3：查看训练进程

```bash
# 查看训练进程
ps aux | grep grpo_post_train

# 查看 GPU 使用情况
nvidia-smi
```

---

## 三、找到最优 Checkpoint

训练完成后，使用辅助脚本找到最优 checkpoint：

### 方法 1：使用辅助脚本（推荐）

```bash
# 自动查找最优 checkpoint（默认使用 val_loss 策略）
python scripts/find_best_checkpoint.py \
    --checkpoint_dir ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct

# 显示所有 epoch 的指标
python scripts/find_best_checkpoint.py \
    --checkpoint_dir ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct \
    --show_all

# 使用其他策略
python scripts/find_best_checkpoint.py \
    --checkpoint_dir ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct \
    --strategy adv_std  # 选择 advantage 标准差最大的
```

### 选择策略说明

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `val_loss` | 验证集损失最小（默认） | 推荐，最稳定 |
| `ppo_loss` | PPO 损失最小 | 关注策略优化效果 |
| `kl` | KL 散度适中（0.01-0.1） | 关注策略稳定性 |
| `adv_std` | Advantage 标准差最大 | 关注梯度信号强度 |

### 方法 2：手动查看训练日志

如果脚本无法解析日志，可以手动查看训练输出，找到：
- **Val Loss 最小**的 epoch
- **PPO Loss 最小**的 epoch
- **KL 散度适中**（0.01-0.1）的 epoch

---

## 四、使用最优 Checkpoint 进行推理

找到最优 checkpoint 后（假设是 `grpo_epoch25.pt`），进行推理：

### 方法 1：使用环境变量指定 checkpoint

```bash
# 设置 checkpoint 路径
export LEVER_LM_CHECKPOINT_PATH=./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch25_v2format.ckpt

# 如果还没有转换为 v2 格式，先转换
python scripts/convert_v3_to_v2_format.py \
    --v3_ckpt ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch25.pt

# 然后推理
bash scripts/inference.sh vqa okvqa_local 5 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3
```

### 方法 2：直接指定 checkpoint 文件

推理脚本会自动查找最新的 checkpoint，但如果想使用特定的 checkpoint：

```bash
# 方法 A：重命名 checkpoint 为最新
cd ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct
cp grpo_epoch25.pt grpo_epoch50.pt  # 让推理脚本找到它

# 方法 B：使用环境变量（推荐）
export LEVER_LM_CHECKPOINT_PATH=./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch25_v2format.ckpt
bash scripts/inference.sh vqa okvqa_local 5 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3
```

---

## 五、完整流程示例

```bash
# Step 1: 检查训练是否完成
ls -lh ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch*.pt | wc -l
# 应该显示 50 个文件

# Step 2: 找到最优 checkpoint
python scripts/find_best_checkpoint.py \
    --checkpoint_dir ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct \
    --show_all

# Step 3: 假设最优是 epoch 25，转换为 v2 格式
python scripts/convert_v3_to_v2_format.py \
    --v3_ckpt ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch25.pt

# Step 4: 使用最优 checkpoint 推理
export LEVER_LM_CHECKPOINT_PATH=./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/grpo_epoch25_v2format.ckpt
bash scripts/inference.sh vqa okvqa_local 5 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3
```

---

## 六、注意事项

1. **训练时间**：50 个 epochs 可能需要数小时，请耐心等待
2. **磁盘空间**：确保有足够的磁盘空间（每个 checkpoint 约几十 MB）
3. **GPU 内存**：确保 GPU 内存充足
4. **训练中断**：如果训练中断，可以从最新的 checkpoint 继续训练（需要修改训练脚本）

---

## 七、故障排查

### 问题 1：训练日志无法解析

**解决方案**：
- 手动查看训练输出，找到最优 epoch
- 或者直接使用最新的 checkpoint（`grpo_epoch50.pt`）

### 问题 2：找不到最优 checkpoint

**解决方案**：
- 检查 checkpoint 目录是否正确
- 确认训练已完成（有 50 个 checkpoint 文件）
- 尝试使用不同的策略（`--strategy`）

### 问题 3：推理时找不到 checkpoint

**解决方案**：
- 确认 checkpoint 已转换为 v2 格式（`*_v2format.ckpt`）
- 使用环境变量 `LEVER_LM_CHECKPOINT_PATH` 明确指定路径

---

*文档生成时间：2025-12-23*


