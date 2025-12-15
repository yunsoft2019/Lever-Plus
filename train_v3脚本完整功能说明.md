# train_v3.sh 脚本完整功能说明

> 更新时间：2025-12-10  
> 功能：一键完成 v3 训练的所有必要步骤

---

## 🎯 核心功能

**一条命令完成所有步骤**：

```bash
bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
```

脚本会自动执行以下步骤（**智能检测，只做必要的工作**）：

---

## 📋 自动执行的步骤

### Step 0: 检查并导出 Embeddings（如果不存在）

**检查**：
- `./results/okvqa/cache/query_embeddings.pt`
- `./results/okvqa/cache/candidate_embeddings.pt`

**如果不存在**：
- 自动查找 v2 checkpoint
- 调用 `export_embeddings.sh` 导出 embeddings
- 如果找不到 v2 checkpoint，提示错误并退出

**如果已存在**：
- 跳过导出步骤，直接使用现有文件

---

### Step 1: 检查并生成 RL 数据（如果不存在）

**检查**：
- `./results/okvqa/generated_data/rl_data_RandSampler_Qwen2_5-VL-3B-Instruct.json`

**如果不存在**：
- 自动调用 `generate_rl_data_for_sampler.sh` 生成 RL 数据
- 使用指定的 sampler 和 beam_model

**如果已存在**：
- 跳过生成步骤，直接使用现有文件

---

### Step 2: 执行 GRPO 强化学习训练

**自动执行**：
- 创建输出目录
- 调用 `grpo_post_train.py` 进行训练
- 支持所有环境变量配置（RCE_EPOCHS, GRPO_EPOCHS, 等）
- 支持 3.4 和 3.5.2 新功能（`--rce_use_normalized_reward`, `--freeze_backbone_in_grpo`）

**输出**：
- RCE checkpoints: `rce_epoch1.pt` ~ `rce_epoch5.pt`
- GRPO checkpoints: `grpo_epoch1.pt` ~ `grpo_epoch3.pt`（如果 GRPO_EPOCHS > 0）

---

### Step 3: 自动转换为 v2 格式（如果不存在）

**检查**：
- 如果 GRPO_EPOCHS=0：检查 `rce_epoch5_v2format.ckpt`
- 如果 GRPO_EPOCHS>0：检查 `grpo_epoch3_v2format.ckpt`

**如果不存在**：
- 自动调用 `convert_v3_to_v2_format.py` 转换 checkpoint
- 转换后的文件保存在同一目录，文件名格式：`xxx_v2format.ckpt`

**如果已存在**：
- 跳过转换步骤，直接使用现有文件

---

## 🔧 智能特性

### 1. 自动检测依赖

- ✅ 自动检查 v2 checkpoint（用于导出 embeddings）
- ✅ 自动检查 embeddings（如果不存在则导出）
- ✅ 自动检查 RL 数据（如果不存在则生成）
- ✅ 自动检查 v2format 文件（如果不存在则转换）

### 2. 智能路径处理

- ✅ 自动根据 sampler 和 beam_model 构建正确的文件路径
- ✅ 自动处理模型名称转换（qwen2.5_vl_3B → Qwen2_5-VL-3B-Instruct）
- ✅ 自动处理数据集名称转换（okvqa_local → okvqa）

### 3. 智能 checkpoint 选择

- ✅ RCE-only 模式（GRPO_EPOCHS=0）：自动选择 `rce_epoch5.pt`
- ✅ RCE + GRPO 模式：自动选择 `grpo_epoch3.pt`
- ✅ 如果推荐 checkpoint 不存在，自动查找最新的 `.pt` 文件

---

## 📝 使用示例

### 基础使用（RCE-only baseline）

```bash
# 一条命令完成所有步骤
bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

# 脚本会自动：
# 1. 检查 embeddings（如果不存在则导出）
# 2. 检查 RL 数据（如果不存在则生成）
# 3. 执行 RCE-only 训练（GRPO_EPOCHS=0）
# 4. 转换为 v2 格式（如果不存在）
```

### 自定义参数

```bash
# 使用归一化后的 reward
export RCE_USE_NORMALIZED_REWARD=true
bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

# GRPO 时冻结 backbone
export GRPO_EPOCHS=3 FREEZE_BACKBONE_IN_GRPO=true
bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
```

---

## 🎯 输出文件

### 训练输出

```
results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/
├── rce_epoch1.pt
├── rce_epoch2.pt
├── rce_epoch3.pt
├── rce_epoch4.pt
├── rce_epoch5.pt                    # RCE-only baseline（推荐）
└── rce_epoch5_v2format.ckpt        # 自动转换的 v2 格式（用于推理）
```

### 如果 GRPO_EPOCHS > 0

```
results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct/
├── rce_epoch1.pt ~ rce_epoch5.pt
├── grpo_epoch1.pt
├── grpo_epoch2.pt
├── grpo_epoch3.pt                  # RCE + GRPO（推荐）
└── grpo_epoch3_v2format.ckpt      # 自动转换的 v2 格式（用于推理）
```

---

## ✅ 优势

1. **一键完成**：一条命令完成所有步骤，无需手动执行多个脚本
2. **智能检测**：自动检测依赖，只做必要的工作，节省时间
3. **错误处理**：如果关键依赖不存在（如 v2 checkpoint），会明确提示
4. **自动转换**：训练完成后自动转换为 v2 格式，可直接用于推理
5. **向后兼容**：支持所有环境变量配置，保持灵活性

---

## 🔍 执行流程示例

```bash
$ bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

==========================================
V3 训练配置
==========================================
Task: vqa
Dataset: okvqa_local → okvqa
GPU ID: 0
Sampler: rand_sampler → RandSampler
Beam Model: qwen2.5_vl_3B → Qwen2_5-VL-3B-Instruct
==========================================
训练参数:
  RCE Epochs: 5
  GRPO Epochs: 0
  → RCE-only baseline 模式（符合文档 Step 3 建议）
  ...

==========================================
Step 0: 检查 Embeddings
==========================================
✓ Embeddings 已存在，跳过导出
  - Query: ./results/okvqa/cache/query_embeddings.pt
  - Candidate: ./results/okvqa/cache/candidate_embeddings.pt

==========================================
Step 1: 检查 RL 数据
==========================================
✓ RL 数据已存在，跳过生成
  - RL Data: ./results/okvqa/generated_data/rl_data_RandSampler_Qwen2_5-VL-3B-Instruct.json

==========================================
Step 2: 执行 GRPO 强化学习训练
==========================================
[训练过程...]

==========================================
Step 3: 检查并转换 checkpoint 格式
==========================================
v2format 文件不存在，开始转换...
  v3 checkpoint: rce_epoch5.pt
  目标路径: rce_epoch5_v2format.ckpt
✓ 转换成功: rce_epoch5_v2format.ckpt

==========================================
✓ V3 训练完成！
==========================================
Checkpoint 保存在: ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct
  - RCE checkpoints: rce_epoch1.pt ~ rce_epoch5.pt
  - 推荐使用: rce_epoch5.pt (RCE-only baseline)
  - v2format: rce_epoch5_v2format.ckpt (可用于推理)

推理命令（自动转换格式）:
  bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3
==========================================
```

---

**更新时间：** 2025-12-10





