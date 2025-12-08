# GRPO Post-Training 实施计划

## 项目概述

基于 `强化学习.md` 的设计方案，为 Lever-Plus 项目实现 v3 版本的 GRPO（Group-Relative Policy Optimization）后训练功能。

---

## 当前进度（来自强化学习.md 1.3节）

- ✅ **束搜索数据生成**：已完成，支持多种采样器（RandSampler, TextSimSampler, ImgSimSampler, MixSampler）
- ✅ **SFT训练**：v0/v1/v2/v2_lora版本已完成训练和推理
- ✅ **v3代码框架**：`pointer_selector_v3.py` 已实现RCE和GRPO核心算法
- ⚠️ **v3训练脚本**：缺少完整的post-training训练流程
- ⚠️ **数据加载**：需要从束搜索JSON文件中提取beam和score作为reward

---

## 束搜索数据结构（来自强化学习.md 1.4节）

每个样本包含：
- `id_list`: 5个beam，每个beam是一个shot序列（如`[7232, 2229, 8211]`）
- `score_list`: 5个beam对应的分数（如`[0.046, 0.045, 0.037, ...]`）

---

## 需要创建的文件（来自强化学习.md 2.4节）

```
lever_lm/
├── models/v3/
│   ├── pointer_selector_v3.py  # ✅ 已实现
│   └── dataset_v3.py           # ⚠️ 需要创建：加载beam数据
├── workflows/
│   └── grpo_post_train.py      # ⚠️ 需要创建：GRPO训练脚本
└── utils/
    └── reward_utils.py         # ⚠️ 需要创建：奖励处理工具

configs/
└── train/lever_lm/v3/
    └── grpo_post_train.yaml    # ⚠️ 需要创建：GRPO训练配置

scripts/
└── grpo_post_train.sh          # ⚠️ 需要创建：训练启动脚本
```

---

## 实施步骤

### 步骤 1：验证现有 v3 模型代码

**目标**：确认 `pointer_selector_v3.py` 已正确实现（来自强化学习.md 1.3节）

**具体任务**：
- 检查 `lever_lm/models/v3/pointer_selector_v3.py` 是否存在
- 验证RCE和GRPO核心算法已实现

**测试方法**：
```bash
cd /mnt/share/yiyun/Projects/Lever-Plus
ls -la lever_lm/models/v3/pointer_selector_v3.py
python -m lever_lm.models.v3.pointer_selector_v3
```

**预期结果**：
- 文件存在
- 模型可正常实例化
- 核心算法可运行

---

### 步骤 2：实现 Beam 数据集加载器 (dataset_v3.py)

**目标**：创建专用数据集类，加载束搜索数据（来自强化学习.md 2.4节）

**具体任务**：
- 创建 `lever_lm/models/v3/dataset_v3.py`
- 实现数据加载，支持：
  - 从JSON文件读取`id_list`和`score_list`（来自强化学习.md 阶段1：数据准备）
  - 解析 `id_list`（5个beam，每个beam是shot序列如`[7232, 2229, 8211]`）
  - 解析 `score_list`（5个beam对应的分数如`[0.046, 0.045, 0.037, ...]`）
- 构建训练样本（来自强化学习.md 2.2节 阶段1）：
  - Query: 原始query的embedding
  - Candidates: 候选池的embedding
  - Labels: beam中的shot序列
  - Rewards: beam的分数（归一化）
  - Old_log_probs: 从SFT模型计算（冻结参数）

**测试方法**：
```bash
python -c "
from lever_lm.models.v3.dataset_v3 import BeamDataset
import json

# 加载现有束搜索数据
data_path = 'results/okvqa/generated_data/vqa-okvqa_local-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800.json'
with open(data_path) as f:
    data = json.load(f)

ds = BeamDataset(data, beam_size=5)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Sample keys: {sample.keys()}')
# 验证包含所有必要字段
assert 'beam_labels' in sample  # Labels: beam中的shot序列
assert 'beam_rewards' in sample  # Rewards: beam的分数
print('Dataset validation passed!')
"
```

**预期结果**：
- 数据集可正常加载
- 每个样本包含5个beam的标签和奖励

---

### 步骤 3：实现奖励处理工具 (reward_utils.py)

**目标**：创建奖励归一化和优势计算工具（来自强化学习.md 2.4节）

**具体任务**（来自强化学习.md 2.3节 奖励归一化策略）：
- 创建 `lever_lm/utils/reward_utils.py`
- 实现以下函数：
  - `normalize_rewards_zscore()` - **组内Z-score**：在每个query的5个beam内计算均值和标准差
  - `clip_advantages()` - **优势裁剪**：限制在[-5, 5]范围内，防止极端梯度
  - `compute_softmax_weights()` - 计算RCE的softmax权重 `w_i = softmax(score_i / τ)`（来自强化学习.md 阶段2）

**测试方法**：
```bash
python -c "
from lever_lm.utils.reward_utils import (
    normalize_rewards_zscore, 
    clip_advantages,
    compute_softmax_weights
)
import torch

# 测试组内Z-score归一化
rewards = torch.tensor([[0.046, 0.045, 0.037, 0.035, 0.030]])
normalized = normalize_rewards_zscore(rewards)
print(f'Normalized rewards: {normalized}')

# 测试优势裁剪到[-5, 5]
advantages = torch.tensor([[-10.0, 5.0, 0.0, 3.0, 8.0]])
clipped = clip_advantages(advantages)
print(f'Clipped advantages: {clipped}')
assert clipped.min() >= -5 and clipped.max() <= 5

# 测试softmax权重
weights = compute_softmax_weights(rewards, temperature=2.0)
print(f'Softmax weights (τ=2.0): {weights}')
print('All tests passed!')
"
```

**预期结果**：
- 归一化后均值接近0，标准差接近1
- 裁剪后值在[-5, 5]范围内
- softmax权重和为1

---

### 步骤 4：创建 GRPO 训练配置文件 (grpo_post_train.yaml)

**目标**：创建Hydra配置文件（来自强化学习.md 2.4节）

**具体任务**：
- 创建 `configs/train/lever_lm/v3/grpo_post_train.yaml`

**配置参数来源**：

**阶段2：RCE预热参数**（来自强化学习.md 2.2节）：
- `rce_epochs`: 1-2
- `rce_temperature_start`: 2.0（τ从2.0线性降到0.5）
- `rce_temperature_end`: 0.5
- `rce_lr_ratio`: 0.1（使用较小的学习率，如SFT的1/10）

**阶段3：GRPO训练参数**（来自强化学习.md 2.2节）：
- `grpo_epochs`: 2-5
- `clip_epsilon`: 0.2（PPO裁剪参数ε，用于clip(r, 1-ε, 1+ε)）
- `kl_beta`: 初始值（KL散度权重β）

**创新点4：课程学习参数**（来自强化学习.md 2.1节）：
- 阶段1（RCE预热）：使用所有beam，softmax加权
- 阶段2（GRPO早期）：只使用top-3 beam，减少噪声
- 阶段3（GRPO后期）：使用所有beam，精细优化

**创新点5：KL散度自适应参数**（来自强化学习.md 2.1节）：
- `kl_target_max`: 0.1（如果偏离过大>0.1，增加kl_beta）
- `kl_target_min`: 0.01（如果KL过小<0.01，减少kl_beta）

**稳定性保障参数**（来自强化学习.md 2.3节）：
- `gradient_clip_norm`: 1.0（梯度裁剪max_norm=1.0）
- `warmup_ratio`: 0.1（warmup=10%）
- `lr_scheduler`: cosine（余弦退火）

**测试方法**：
```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/train/lever_lm/v3/grpo_post_train.yaml')
print(OmegaConf.to_yaml(cfg))
# 验证关键参数存在
assert 'rce_epochs' in cfg
assert 'grpo_epochs' in cfg
assert 'kl_beta' in cfg
assert 'gradient_clip_norm' in cfg
print('Config validation passed!')
"
```

**预期结果**：配置文件可正常加载，所有参数有合理默认值

---

### 步骤 5：实现 GRPO 训练入口脚本 (grpo_post_train.py)

**目标**：创建完整的训练流程脚本（来自强化学习.md 2.4节）

**具体任务**：
- 创建 `lever_lm/workflows/grpo_post_train.py`

**实现详细训练流程**（来自强化学习.md 2.2节）：

**阶段1：数据准备**
- 加载束搜索数据：从JSON文件读取`id_list`和`score_list`
- 构建训练样本：Query, Candidates, Labels, Rewards, Old_log_probs

**阶段2：RCE预热（1-2 epochs）**
- 目标：稳定地从监督学习过渡到强化学习
- 损失：`L_RCE = Σ w_i * CE(π_new, labels_i)`，其中`w_i = softmax(score_i / τ)`
- 温度调度：τ从2.0线性降到0.5
- 学习率：使用较小的学习率（如SFT的1/10）

**阶段3：GRPO训练（2-5 epochs）**
- 目标：最大化高分beam的概率，同时保持策略稳定性
- 损失：`L_GRPO = L_PPO + β * L_KL`
  - `L_PPO = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)]`
  - `L_KL = E[r - 1 - log(r)]`（近似KL散度）
- 优势计算：组内相对优势（每个query的5个beam内归一化）
- KL自适应：根据KL散度动态调整β

**实现创新点**（来自强化学习.md 2.1节）：

**创新点1：多层级奖励设计**
- Beam级奖励：直接使用束搜索分数（info score）
- 序列级奖励：考虑整个序列的连贯性和多样性
- 任务级奖励：端到端VQA准确率（可选，需要额外评估）

**创新点2：自适应温度调度**
- RCE阶段：温度从高到低（τ: 2.0 → 0.5），逐步聚焦高分beam
- GRPO阶段：根据KL散度动态调整温度，平衡探索与利用

**创新点3：组内相对优势（Group-Relative Advantage）**
- 在每个query的5个beam内计算相对优势
- 避免跨query的奖励分布差异影响训练
- 更稳定的梯度信号

**创新点4：课程学习策略**
- 阶段1（RCE预热）：使用所有beam，softmax加权
- 阶段2（GRPO早期）：只使用top-3 beam，减少噪声
- 阶段3（GRPO后期）：使用所有beam，精细优化

**创新点5：KL散度自适应调整**
- 监控KL散度，如果偏离过大（>0.1），增加kl_beta
- 如果KL过小（<0.01），减少kl_beta，允许更大更新

**实现稳定性保障**（来自强化学习.md 2.3节）：
- 梯度裁剪：max_norm=1.0
- 学习率调度：余弦退火，warmup=10%
- 检查点保存：每个epoch保存，保留最佳KL散度的模型

**实现监控指标**（来自强化学习.md 2.3节）：
- 训练指标：PPO loss, KL loss, mean ratio, mean advantage
- 验证指标：Val loss, KL散度, 优势分布
- 下游指标：VQA准确率（可选，需要额外推理）

**测试方法**：
```bash
# 使用小数据集进行快速测试
python lever_lm/workflows/grpo_post_train.py \
    --sft_ckpt results/okvqa/model_cpk/v2/test.ckpt \
    --beam_data results/okvqa/generated_data/test.json \
    --rce_epochs 1 \
    --grpo_epochs 1 \
    --max_samples 100 \
    --dry_run
```

**预期结果**：
- 脚本可正常启动
- 数据加载成功
- RCE和GRPO训练循环可正常执行

---

### 步骤 6：创建训练启动脚本 (grpo_post_train.sh)

**目标**：创建便捷的bash启动脚本（来自强化学习.md 2.4节和2.5节）

**具体任务**：
- 创建 `scripts/grpo_post_train.sh`
- 支持命令行参数（来自强化学习.md 2.5节使用流程）：
  - `--sft_ckpt`: SFT模型检查点路径
  - `--beam_data`: 束搜索数据路径
  - `--rce_epochs`: RCE预热轮数
  - `--grpo_epochs`: GRPO训练轮数
  - `--output_dir`: 输出目录

**使用示例**（来自强化学习.md 2.5节）：
```bash
# 1. 完成SFT训练（v2版本）
bash scripts/train_lever_lm.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v2

# 2. 进行GRPO post-training
bash scripts/grpo_post_train.sh \
    --sft_ckpt results/okvqa/model_cpk/v2/xxx.ckpt \
    --beam_data results/okvqa/generated_data/xxx.json \
    --rce_epochs 1 \
    --grpo_epochs 3 \
    --output_dir results/okvqa/model_cpk/v3/
```

**测试方法**：
```bash
# 显示帮助信息
bash scripts/grpo_post_train.sh --help

# 验证脚本语法
bash -n scripts/grpo_post_train.sh
```

**预期结果**：脚本语法正确，帮助信息完整

---

### 步骤 7：RCE 预热阶段集成测试

**目标**：验证RCE预热阶段完整流程（来自强化学习.md 阶段2）

**具体任务**：
- 加载预训练的V2 SFT模型
- 加载束搜索数据
- 执行1个epoch的RCE预热
- 验证：
  - 损失：`L_RCE = Σ w_i * CE(π_new, labels_i)`
  - 温度调度：τ从2.0线性降到0.5
  - 学习率：SFT的1/10

**测试方法**：
```bash
bash scripts/grpo_post_train.sh \
    --sft_ckpt results/okvqa/model_cpk/v2/best_model.ckpt \
    --beam_data results/okvqa/generated_data/xxx.json \
    --rce_epochs 1 \
    --grpo_epochs 0 \
    --output_dir results/okvqa/model_cpk/v3/rce_test/
```

**预期结果**：
- RCE损失在训练过程中下降
- 温度从2.0降到0.5
- 检查点成功保存
- 无NaN/Inf错误

---

### 步骤 8：GRPO 训练阶段集成测试

**目标**：验证GRPO训练阶段完整流程（来自强化学习.md 阶段3）

**具体任务**：
- 从RCE检查点继续
- 执行1个epoch的GRPO训练
- 验证：
  - 损失：`L_GRPO = L_PPO + β * L_KL`
  - 优势计算：组内相对优势
  - KL自适应：KL>0.1增加β，KL<0.01减少β
  - 课程学习：GRPO早期只用top-3 beam

**测试方法**：
```bash
bash scripts/grpo_post_train.sh \
    --sft_ckpt results/okvqa/model_cpk/v3/rce_test/rce_final.ckpt \
    --beam_data results/okvqa/generated_data/xxx.json \
    --rce_epochs 0 \
    --grpo_epochs 1 \
    --output_dir results/okvqa/model_cpk/v3/grpo_test/
```

**预期结果**：
- GRPO损失稳定
- KL散度在合理范围内（0.01 ~ 0.1）
- 无训练崩溃

---

### 步骤 9：端到端完整训练测试

**目标**：验证完整的RCE + GRPO训练流程

**具体任务**：
- 执行完整的两阶段训练（来自强化学习.md 2.5节）
- 监控所有训练指标（来自强化学习.md 2.3节）：
  - PPO loss, KL loss, mean ratio, mean advantage
- 保存最佳模型（基于KL散度）

**测试方法**：
```bash
bash scripts/grpo_post_train.sh \
    --sft_ckpt results/okvqa/model_cpk/v2/best_model.ckpt \
    --beam_data results/okvqa/generated_data/xxx.json \
    --rce_epochs 1 \
    --grpo_epochs 3 \
    --output_dir results/okvqa/model_cpk/v3/full_train/
```

**预期结果**：
- 训练完成无错误
- 最佳模型已保存
- 训练日志完整

---

### 步骤 10：推理脚本适配

**目标**：更新推理脚本以支持V3模型

**具体任务**：
- 修改 `predict.py` 支持加载V3模型
- 修改 `icl_inference.py` 支持V3推理
- 更新相关配置文件

**测试方法**：
```bash
python predict.py \
    --model_version v3 \
    --checkpoint results/okvqa/model_cpk/v3/full_train/best_model.ckpt \
    --test_data results/okvqa/test_data.json
```

**预期结果**：
- V3模型可正常加载
- 推理结果格式正确

---

### 步骤 11：VQA 准确率评估

**目标**：评估V3模型在下游VQA任务上的表现（来自强化学习.md 2.3节下游指标）

**具体任务**：
- 使用V3模型选择范例
- 在OKVQA测试集上评估
- 与V2基线对比

**测试方法**：
```bash
bash scripts/inference.sh \
    --model_version v3 \
    --checkpoint results/okvqa/model_cpk/v3/full_train/best_model.ckpt \
    --dataset okvqa \
    --output results/okvqa/icl_inference/v3/
```

**预期结果**：
- VQA准确率 ≥ V2基线
- 推理时间合理

---

## 核心创新点实现检查清单（来自强化学习.md 2.1节）

- [ ] **创新点1：多层级奖励设计**
  - [ ] Beam级奖励：直接使用束搜索分数（info score）
  - [ ] 序列级奖励：考虑整个序列的连贯性和多样性
  - [ ] 任务级奖励：端到端VQA准确率（可选，需要额外评估）

- [ ] **创新点2：自适应温度调度**
  - [ ] RCE阶段：温度从高到低（τ: 2.0 → 0.5），逐步聚焦高分beam
  - [ ] GRPO阶段：根据KL散度动态调整温度，平衡探索与利用

- [ ] **创新点3：组内相对优势（Group-Relative Advantage）**
  - [ ] 在每个query的5个beam内计算相对优势
  - [ ] 避免跨query的奖励分布差异影响训练
  - [ ] 更稳定的梯度信号

- [ ] **创新点4：课程学习策略**
  - [ ] 阶段1（RCE预热）：使用所有beam，softmax加权
  - [ ] 阶段2（GRPO早期）：只使用top-3 beam，减少噪声
  - [ ] 阶段3（GRPO后期）：使用所有beam，精细优化

- [ ] **创新点5：KL散度自适应调整**
  - [ ] 监控KL散度，如果偏离过大（>0.1），增加kl_beta
  - [ ] 如果KL过小（<0.01），减少kl_beta，允许更大更新

---

## 创新性亮点总结（来自强化学习.md 第三节）

1. **多层级奖励**：结合beam级、序列级、任务级奖励
2. **自适应调度**：温度、KL权重、课程学习三管齐下
3. **组内相对优势**：更稳定的梯度信号
4. **稳定性保障**：多重机制防止训练崩溃
5. **端到端优化**：可选的任务级奖励，直接优化VQA准确率

---

## 依赖关系

```
步骤 1（验证v3模型）
    ↓
步骤 2（dataset_v3.py）──┐
    ↓                   │
步骤 3（reward_utils.py）├──→ 步骤 5（grpo_post_train.py）
    ↓                   │           ↓
步骤 4（grpo_post_train.yaml）──────┘
                                    ↓
                        步骤 6（grpo_post_train.sh）
                                    ↓
                        步骤 7（RCE测试）
                                    ↓
                        步骤 8（GRPO测试）
                                    ↓
                        步骤 9（完整测试）
                                    ↓
                        步骤 10（推理适配）
                                    ↓
                        步骤 11（VQA评估）
```

---

## 注意事项（来自强化学习.md 2.3节 稳定性保障）

1. **梯度裁剪**：max_norm=1.0
2. **学习率调度**：余弦退火，warmup=10%
3. **检查点保存**：每个epoch保存，保留最佳KL散度的模型
4. **优势裁剪**：限制在[-5, 5]范围内，防止极端梯度
5. **组内Z-score**：在每个query的5个beam内计算均值和标准差

---

## 进度跟踪

| 步骤 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 1 | ✅ 已完成 | 2025-12-02 | pointer_selector_v3.py 验证通过 |
| 2 | ✅ 已完成 | 2025-12-02 | dataset_v3.py 验证通过 |
| 3 | ✅ 已完成 | 2025-12-02 | reward_utils.py 验证通过 |
| 4 | ✅ 已完成 | 2025-12-02 | grpo_post_train.yaml 创建完成 |
| 5 | ✅ 已完成 | 2025-12-02 | grpo_post_train.py 创建完成 |
| 6 | ✅ 已完成 | 2025-12-02 | grpo_post_train.sh 创建完成 |
| 7 | ✅ 已完成 | 2025-12-02 | RCE集成测试通过 |
| 8 | ✅ 已完成 | 2025-12-02 | GRPO集成测试通过 |
| 9 | ✅ 已完成 | 2025-12-02 | 端到端测试通过 |
| 10 | ✅ 已完成 | 2025-12-02 | 推理适配测试通过 |
| 11 | ✅ 已完成 | 2025-12-02 | VQA评估脚本测试通过 |
