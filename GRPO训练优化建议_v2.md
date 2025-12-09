# GRPO 训练优化建议 V2

## 问题诊断

根据 `新强化学习结果.md` 的实验数据，新方案（hard_plus_soft）相比旧方案（legacy/InfoScore）存在以下问题：

### 核心发现

| Shot Num | 新方案 vs 旧方案 | 趋势 |
|----------|-----------------|------|
| 1 | **-2.0% ~ -2.6%** | 明显下降 |
| 2 | **-0.4% ~ -1.2%** | 略有下降 |
| 3 | **+0.15% ~ +2.0%** | 略有提升 |
| 4 | **+0.1% ~ +1.0%** | 略有提升 |

### 问题分析

1. **低 shot 数时新方案效果差**：shot_num=1,2 时下降明显
2. **整体不如 v2 baseline**：新 v3 在大多数情况下都低于或等于 v2
3. **旧方案在低 shot 时更好**：v3_1layer (legacy) 在 shot_num≤2 时超越 v2

---

## 根因分析

### 1. Reward 设计问题

**当前新方案**：
```python
reward = hard_weight * vqa_correct + soft_weight * vqa_acc_score
# 范围: [0, 2]，正样本 [1, 2]，负样本 [0, 1)
```

**问题**：
- `vqa_correct` 是 0/1 二值，区分度不够
- `vqa_acc_score` 通常在 [0, 1] 范围，但分布可能不均匀
- 正负样本的 reward 差距可能不够大（正样本最低 1.0，负样本最高接近 1.0）

**旧方案（InfoScore）**：
```python
reward = InfoScore（增益）
# 直接使用 beam search 的信息增益分数
```

**优势**：InfoScore 是连续的、有区分度的分数，能更好地反映候选序列的质量差异。

### 2. Advantage 计算问题

查看 `pointer_selector_v3.py` 中的 `compute_advantage` 方法：

```python
def compute_advantage(self, rewards, normalize=True, use_rank=True):
    if use_rank:
        # 使用排名归一化
        ranks = rewards.argsort(dim=-1, descending=True).argsort(dim=-1).float()
        advantages = 1.0 - 2.0 * ranks / (num_beams - 1)
    else:
        # Z-score 归一化
        advantages = (rewards - mean) / std
```

**问题**：
- 排名归一化丢失了原始 reward 的绝对差异信息
- 当所有候选都是负样本（vqa_correct=0）时，排名归一化仍会产生 [-1, 1] 的 advantage
- 这可能导致模型学习到错误的信号

### 3. RCE Loss 计算问题

```python
def compute_rce_loss(self, ..., use_rank_normalization=True):
    if use_rank_normalization:
        # 使用排名计算权重
        ranks = beam_rewards.argsort(dim=-1).argsort(dim=-1).float()
        normalized_scores = ranks / (num_beams - 1)
        weights = F.softmax(normalized_scores / temperature, dim=-1)
```

**问题**：
- 排名归一化后，所有 query 的权重分布相同
- 丢失了不同 query 之间 reward 差异的信息

---

## 优化方案

### 方案 1：改进 Reward 设计（推荐）

**核心思想**：增大正负样本的 reward 差距，同时保留连续性。

```python
# 新的 reward 公式
def compute_reward_v2(vqa_correct, vqa_acc_score, beam_score=None):
    # 基础分：使用 vqa_acc_score 作为连续信号
    base_reward = vqa_acc_score  # [0, 1]
    
    # 正确性加成：正确答案获得额外奖励
    correctness_bonus = 2.0 * vqa_correct  # 0 或 2
    
    # 组合：正样本 [2, 3]，负样本 [0, 1]
    reward = base_reward + correctness_bonus
    
    return reward
```

**实现**：在 `reward_utils.py` 中添加新模式 `"hard_plus_soft_v2"`

```python
elif reward_mode == "hard_plus_soft_v2":
    # 增大正负样本差距
    soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
    hard = float(vqa_correct) if vqa_correct is not None else 0.0
    # 正样本: [2, 3], 负样本: [0, 1]
    reward = soft + 2.0 * hard
```

### 方案 2：混合 Reward 策略

**核心思想**：结合 InfoScore 和 correctness 信号。

```python
# 混合 reward 公式
def compute_reward_hybrid(beam_score, vqa_correct, vqa_acc_score, alpha=0.5):
    # InfoScore 部分（归一化到 [0, 1]）
    info_reward = normalize_beam_score(beam_score)
    
    # Correctness 部分
    correct_reward = vqa_correct + 0.5 * vqa_acc_score  # [0, 1.5]
    
    # 混合
    reward = alpha * info_reward + (1 - alpha) * correct_reward
    
    return reward
```

**实现**：添加新模式 `"hybrid"`

### 方案 3：改进 Advantage 计算

**核心思想**：保留原始 reward 的绝对差异信息。

```python
def compute_advantage_v2(self, rewards, normalize=True):
    # 1. 计算组内均值
    mean = rewards.mean(dim=-1, keepdim=True)
    
    # 2. 计算相对优势（不使用排名）
    advantages = rewards - mean
    
    # 3. 可选：缩放到合理范围
    if normalize:
        std = rewards.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=0.1)  # 设置最小 std，避免除零
        advantages = advantages / std
    
    # 4. 裁剪
    advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
    
    return advantages
```

### 方案 4：条件性 GRPO 训练

**核心思想**：只在有正样本的 query 上进行 GRPO 训练。

```python
def compute_grpo_loss_conditional(self, ..., min_positive_ratio=0.2):
    # 检查每个 query 的正样本比例
    positive_mask = (beam_rewards > threshold).float()
    positive_ratio = positive_mask.mean(dim=-1)
    
    # 只在有足够正样本的 query 上训练
    valid_mask = positive_ratio >= min_positive_ratio
    
    if valid_mask.sum() == 0:
        return {"loss": torch.tensor(0.0)}
    
    # 只计算有效 query 的损失
    ...
```

### 方案 5：回归 Legacy 模式 + 微调

**核心思想**：既然 legacy 模式在低 shot 时效果更好，可以基于 legacy 进行微调。

```bash
# 训练配置
export REWARD_MODE=legacy
export REWARD_ALPHA=0.0  # 不使用 beam_score
export REWARD_BETA=1.0   # 只使用 correctness
```

---

## 具体实施建议

### 短期优化（立即可行）

1. **切换回 legacy 模式**：
```bash
bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
# 设置环境变量
export REWARD_MODE=legacy
```

2. **调整 hard/soft 权重**：
```bash
export REWARD_MODE=hard_plus_soft
export HARD_WEIGHT=2.0  # 增大正确性权重
export SOFT_WEIGHT=1.0
```

### 中期优化（需要代码修改）

1. **实现 `hard_plus_soft_v2` 模式**
2. **改进 advantage 计算逻辑**
3. **添加条件性训练机制**

### 长期优化（需要实验验证）

1. **混合 reward 策略实验**
2. **不同 shot_num 使用不同策略**
3. **自适应 reward 权重**

---

## 代码修改建议

### 1. 修改 `reward_utils.py`

```python
def compute_reward_for_candidate(..., reward_mode="hard_plus_soft", ...):
    # 新增 hard_plus_soft_v2 模式
    if reward_mode == "hard_plus_soft_v2":
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
        # 增大正负样本差距：正样本 [2, 3]，负样本 [0, 1]
        reward = soft + 2.0 * hard
    
    # 新增 hybrid 模式
    elif reward_mode == "hybrid":
        # 结合 beam_score 和 correctness
        info_score = float(beam_score) if beam_score is not None else 0.0
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
        
        # 归一化 info_score（假设范围 [0, 0.1]）
        info_normalized = min(1.0, info_score * 10)
        
        # 混合
        alpha = hard_weight  # 复用 hard_weight 作为混合系数
        reward = alpha * info_normalized + (1 - alpha) * (hard + 0.5 * soft)
```

### 2. 修改 `pointer_selector_v3.py`

```python
def compute_advantage(self, rewards, normalize=True, use_rank=False):  # 默认关闭排名归一化
    if use_rank:
        # 保持原有逻辑
        ...
    else:
        # 改进的 Z-score 归一化
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=0.1)  # 设置最小 std
        advantages = (rewards - mean) / std
    
    advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
    return advantages
```

### 3. 修改 `train_v3.sh`

```bash
# 添加新的环境变量支持
reward_mode=${REWARD_MODE:-hard_plus_soft_v2}  # 默认使用改进版
use_rank_advantage=${USE_RANK_ADVANTAGE:-false}  # 默认关闭排名归一化
```

---

## 实验计划

### 实验 1：Reward 模式对比

| 实验 | Reward Mode | 预期效果 |
|------|-------------|----------|
| 1.1 | legacy | 基线（已知在低 shot 时效果好） |
| 1.2 | hard_plus_soft | 当前新方案 |
| 1.3 | hard_plus_soft_v2 | 增大正负样本差距 |
| 1.4 | hybrid | 混合 InfoScore 和 correctness |

### 实验 2：Advantage 计算对比

| 实验 | use_rank | 预期效果 |
|------|----------|----------|
| 2.1 | True | 当前方案（排名归一化） |
| 2.2 | False | 改进方案（Z-score 归一化） |

### 实验 3：权重调优

| 实验 | hard_weight | soft_weight | 预期效果 |
|------|-------------|-------------|----------|
| 3.1 | 1.0 | 1.0 | 当前默认 |
| 3.2 | 2.0 | 1.0 | 增大正确性权重 |
| 3.3 | 1.0 | 0.5 | 降低 soft 权重 |
| 3.4 | 2.0 | 0.5 | 组合调整 |

---

## 总结

根据实验结果分析，新方案在低 shot 数时效果下降的主要原因是：

1. **Reward 设计**：正负样本差距不够大
2. **Advantage 计算**：排名归一化丢失了绝对差异信息
3. **训练信号**：当所有候选都是负样本时，仍会产生学习信号

**推荐的优化路径**：

1. **短期**：切换回 legacy 模式，或调整 hard_weight=2.0
2. **中期**：实现 `hard_plus_soft_v2` 模式，关闭排名归一化
3. **长期**：实验混合 reward 策略，针对不同 shot_num 使用不同策略
