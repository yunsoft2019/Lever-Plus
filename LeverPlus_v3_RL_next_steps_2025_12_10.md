# Lever-Plus v3 强化学习现状 & 下一步改造方案（基于 2025-12-10 代码与结果）

> 基于你打包的当前仓库（含 RL 相关所有 Python 脚本和 `*.md` 文档），结合  
> `2025_12_10正确率.md` 的最新实验结果，对 **现状** 和 **下一步可以做的事** 做一次重新整理。  
> 重点只讨论 v3 / RL 主链路：`export_embeddings.py → generate_rl_data.py → dataset_v3.py → grpo_post_train.py → pointer_selector_v3.py`。

---

## 0. 一页纸总结

### 现在的真实状态

1. **数据 & Reward**
   - RL 数据已经是 **整条 pointer 序列的绝对 correctness** 驱动：
     - 每个候选 pointer 都有 `vqa_correct ∈ {0,1}` 和 `vqa_acc_score ∈ [0,1]`。
     - `reward_utils.compute_reward_for_candidate` 支持多种模式：`hard_plus_soft / hard_plus_soft_v2 / separated / hybrid / legacy` 等。
   - RL JSON 结构（`pointer_candidates`）和 `RLBeamDatasetWithEmbedding` 映射逻辑已经打通，并且：
     - pointer 索引使用 **严格映射**（`idx` 必须在 `candidate_indices` 中，否则直接 `KeyError`）。

2. **训练流程**
   - RCE 预热（阶段 2）& GRPO（阶段 3）都已经按设计实现。
   - 最新实验（`2025_12_10正确率.md`）里：
     - 使用 **RCE-only baseline**（`RCE_EPOCHS=5, GRPO_EPOCHS=0, reward_mode=hard_plus_soft`）。
     - 在 shot_num=2、不同测试规模（100 / 200 / 400 / 800）上，**v3(RCE-only) 明显优于 v2/v1/v0**。

3. **关键结论的更新**

- 过去的结论是：**新 reward（hard_plus_soft）+ GRPO 效果不如旧的 InfoScore 方案**（见 `新强化学习结果.md`）。
- 现在根据 **RCE-only** 的结果，需要修正结论：
  - ✅ **新 reward 本身是 OK 的，甚至在 RCE-only 情况下是最优的。**
  - ❌ 当前 **GRPO 阶段** 才是主要问题：在已有强 RCE 模型上继续做 GRPO，容易把性能拉低。

> 所以：**短期策略 = 把“RCE-only 模型”当成 v3 正式版，GRPO 只当作“实验功能”谨慎继续。**

---

## 1. 代码级现状梳理（基于仓库当前版本）

这一节只总结「已经做到什么」，方便后面指出还需要改什么。

### 1.1 RL 数据生成：`lever_lm/models/v3/generate_rl_data.py`

**你现在已经做到：**

1. 使用 `export_embeddings.py` 导出：
   - `query_embeddings.pt`：按训练集顺序，对每条样本抽取 query embedding。
   - `candidate_embeddings.pt`：和 query 一一对应（候选池=训练集本身）。

2. 在 `generate_rl_data.py` 中：
   - 加载 SFT pointer 模型（`load_sft_model`）。
   - 对每个 query：
     - 用 `generate_pointer_candidates_for_query` 生成 pointer 候选：
       - beam search（top beams）
       - 温度采样（多个 `tau`）
       - 随机组合（random pointers）
     - 对每个 pointer：
       - 调用 VQA 模型（如 Qwen2.5-VL-3B）做 2‑shot 推理。
       - 调用 `compute_vqa_accuracy` 得到：
         - `vqa_correct ∈ {0,1}`
         - `vqa_acc_score ∈ [0,1]`
       - 写回 RL JSON：
         ```jsonc
         {
           "pointer": [icd1, icd2],
           "beam_score": ...,
           "logprob_score": ...,
           "vqa_pred_answer": "...",
           "vqa_correct": 0/1,
           "vqa_acc_score": 0.x,
           "gen_method": "beam" | "sample" | "random"
         }
         ```

3. VQA 准确率计算：
   - 优先使用 **官方 VQA metric**：
     - 通过 `open_mmicl.metrics.vqa_metrics.VQA / VQAEval`，并使用 `vqa_train_cache / vqa_val_cache` 只加载一次标注文件（见 `RL数据生成优化说明.md`）。
   - 如果官方 metric 失败或没有标注文件时：
     - 回退到 `compute_vqa_accuracy` 内部的 **字符串匹配 fallback**。
   - 额外统计：
     - `file_metric_count / fallback_count` 以及比例，都在脚本末尾打印。

> ✅ RL 数据生成这一块，从「整条序列的硬+软正确率」视角看，设计和实现已经是对齐的。

---

### 1.2 Reward 设计：`lever_lm/utils/reward_utils.py`

当前核心函数：

```python
def compute_reward_for_candidate(
    beam_score=None,
    logprob_score=None,
    vqa_correct=None,
    vqa_acc_score=None,
    reward_mode="hard_plus_soft",
    hard_weight=1.0,
    soft_weight=1.0,
    alpha=0.0,
    beta=0.0,
    correctness_mode="01",
    use_logprob=False,
    reward_clip=(-5.0, 5.0),
):
    ...
```

支持模式（已在代码中实现）：

- `"hard_plus_soft"`：  
  `reward = hard_weight * vqa_correct + soft_weight * vqa_acc_score`，范围约 `[0, 2]`。
- `"hard_plus_soft_v2"`：  
  `reward = vqa_acc_score + 2 * vqa_correct`，正样本约 `[2,3]`，负样本 `[0,1]`。
- `"separated"`：  
  正样本 `2.0 + soft`，负样本 `soft`，**正负之间至少有 1.0 的 gap**。
- `"hard_only"` / `"soft_only"`：只用其中一个信号。
- `"hybrid"`：混合 InfoScore 和 correctness（参见 `v3改进方案.md`）。
- `"legacy"`：沿用旧的 InfoScore 线性组合。

附加工具函数：

- `normalize_rewards_zscore`, `compute_group_relative_advantage`, `compute_softmax_weights`, `compute_temperature_schedule`, `adaptive_kl_beta` 等。

> ✅ 从「能表达的 reward family」角度看，已经非常丰富，**目前瓶颈不是“没有更好的公式”，而是训练阶段（尤其 GRPO）把它“用坏了”**。

---

### 1.3 Dataset & Collate：`lever_lm/models/v3/dataset_v3.py`

**重要类：**

1. `BeamDataset / BeamDatasetWithEmbedding`  
   - 负责旧格式 `id_list + score_list` 的 beam 数据（InfoScore 用的）。

2. `RLBeamDatasetWithEmbedding`（这次的重点）：

   - 输入：RL JSON（包含 `pointer_candidates`）。
   - 每个 query 会产出：
     ```python
     {
       "query_id": int,
       "beam_labels": [[shot1_idx, shot2_idx], ...],  # 多个 pointer
       "beam_rewards": [r1, r2, ...],                 # 归一化前的 reward
       "beam_logprobs": [...],                        # old_log_probs
     }
     ```
   - 关键点：

     - ✅ **指针映射严格检查**：
       ```python
       mapped_pointer = []
       for idx in pointer:
           if idx not in self.cand_idx_to_pos:
               raise KeyError(...)
           mapped_pointer.append(self.cand_idx_to_pos[idx])
       ```
       不再有 `.get(idx, idx)` 这种 silent fallback。

     - ✅ **通过 `compute_reward_for_candidate` 统一计算 reward**，支持 `hard_plus_soft / separated / hybrid / legacy` 等。
     - ✅ 组内做 Z-score，并同时保留 `beam_rewards_raw`：
       ```python
       beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)
       if std > 1e-12:
           beam_rewards_normalized = (beam_rewards_raw - mean) / std
       ```
       - `beam_rewards`：默认是 **Z-score 后的 reward**。
       - `beam_rewards_raw`：原始 reward，供 RCE 使用。

3. `collate_fn_rl_v3`：
   - 目前假设 `batch_size=1`，每个 batch 就是一个 query 的全体 candidates：
     ```python
     {
       "query_id": int,
       "query_emb": [1, d],
       "cand_emb": [1, K, d],
       "beam_labels": [1, num_candidates, shot_num],
       "beam_rewards": [1, num_candidates],
       "beam_rewards_raw": [1, num_candidates],
       "beam_logprobs": [1, num_candidates] (可选)
     }
     ```

---

### 1.4 训练脚本：`lever_lm/workflows/grpo_post_train.py`

**数据加载部分：**

- 自动检测 JSON 是旧格式（`id_list/score_list`）还是新格式（`pointer_candidates`）：
  ```python
  if "pointer_candidates" in first_data:
      data_format = "rl"
  elif "id_list" in first_data:
      data_format = "legacy"
  ```
- 对新格式：
  - 用 `split_beam_data` 按 query 划分 train/val（默认 0.8/0.2）。
  - 从所有 `pointer` 中收集候选池索引 `candidate_indices`。
  - 从 `img_emb` 中切出 `candidate_embeddings = img_emb_data[candidate_indices]`。
  - 用 `RLBeamDatasetWithEmbedding(..., normalize_rewards=True, reward_mode=args.reward_mode, ...)` 构造数据集。

**RCE 训练：**

```python
for batch in train_loader:
    query_emb = batch["query_emb"].to(device)
    cand_emb = batch["cand_emb"].to(device)
    beam_labels = batch["beam_labels"].to(device)
    beam_rewards = batch["beam_rewards"].to(device)          # 注意：这里用的是“归一化后的 reward”
    loss = model.compute_rce_loss(query_emb, cand_emb, beam_labels, beam_rewards, temperature=τ)
```

- 验证阶段才用 `beam_rewards_raw` 调一次 `compute_rce_loss`（仅作 logging）。

**GRPO 训练：**

```python
for batch in train_loader:
    query_emb = batch["query_emb"].to(device)
    cand_emb = batch["cand_emb"].to(device)
    beam_labels = batch["beam_labels"].to(device)
    beam_rewards = batch["beam_rewards"].to(device)
    old_log_probs = batch["beam_logprobs"].to(device)

    grpo_result = model.compute_grpo_loss(
        query_emb, cand_emb, beam_labels,
        beam_rewards, old_log_probs, use_top_k=...
    )
```

在 `PointerSelectorV3.compute_grpo_loss` 内部：

- 使用 `compute_log_probs_per_beam` 重新算 `new_log_probs`。
- 调用 `self.compute_advantage(beam_rewards, normalize=True)`，做组内 Z‑score 和裁剪。
- 用 PPO + KL penalty 形式组合损失。

> ✅ 总体流程是对齐 GRPO 设计文档的。  
> ❗ 但是：**RCE 用的是“归一化后的 reward”做 softmax，GRPO 也是在 Z‑score 之后再做一次 Z‑score**，这会让绝对 reward 区分度进一步被削弱。

---

### 1.5 最新实验结果：`2025_12_10正确率.md`

- 实验配置：
  - OKVQA，Qwen2.5-VL-3B-Instruct，RandSampler。
  - v3 模型：**RCE-only baseline（5 epochs，无 GRPO）**。
  - Reward：`reward_mode=hard_plus_soft`。
- 关键结论（文档里已经写明）：
  - shot_num=2 情况下，**v3(RCE-only)** 在 100 / 200 / 400 / 800 条测试数据上都 **稳定优于 v2/v1/v0**。
  - 这是目前所有方案中 **最稳、最好的版本**。

> 这说明：  
> - 新 reward = **可靠信号**；  
> - RCE 阶段 = **好用的 offline “supervised RL”**；  
> - GRPO 阶段 = 目前主要风险点（容易把好模型拉坏）。

---

## 2. 为什么 GRPO 目前提升有限 / 甚至拉胯？

结合 `正确率.md`、`新强化学习结果.md`、`GRPO训练优化建议_v2.md` 和现在的代码，可以更精细地给一个“问题画像”。

### 2.1 Reward 信号本身：现在问题不在 reward，而在“如何用”

- 旧结论（基于早期实验）：
  - 新的 correctness reward（`hard_plus_soft`）在 shot=1,2 时 **不如** 旧的 InfoScore（见 `新强化学习结果.md`）。
- 新结论（基于 2025‑12‑10）：
  - **同样的 reward（hard_plus_soft），只在 RCE 阶段使用，效果非常好。**
  - 问题变成：**RCE → GRPO 这一步，把本来好的表示打乱了。**

核心问题：

1. **RCE 阶段已经把 pointer 排序学得很好**  
   - 模型已经接近「正确/GAP 明显」的状态。
2. GRPO 再用 PPO 形式去“放大差距”时：
   - 使用的是 **同一份离线数据**，没有新的 exploration。
   - advantage 是在 **组内 Z‑score** 后的值，方向对，但绝对大小已经被压平。
   - KL regularization 虽然存在，但 offline 设置里很容易出现：
     - 某些 query 的少数 high‑reward pointer 被过度放大；
     - 部分 noisy 样本拉着模型往错误方向走。

> 直观比喻：  
> - RCE 阶段已经把“类内排序”做得差不多了；  
> - GRPO 又在同一批样本上做大步更新，等于是**在没有新信息的情况下，继续 push policy，容易过拟合 reward 噪声**。

### 2.2 归一化链路太长，削弱了 reward 的“绝对差异”

当前链路：

1. 在 `RLBeamDatasetWithEmbedding` 里：
   - 对每个 query 内的 reward 做了一次 Z‑score（如果 `normalize_rewards=True`）。
2. 在 `compute_grpo_loss` 里：
   - 又调用 `self.compute_advantage(rewards, normalize=True)`，再次做 Z‑score。

结果：

- 真正进入 PPO 的 `advantages`，已经是「二次标准化」的结果：
  - 正样本和负样本的 **绝对间隔只剩下「高一点/低一点」**；
  - 和最初设计中「正样本 [2,3] vs 负样本 [0,1] 有明显 gap」不再一致。
- 对于 offline RL 来说，这是**放大噪声、削弱信号**的一种典型现象。

### 2.3 数据分布 & 目标不完全一致

- RL 数据是用 **shot_num=2** 的 prompt 生成的 pointer 和 correctness。
- 评估时会看 shot_num=1/2/3/4：
  - shot_num=1：接近「找最优单 ICD」的任务。
  - shot_num=3/4：要求模型在更长链上保持排序合理。
- RCE-only 已经能把「2‑shot→多 shot」迁移做得不错；  
  GRPO 在只用 2‑shot 数据训练时，很可能进一步 **overfit 在 2‑shot 上的局部结构**，反而损害对其他 shot 的泛化。

---

## 3. 短期可以马上做、且风险可控的修改（推荐优先做）

这一节给的是「**1～3 天内可以完成**」的代码级修改建议，重点是：

- 保住/强化 RCE-only baseline；
- 让后续再做 GRPO 时更可控；
- 避免再踩“隐性坑”。

---

### 3.1 把 RCE-only 明确固化成 v3 默认训练方式

**目标：**  
现在 `rce_epoch5.pt` 已经是现阶段最好的模型。建议在代码和脚本层面，都把「RCE-only」当成 v3 的**主线方案**。

**需要改动：**

1. 在 `grpo_post_train.py` 的命令行说明里，明确：
   - 推荐默认配置：`--rce_epochs 5 --grpo_epochs 0`。
   - GRPO 只作为可选后续步骤（加一句注释即可）。

2. 可以在 `GRPOTrainer.train()` 里加一个小保护：

```python
# grpo_post_train.py : GRPOTrainer.train()

def train(self):
    # 1) RCE 预热
    if self.rce_epochs > 0:
        ...

    # 2) GRPO 训练（可选）
    if self.grpo_epochs <= 0:
        print("⚠️ GRPO epochs == 0，仅进行 RCE 预热，不执行 GRPO。")
        return  # 直接结束，当前模型即为 RCE-only baseline

    for epoch in range(self.grpo_epochs):
        ...
```

**这样做的好处：**

- 避免误操作：不小心把 `grpo_epochs` 设成非 0 而覆盖了一个已经很好的 RCE 模型。
- 把「RCE-only」这个状态显式地变成流程里的一个“终点”。

---

### 3.2 SFT checkpoint 加载时增加一致性检查

目前 `generate_rl_data.py` 和 `grpo_post_train.py` 里对 checkpoint 的加载都是：

```python
model.load_state_dict(state_dict, strict=False)
```

**风险：**

- 如果 ckpt 和当前 `PointerSelectorV3` 结构不完全匹配（比如层数、名字变了）：
  - 会静默丢一些参数；
  - RL 数据或训练时其实用的是“半随机初始化”的模型。
- 这类问题很难从 log 里看出来，但会直接影响：
  - RL 数据的 pointer 分布；
  - RCE / GRPO 的效果。

**建议改动（示例伪代码）：**

```python
# generate_rl_data.py / grpo_post_train.py

missing, unexpected = model.load_state_dict(state_dict, strict=False)

if missing:
    print(f"[警告] 加载 checkpoint 时有 {len(missing)} 个参数缺失，例如：")
    for k in list(missing)[:10]:
        print("  - missing:", k)

if unexpected:
    print(f"[警告] 有 {len(unexpected)} 个多余参数，例如：")
    for k in list(unexpected)[:10]:
        print("  - unexpected:", k)

# 可选：如果缺失的关键参数太多，可以直接 raise
if len(missing) > 1000:
    raise RuntimeError("Checkpoint 与当前模型结构差异过大，请检查。")
```

**文件位置：**

- `lever_lm/models/v3/generate_rl_data.py → load_sft_model`
- `lever_lm/workflows/grpo_post_train.py → if args.sft_ckpt: ...` 模块

---

### 3.3 在 RL JSON 中显式标记 reward 质量，并允许训练阶段过滤 fallback 样本

现在 RL 数据里只有：

- `vqa_correct`, `vqa_acc_score`，但**看不出来**这个分数是：
  - 用官方 VQA metric 计算的，还是
  - fallback 字符串匹配得到的。

#### 3.3.1 修改 RL 数据生成：加上 `vqa_eval_mode`

**修改位置：** `generate_rl_data.generate_rl_data` 中，对每个 pointer 计算完 correctness 的部分。

**伪代码：**

```python
# 原来：
correct, acc_score, used_file_metric = compute_vqa_accuracy(...)

c["vqa_pred_answer"] = pred_answer
c["vqa_correct"] = correct
c["vqa_acc_score"] = acc_score

# 建议修改为：
c["vqa_pred_answer"] = pred_answer
c["vqa_correct"] = correct
c["vqa_acc_score"] = acc_score
c["vqa_eval_mode"] = "file" if used_file_metric else "fallback"
```

同时，在统计信息里也打印一下：

```python
print(f"  - 使用文件方式计算: {file_metric_count} ({file_metric_ratio:.1f}%)")
print(f"  - 使用 fallback 字符串匹配: {fallback_count} ({fallback_ratio:.1f}%)")
```

#### 3.3.2 在 Dataset 中允许跳过 fallback 样本

**修改位置：** `RLBeamDatasetWithEmbedding.__init__`（`lever_lm/models/v3/dataset_v3.py`）。

增加一个参数：

```python
def __init__(
    ...,
    skip_fallback_reward: bool = False,
):
    self.skip_fallback_reward = skip_fallback_reward
```

在遍历 `pointer_candidates` 时加入过滤逻辑：

```python
for c in pointer_candidates:
    # 如果要求跳过 fallback 样本
    if self.skip_fallback_reward and c.get("vqa_eval_mode") == "fallback":
        continue

    pointer = c["pointer"]
    ...

    reward = compute_reward_for_candidate(...)
    beam_rewards.append(reward)
```

如果过滤完之后一个 query 没有任何候选，可以直接 `continue` 掉这个 query。

#### 3.3.3 在 GRPO 训练脚本里暴露开关

**修改位置：** `grpo_post_train.py` 的 argparse 和 dataset 构造。

1. 增加参数：

```python
parser.add_argument(
    "--skip_fallback_reward",
    action="store_true",
    help="是否跳过使用 fallback 方式计算的 RL 样本（推荐在 OKVQA 上打开）"
)
```

2. 构造 dataset 时传入：

```python
train_dataset = RLBeamDatasetWithEmbedding(
    ...,
    skip_fallback_reward=args.skip_fallback_reward,
)
val_dataset = RLBeamDatasetWithEmbedding(
    ...,
    skip_fallback_reward=args.skip_fallback_reward,
)
```

**这样做的好处：**

- 让 RL 训练只信任「来源更干净」的 reward（官方 VQA metric）。
- fallback 样本仍然保留在 JSON 里，如果未来要做对比实验也方便。

---

### 3.4 RCE 阶段优先使用原始 reward（保留当前实现作为可选）

现在的实现是：

- `RLBeamDatasetWithEmbedding` 里已经把 reward 做了一次 Z‑score，并把结果放在 `beam_rewards`。
- `train_rce_epoch` 用的是 `beam_rewards`，即「已标准化的 reward」。

RCE 理论上是：

- `w_i = softmax(score_i / τ)`，这里的 `score_i` 可以是原始 reward，也可以是线性变换；
- Z‑score 在数学上不会改变排序，但会改变 **温度的实际效果**。

**建议做法：**

1. 在 `GRPOTrainer` 里增加一个配置开关，例如 `self.rce_use_raw_reward`（或 CLI 参数 `--rce_use_raw_reward`）。

2. 在 `train_rce_epoch` 中根据开关决定使用哪个：

```python
for batch in self.train_loader:
    ...
    if self.rce_use_raw_reward:
        reward_for_rce = batch["beam_rewards_raw"].to(self.device)
    else:
        reward_for_rce = batch["beam_rewards"].to(self.device)

    loss = self.model.compute_rce_loss(
        query_emb, cand_emb, beam_labels,
        reward_for_rce,
        temperature=current_temp,
    )
```

**为什么短期建议只是“增加开关”，而不是立刻改默认行为？**

- 你现在的 RCE-only baseline（hard_plus_soft + 使用归一化 reward）已经很强了；
- 直接改默认行为可能会把现有最优结果打乱；
- 更稳妥的做法是：
  - 保留当前 pipeline；
  - 新开一组实验，对比：
    - `rce_use_raw_reward = False`（当前行为） vs `True`。

---

### 3.5 GRPO 阶段：先减小“破坏力”，再谈收益

鉴于目前 GRPO 的作用更像是「把好模型拉坏」，短期建议：

1. **把 GRPO 当成“小步微调”，而不是大规模再训练**

在 `GRPOTrainer` 的默认超参里建议改成：

- `grpo_epochs`：先从 **1~3** 做对照实验，而不是一上来 8 或 25。
- `grpo_lr`：比 RCE 再小一档，例如：
  - RCE：`1e-4`；
  - GRPO：`3e-5` 或 `1e-5`。

2. **限制更新的参数范围**

在 `PointerSelectorV3` 之外，可以加一个简单的“冻结策略”：

- 在 `grpo_post_train.py` 里：

```python
if args.freeze_backbone_in_grpo:
    for name, param in model.named_parameters():
        # 例如：只训练最后一层 MLP 或 pointer head
        if "pointer_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
```

- CLI 中加参数：

```python
parser.add_argument(
    "--freeze_backbone_in_grpo",
    action="store_true",
    help="GRPO 时只训练 pointer head，冻结 backbone"
)
```

**这样做的目的：**

- 让 GRPO 只能在「已经学好的 RCE 表示」上做一点点排序微调；
- 避免它把底层的 query/candidate 表示整个打乱。

---

## 4. 中期可以规划的改动（1～2 周）

这一部分不需要立刻改代码，但可以作为后续 Roadmap。

### 4.1 针对不同 shot_num 单独生成 RL 数据 & 训练

目前 RL 数据和 pointer 模型都是针对 **shot_num=2** 设计的：

- RL 数据：所有 pointer 都是长度 2 的 `[icd1, icd2]`。
- 训练：`shot_num=2`，GRPO / RCE 都是在 2‑shot 场景下优化。

**在 shot_num ≥ 3 时性能不稳定，很大一部分原因其实是：**

- 模型从来没有在「3/4‑shot 的 pointer 序列」上被 reward 过；
- 只能靠“2‑shot 经验”泛化。

**中期可以考虑：**

1. 在 `generate_rl_data.py` 中增加可选参数：
   - `--shot_num 2/3/4`；
   - 或者直接生成多份 RL 数据：2‑shot / 3‑shot / 4‑shot。

2. 在 `rl_data_generation.py` 中：
   - 针对不同长度的 pointer（`shot_num` 不同），复用同一个生成逻辑；
   - 对每条 pointer 再跑一次 VQA correctness。

3. 在训练脚本中：
   - 支持加载多份 RL JSON（2‑shot + 3‑shot + 4‑shot）；
   - 通过一个简单的 multi-task 方式训练：
     ```python
     # 伪代码
     for batch in multi_shot_dataloader:
         # batch 里包含不同 shot_num 的样本
         loss = rce_loss(batch_2shot) + rce_loss(batch_3shot) + rce_loss(batch_4shot)
         loss.backward()
     ```

> 目标：让 pointer 模型对「长一点的 few-shot」也有直接 reward 监督，而不是全靠 2‑shot 泛化。

---

### 4.2 更系统地探索 reward_mode（而不是只靠一次试验）

你现在的代码已经支持了多种 reward 模式：

- `hard_plus_soft`, `hard_plus_soft_v2`, `separated`, `hard_only`, `soft_only`, `hybrid`, `legacy`。

中期可以设计一个 **系统的网格实验**，而不是一次性的试验：

- 维度 1：`reward_mode ∈ {hard_plus_soft, hard_plus_soft_v2, separated, hybrid}`。
- 维度 2：`hard_weight / soft_weight` 比例，比如：
  - (1.0, 1.0), (1.0, 0.5), (0.5, 1.0)。
- 维度 3：是否在 RCE 中使用 raw reward。

> 重点是：**先在 RCE-only 场景下做完一轮对比，确定最优 reward_mode，再考虑是否在 GRPO 中沿用或调整。**

---

### 4.3 Hybrid：让 InfoScore 回来当“辅助信号”，而不是主角

从 `新强化学习结果.md` 看：

- 旧的 InfoScore 方案在 shot=1,2 时其实挺强；
- 新的 correctness 方案在多 shot 时更稳。

`reward_utils.py` 里已经有了 `"hybrid"` 模式，可以考虑：

- 在 RL 数据中保留 beam_score（InfoScore）；
- 在 `compute_reward_for_candidate` 里用类似：

```python
elif reward_mode == "hybrid":
    # InfoScore 部分（归一化到 [0, 1]）
    info = float(beam_score) if beam_score is not None else 0.0
    info_norm = (info - info_min) / (info_max - info_min + 1e-8)

    # Correctness 部分
    hard = float(vqa_correct) if vqa_correct is not None else 0.0
    soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0

    reward = alpha * info_norm + (1 - alpha) * (hard + soft)
```

中期可以专门做一组「hybrid vs pure correctness vs pure InfoScore」的对比。

---

## 5. 总结 & 下一步执行顺序建议

结合你现在仓库里的代码和最新的实验，可以把优先级排成这样：

1. **马上做（短期）**
   1. 把 `RCE-only (rce_epoch5.pt)` 明确标记为当前 v3 默认最佳模型。
   2. 在 `generate_rl_data.py` 和 `grpo_post_train.py` 中加入 checkpoint 加载的 missing/unexpected 参数打印。
   3. 在 RL JSON 中加入 `vqa_eval_mode` 字段，并在 `RLBeamDatasetWithEmbedding` 支持 `skip_fallback_reward` 开关，训练脚本暴露 `--skip_fallback_reward`。
   4. 给 RCE 训练增加 `--rce_use_raw_reward` 这种轻量开关，方便实验“不归一化 reward”的效果。
   5. 在 GRPO 阶段降低学习率 & 训练轮数，并增加 `--freeze_backbone_in_grpo`，把它当成“小步微调”而不是大手术。

2. **一两周内规划（中期）**
   1. 设计多 shot（2/3/4）对应的 RL 数据和联合训练流程。
   2. 做一轮系统性的 `reward_mode × (hard_weight, soft_weight) × rce_use_raw_reward` 网格实验，先在 RCE-only 场景下确定最佳配置。
   3. 再根据最佳 reward 配置，尝试低强度 GRPO 微调（小 lr、少 epochs、冻结 backbone）。

---

你可以把这份方案当成 **“在当前仓库上的增量 todo 列表”** 来执行：  
每个小项都有清晰的文件位置和伪代码，不会影响你现在已经得到的最优 RCE-only 结果，也为后续安全地继续研究 GRPO 留出了空间。
