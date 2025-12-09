# Lever-Plus v3 强化学习实现评估与改进计划

> 基于当前 GitHub 仓库代码（`yunsoft2019/Lever-Plus`）的深入阅读与多轮讨论整理。  
> 重点覆盖：当前实现是否符合设计、为何效果有限、以及下一步的具体改进方案。

---

## 1. 总体结论

1. **实现层面：当前 v3 强化学习管线整体是「对的」**  
   - RL 数据生成脚本 `generate_rl_data.py`：  
     确实是用 v2/v3 pointer 模型重新做了 **beam + 温度采样 + 随机组合**，对每条 pointer 用 VQA 模型生成答案并计算 `vqa_correct` 与 `vqa_acc_score`，**不再依赖 InfoScore 作为主 reward**。
   - `reward_utils.compute_reward_for_candidate`：  
     默认 `reward_mode="hard_plus_soft"`，`hard_weight=soft_weight=1.0`，`alpha=beta=0.0`，因此 reward = `vqa_correct + vqa_acc_score`，范围 [0, 2]。  
   - `RLBeamDatasetWithEmbedding`：  
     从 RL JSON 中读取 `pointer_candidates`，通过 `compute_reward_for_candidate` 计算每个 pointer 的 reward，组内做 Z-score 归一化后喂给 GRPO，同时保留原始 reward 供 RCE 使用。

2. **效果层面：性能提升有限的主要原因是结构性问题，而不是代码 bug**  
   - 训练目标只包含 **2‑shot correctness**：RL 只优化“用两条 ICD 时答题表现”，但评估却看 1/2/3/4‑shot，因此 **1-shot、3/4-shot 本身没有被写进目标函数**。
   - `hard+soft` correctness 作为 reward，**信号较弱且噪声较大**：
     - 很多 query 上所有 pointer 都错，`hard=0`，`soft` 大多是 0 / 0.5，组内差异小；
     - 如果 RL 数据生成阶段没有使用官方 VQA metric，而是 fallback 到字符串匹配，噪声会进一步放大。
   - RL 数据量和覆盖有限（典型是 ~800 条 RandSampler query）：  
     offline RL 在这样窄的数据分布上做 reweight，上限本来就不高。

3. **实现上存在少量需要修补的小坑**：  
   - pointer index → candidate embedding index 映射时对缺失索引使用了 fallback；
   - 生成 RL 数据时如果没有提供 `val_ques_path` / `val_ann_path`，会退回到简单字符串匹配，reward 噪声较大。

接下来按模块详细说明，并列出具体改动建议与实验路线。

---

## 2. 当前实现拆解与检查

### 2.1 RL 数据生成：`lever_lm/models/v3/generate_rl_data.py`

#### 2.1.1 pointer 候选生成

核心函数：

```python
pointer_candidates = generate_pointer_candidates_for_query(
    model=sft_model,
    query_emb=query_emb,
    cand_emb=cand_emb,
    num_beams=num_beams,
    temps=temps,
    num_samples_per_temp=num_samples_per_temp,
    num_random=num_random,
    beam_search_fn=None  # 如已有 beam_search，可传入替换
)
```

特点：

- 使用当前加载的 SFT pointer 模型（通常是 v2），基于 `query_emb` 和 `cand_emb` **重新做 beam + 温度采样 + 随机组合**：  
  - beam：几条得分最高的 pointer；  
  - sampling：给定温度 `tau` 做按分布采样；  
  - random：从候选池中随机采样 pointer。
- **不再依赖旧的 InfoScore beam JSON 的排序**，而是以当前 pointer 模型的行为为准，这符合“必须真的用 v2/v3 采样来构造 RL 数据”的要求。

#### 2.1.2 correctness 计算

对每个 pointer：

```python
pred_answer = build_vqa_prompt_and_generate(
    interface=vqa_model,
    image=image,
    question=question,
    ex1=ex1,
    ex2=ex2,
    generation_kwargs=generation_kwargs or {}
)

question_id_str = query_item.get("question_id", str(query_id))

correct, acc_score = compute_vqa_accuracy(
    pred_answer=pred_answer,
    ground_truth_answers=gt_answers,
    question_id=question_id_str,
    val_ques_path=val_ques_path,
    val_ann_path=val_ann_path
)

c["vqa_pred_answer"] = pred_answer
c["vqa_correct"] = correct
c["vqa_acc_score"] = acc_score
```

- `build_vqa_prompt_and_generate` 仿照 `icl_inference.py`：
  - 构造 `[ex1, ex2, query]` 的 few-shot 列表；
  - 调用 interface 的 `transfer_prompts` / `prepare_input` / `generate`；
  - 根据模型类型（Qwen / Flamingo 等）做适配与 `vqa_postprocess`。

- `compute_vqa_accuracy`：
  - **优先**：如果提供了 `val_ques_path` + `val_ann_path` + `question_id`，使用 `open_mmicl.metrics.vqa_metrics.compute_vqa_accuracy` 做“官方 VQA 打分”；
  - **否则 fallback**：使用简单字符串匹配：
    - 完全匹配 → `(correct=1, acc_score=1.0)`；
    - 部分包含 → `(correct=1, acc_score=0.5)`；
    - 否则 → `(0, 0.0)`。

⚠ **需要注意的点**：

1. 如果生成 RL 数据时没有传 `--val_ques_path` / `--val_ann_path`，reward 全是基于字符串匹配的“粗糙 acc”，噪声会比较大；  
2. 即便传了文件，也要确保 `question_id` 与标注文件中的 id 对得上，否则仍会 fallback 到字符串匹配。  
   - 建议在生成 RL 数据时显式检查：
     - 如果 `compute_vqa_accuracy_metric` 失败的比例很高，需要排查 `question_id` 映射问题。

---

### 2.2 奖励定义：`lever_lm/utils/reward_utils.py`

核心函数：

```python
def compute_reward_for_candidate(
    beam_score: Optional[float] = None,
    logprob_score: Optional[float] = None,
    vqa_correct: Optional[int] = None,
    vqa_acc_score: Optional[float] = None,
    reward_mode: str = "hard_plus_soft",
    hard_weight: float = 1.0,
    soft_weight: float = 1.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    correctness_mode: str = "01",
    use_logprob: bool = False,
    reward_clip: Tuple[float, float] = (-5.0, 5.0)
) -> float:
    ...
```

在默认设置下：

```python
if reward_mode == "hard_plus_soft":
    hard = float(vqa_correct) if vqa_correct is not None else 0.0
    soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
    reward = hard_weight * hard + soft_weight * soft
```

- 默认：`reward_mode="hard_plus_soft"`, `hard_weight=1.0`, `soft_weight=1.0`；
- `alpha=beta=0.0`，因此 `legacy` 分支不会被用到；
- `reward_clip=(-5, 5)`，而 `hard+soft ∈ [0,2]`，不会被裁剪。

**因此当前 reward = vqa_correct + vqa_acc_score**：

- 正样本：范围 `[1,2]`；
- 负样本：范围 `[0,1)`。

其他模式：

- `"hard_only"`：只用 `vqa_correct`；  
- `"soft_only"`：只用 `vqa_acc_score`；  
- `"legacy"`：保留 InfoScore/quality + correctness 的线性组合（目前建议不要再用）。

辅助函数：

- `normalize_rewards_zscore` / `clip_advantages` / `compute_group_relative_advantage` 用于 GRPO 阶段对 reward 做组内归一化与裁剪；  
- `compute_softmax_weights` / `compute_temperature_schedule` 用于 RCE 阶段构造 sample weights。

---

### 2.3 RL 数据集：`lever_lm/models/v3/dataset_v3.py` 中的 `RLBeamDatasetWithEmbedding`

数据结构：

```python
rl_data = {
  "<query_id>": {
    "pointer_candidates": [
      {
        "pointer": [i, j],
        "beam_score": ...,
        "logprob_score": ...,
        "gen_method": "beam" | "sample" | "random",
        "vqa_pred_answer": "...",
        "vqa_correct": 0 or 1,
        "vqa_acc_score": 0.0 ~ 1.0,
      },
      ...
    ]
  },
  ...
}
```

构造逻辑简要：

```python
self.cand_idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}

for query_id_str, query_data in rl_data.items():
    query_id = int(query_id_str)
    pointer_candidates = query_data.get("pointer_candidates", [])

    # 可选：按 gen_method 过滤
    if self.filter_gen_methods is not None:
        pointer_candidates = [
            c for c in pointer_candidates
            if c.get("gen_method") in self.filter_gen_methods
        ]

    beam_labels = []
    beam_rewards = []
    beam_logprobs = []

    for c in pointer_candidates:
        pointer = c["pointer"]
        # 映射 pointer → candidate_embeddings 行号
        mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]
        beam_labels.append(mapped_pointer)

        reward = compute_reward_for_candidate(
            beam_score=c.get("beam_score"),
            logprob_score=c.get("logprob_score"),
            vqa_correct=c.get("vqa_correct"),
            vqa_acc_score=c.get("vqa_acc_score"),
            reward_mode=self.reward_mode,
            hard_weight=self.hard_weight,
            soft_weight=self.soft_weight,
            alpha=self.reward_alpha,
            beta=self.reward_beta,
            correctness_mode=self.reward_correctness_mode,
            use_logprob=self.use_logprob
        )
        beam_rewards.append(reward)
        beam_logprobs.append(c.get("logprob_score"))
```

在 `__getitem__` 中：

```python
query_emb = self.query_embeddings[query_id]      # [d]
cand_emb = self.candidate_embeddings            # [K, d]

beam_labels_tensor = torch.tensor(beam_labels, dtype=torch.long)
beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)

# 组内 Z-score
beam_rewards_normalized = beam_rewards_raw.clone()
mean = beam_rewards_raw.mean()
std = beam_rewards_raw.std()
if std > 1e-12:
    beam_rewards_normalized = (beam_rewards_raw - mean) / std

result = {
    "query_id": query_id,
    "query_emb": query_emb,          # [d]
    "cand_emb": cand_emb,            # [K, d]
    "beam_labels": beam_labels_tensor,
    "beam_rewards": beam_rewards_normalized if self.normalize_rewards else beam_rewards_raw,
    "beam_rewards_raw": beam_rewards_raw,
}
if beam_logprobs and all(lp is not None for lp in beam_logprobs):
    result["beam_logprobs"] = torch.tensor(beam_logprobs, dtype=torch.float32)
```

**与训练脚本的对应关系**（在 `GRPOTrainer` 里）：

- RCE 阶段：
  - 用 `beam_rewards_raw` 作为原始 reward，经过 softmax(·/τ) 构造 sample 权重；
- GRPO 阶段：
  - 用 `beam_rewards`（即组内 Z-score）作为 group-relative reward；
  - 再经过 `compute_group_relative_advantage` / KL 约束等得到 loss。

#### 2.3.1 这里的一个小坑：索引 fallback

```python
mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]
```

- 正常情况下：`pointer` 中的每个 idx 都是 `candidate_indices` 的一个元素，对应于 `candidate_embeddings` 的一行；
- **问题**：一旦某个 idx _不在_ `candidate_indices` 里，就会 fallback 到自身 `idx`：
  - 这会悄悄把 pointer 映射到错误的 embedding 行，但不会抛错，很难发现。

**建议改成更严格的写法：**

```python
mapped_pointer = []
for idx in pointer:
    if idx not in self.cand_idx_to_pos:
        raise KeyError(f"Pointer index {idx} not in candidate_indices (query_id={query_id})")
    mapped_pointer.append(self.cand_idx_to_pos[idx])
```

这样如果 RL JSON 与 embedding 对不齐，会立即 crash，便于调试。

---

### 2.4 训练流程（RCE + GRPO）逻辑回顾

虽然你没贴 `grpo_post_train.py`，但根据之前的讨论和数据结构可以确定：

1. **RCE 阶段**（Reward-Weighted CE 预热）：
   - 从 dataloader 取：`query_emb`, `cand_emb`, `beam_labels`, `beam_rewards_raw`；
   - pointer 模型前向得到每条候选 pointer 的 logit / log_prob；
   - 使用 `softmax(beam_rewards_raw / τ)` 作为 sample 权重，对“选择正确 pointer 序列”的 CE 做加权；
   - 目的是：先往“高 reward 的 pointer”方向收敛，稳定分布。

2. **GRPO 阶段**（Group-Relative PPO 风格更新）：
   - 从 dataloader 取：`beam_rewards`（组内 Z-score）、`beam_labels`；
   - 查 `old_log_probs`（SFT/v2 pointer 产生该 pointer 的 logprob）；
   - 计算：
     - `advantage = group_relative_advantage(beam_rewards)`；
     - PPO 风格的 clip ratio；
     - KL penalty（`compute_kl_penalty` + `adaptive_kl_beta`）；
   - 更新策略，使其在每个 query 的候选 pointer group 内朝着高 reward 的 pointer 倾斜。

整体逻辑与我们之前设计的“RCE + GRPO + group-relative reward + KL 自适应”是一致的。

---

## 3. 为什么性能提升有限？

结合代码与已有实验结果，可以归纳出以下几个核心原因。

### 3.1 训练目标只覆盖了「2‑shot」，而评估看 1/2/3/4‑shot

- 当前 RL 数据 **固定 `shot_num=2`**：
  - pointer 永远是 `[i, j]`；
  - correctness / reward 也是“用这两个 ICD 做 VQA 的得分”。
- 于是 RL 实际在解的问题是：

> 「在必须使用 2 条 ICD 的前提下，选哪两条能让 VQA 得分最高？」

而你评估时：

- 1‑shot：只用 pointer 的第一条 ICD；  
- 2‑shot：用前两条 ICD；  
- 3/4‑shot：通常是对 pointer 分数排序后取前 3/4 条 ICD。

**后果：**

1. **1‑shot 没被写进目标函数**  
   - 模型完全可以选“作为 pair 很好，但第一条单独用一般”的 ICD → 1‑shot 表现变差是预期行为。

2. **3/4‑shot 完全是 2‑shot 策略的副产品**  
   - pointer 模型没被训练去考虑第三、第四条 ICD 的作用；  
   - 在这种情况下，3/4‑shot 的改善只能是“从更好的 2‑shot 策略迁移来的顺带提升”，很难指望大幅度超越 v2。

这解释了：为什么旧的 InfoScore 版 v3 在 shot≤2 明显优于 v2，而在 shot≥3 时不稳定甚至略差；新方案也很难在 3/4‑shot 上有显著提升。

---

### 3.2 correctness reward 信号偏弱且噪声较大

当前 reward：

```text
hard = vqa_correct ∈ {0,1}
soft = vqa_acc_score ∈ [0,1]
reward = hard + soft ∈ [0,2]
```

在很多 query 上会出现以下情况：

- 所有 pointer 都是错的 → `hard = 0`；
- `soft` 大多是 0 / 0.5（部分匹配），十几个候选之间差异极小；
- 如果使用 fallback 字符串匹配，`0.5` 的判定本身就很脆弱（大小写、标点、数字表达等都可能导致误判）。

然后：

- 在 `RLBeamDatasetWithEmbedding` 中对同一 query 的 `beam_rewards_raw` 做 Z‑score；  
- 在 GRPO 中又会对 rewards / advantages 做一次 group 拉平与裁剪。

**结果**：

- **组内 variance 很小，advantage 数值本身也很小**，梯度信号弱；  
- 少量被错误高估的 pointer 可能因为噪声被当成“好样本”，RL 反而往错误方向调整。

对比旧 InfoScore：

- 尽管理论上有问题（只看增益、且在一些模型上全是负数），但在许多 query 上，InfoScore 数值差异较大：  
  - 一条 pointer 的增益明显高于其他；  
  - group 内的排序信息更清晰 → 有利于 GRPO 更新。

这也是为什么你看到：**旧 v3_1layer 在 2‑shot 上比 v2 稳定优秀，而新 hard+soft correctness 方案在 2‑shot 上反而很难稳定领先**。

---

### 3.3 RL 数据量与覆盖有限

- 目前 RL 数据通常是在 OKVQA 中抽取 ~800 条 RandSampler query 来生成：
  - 每个 query 8~12 个 pointer（beam + sample + random）；
- 相比 v2 的 SFT 训练集，这只是一个很小的子集：
  - 本身覆盖有限；
  - pointer 候选都聚焦在 v2 的分布附近。

在这样的 setting 下：

- offline RL 的上限被强约束：只能在小范围内做 reweight；  
- 遇到 reward 噪声或某些 query 本身极难时，很容易出现“微小的 overfit + 泛化收益被抵消”的情况；  
- 在你用更大规模的评估集（200 / 400 条）做对比时，这种效应会进一步暴露出来。

---

### 3.4 实现细节上的小风险叠加

- `self.cand_idx_to_pos.get(idx, idx)` 的 fallback 可能导致 pointer embedding 错位，虽然概率不一定高；  
- 生成 RL 数据时如果没传 VQA 标注文件，reward 全是字符串匹配打分，噪声显著；  
- 多种 gen_method（beam / sample / random）混在一起，质量差异较大，如果没有做适当的筛选，也会放大噪声。

这些细节单独看都不是“致命 bug”，但叠加在一起，容易进一步削弱 RL 的有效性。

---

## 4. 推荐改动与优化方案（基于当前代码）

以下改动都可以在现有实现的基础上 **增量修改**，不需要推倒重来。

### 4.1 严格 pointer → candidate embedding 的索引映射

**问题回顾**：

```python
mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]
```

- 一旦 `idx` 不在 `candidate_indices` 里，就会 fallback 到自身 `idx`，默默选错 embedding 行。

**建议修改**：

```python
# 原代码片段（有隐式 fallback 风险）
mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]
beam_labels.append(mapped_pointer)
```

改成：

```python
mapped_pointer = []
for idx in pointer:
    if idx not in self.cand_idx_to_pos:
        raise KeyError(
            f"[RLBeamDatasetWithEmbedding] Pointer index {idx} not in candidate_indices "
            f"(query_id={query_id})"
        )
    mapped_pointer.append(self.cand_idx_to_pos[idx])

beam_labels.append(mapped_pointer)
```

**收益：**

- 一旦 RL JSON 与 candidate_embeddings 对不齐，当场报错 → 能快速修正数据管线问题；  
- 避免在“embedding 错位”的情况下训练，排除一个潜在的隐形干扰源。

---

### 4.2 确保 reward 真正等于「hard + soft correctness」

当前实现已经具备这个能力，但需要注意以下几点：

1. **构造 `RLBeamDatasetWithEmbedding` 时保持默认参数**：

   ```python
   ds = RLBeamDatasetWithEmbedding(
       rl_data=rl_data,
       query_embeddings=query_embeddings,
       candidate_embeddings=candidate_embeddings,
       candidate_indices=candidate_indices,
       shot_num=2,
       normalize_rewards=True,
       reward_mode="hard_plus_soft",   # 默认
       hard_weight=1.0,
       soft_weight=1.0,
       reward_alpha=0.0,               # 保证 legacy 分支不会生效
       reward_beta=0.0,
       reward_correctness_mode="01",
       use_logprob=False,
   )
   ```

2. **不要再把 beam_score / logprob_score 混入 reward**（除非刻意做 legacy 对照实验）；  
3. 可以写一个小检查脚本，从 dataset 里取一个样本，验证：

   ```python
   # 伪代码
   sample = ds[0]
   print(sample["beam_rewards_raw"])  # 应该大致在 [0,2] 之间
   ```

   同时对照对应 pointer 的 `vqa_correct` / `vqa_acc_score`，确认：

   ```text
   beam_rewards_raw[k] ≈ vqa_correct[k] + vqa_acc_score[k]
   ```

---

### 4.3 生成 RL 数据时优先使用官方 VQA metric

为了减少 reward 噪声，建议：

1. **调用 `generate_rl_data.py` 时显式传入**：

   ```bash
   python -m lever_lm.models.v3.generate_rl_data \
       --sft_ckpt path/to/v2_pointer.ckpt \
       --beam_data path/to/beam_data.json \
       --output_path path/to/rl_data.json \
       --query_emb path/to/query_embeddings.pt \
       --cand_emb path/to/candidate_embeddings.pt \
       --vqa_model qwen2.5_vl_3B \
       --dataset okvqa_local \
       --val_ques_path path/to/vqav2_mscoco_val2014.json \
       --val_ann_path path/to/vqav2_mscoco_val2014_annotations.json \
       ...
   ```

2. 在 `compute_vqa_accuracy` 里可以做一些轻量 logging：
   - 统计多少比例的 query 是通过文件方式打分，多少是 fallback；  
   - 如果 fallback 比例过高，优先修 question_id 映射。

3. **在这一阶段先不追求非常大的 RL 数据量**：
   - 把一部分 query 做到“打分准确 + reward 可靠”，往往比在大规模 noisy reward 上训练要更值得。

---

### 4.4 训练策略上的调整：先稳住 2‑shot，再谈多 shot

在 reward 与数据确认没问题的前提下，可以优先做一些低成本的训练策略实验。

#### 4.4.1 RCE-only baseline

目标：**先看纯 RCE（没有 GRPO）的收益上限**。

- 在 `grpo_post_train` 的配置里将 GRPO 关掉或极弱化：

  - `RCE_EPOCHS`：保持 3~5（或略多）；  
  - `GRPO_EPOCHS`：设为 0（或 1 作为 sanity check）；

- 效果判断：
  - 2‑shot 上是否能做到至少与 v2 持平，甚至略有提升；  
  - 1/3/4‑shot 是否不至于明显崩坏。

如果 RCE-only 就能达到一个“不错且稳定”的水平，那么 GRPO 的 PPO 部分可能需要大幅弱化（因为在小数据 + noisy reward 的 offline setting 下，强 PPO 很容易把策略拉离 SFT 再过拟合）。

#### 4.4.2 减弱 GRPO 的力度 + 加强 KL 约束

在确认 RCE 结果的基础上，如果仍希望引入少量 policy gradient，可以尝试：

- 降低 GRPO learning rate（例如减半或 1/3）；  
- 减少 GRPO epochs（例如 1~2 个 epoch）；  
- 在 `compute_kl_penalty` / `adaptive_kl_beta` 相关超参里：
  - 适当提高初始 `kl_beta`；  
  - 收紧 KL target 区间 (`kl_target_min`, `kl_target_max`)，让策略不要偏离 v2 太远。

目标：**把 GRPO 当作一个“轻柔的微调”，而不是强力重塑策略**。

---

### 4.5 利用 `filter_gen_methods` 做 beam / sample / random 消融

当前 RL 数据里混入了多种来源：

- beam：通常质量较高；  
- sample（不同温度）：质量参差不齐；  
- random pointer：质量最差，但提供探索。

你已经在 `RLBeamDatasetWithEmbedding` 里预留了：

```python
filter_gen_methods: Optional[List[str]] = None

if self.filter_gen_methods is not None:
    pointer_candidates = [
        c for c in pointer_candidates
        if c.get("gen_method") in self.filter_gen_methods
    ]
```

建议用它做几组消融实验：

1. **只用 beam**：

   ```python
   ds_beam_only = RLBeamDatasetWithEmbedding(
       rl_data=rl_data,
       ...,
       filter_gen_methods=["beam"],
   )
   ```

   - 看看在只保留“高质量候选”的情况下，RL 能否在 2‑shot 上稳定提升；

2. **beam + sample**：

   ```python
   filter_gen_methods=["beam", "sample"]
   ```

   - 允许一定程度的探索，但避免 random 极端噪声；

3. **beam + sample + random**（当前配置的对照组）：

   - 比较三种设置下的 2/3/4‑shot 表现，分析噪声来源。

这样可以回答一个关键问题：

> “是 reward 设计本身就不够区分，还是 random / 高温采样带来的劣质 pointer 把 RL 搅乱了？”

---

### 4.6 多 shot RL：中长期可以考虑的方向

如果未来你非常在意 3/4‑shot 的表现，可以考虑真正把多 shot 写进训练目标。简述一个可行方向：

1. 固定 pointer 输出 4 条 ICD：`[i1, i2, i3, i4]`；  
2. 对同一条 pointer 序列，分别算：
   - `R1`：只用 `i1` 做 1‑shot VQA；  
   - `R2`：用 `i1, i2` 做 2‑shot；  
   - `R3`：用 `i1, i2, i3` 做 3‑shot；  
   - `R4`：用 `i1, i2, i3, i4` 做 4‑shot；
3. 定义综合 reward：

   ```text
   R_total = w1 * R1 + w2 * R2 + w3 * R3 + w4 * R4
   ```

   - 如果你更看重 2‑shot：可以令 `w2` 最大，`w1,w3,w4` 稍小；  
   - 这样 1/2/3/4‑shot 都会出现在目标函数里。

这条路线实现和算力成本都比较高（每个 pointer 要跑 4 次 VQA），更适合作为中长期优化方向。短期还是建议先把 **2‑shot RL 调到“至少不输 v2，部分 shot 稍有提升”**。

---

## 5. 一个可执行的实验路线（建议）

这里给出一个你可以直接照着走的 checklist：

### Step 1：修代码小坑

- [ ] 修改 `RLBeamDatasetWithEmbedding` 的 pointer 映射，去掉 `.get(idx, idx)` fallback，改为显式 `KeyError`；  
- [ ] 确认训练脚本中 `reward_mode="hard_plus_soft"`，`reward_alpha=reward_beta=0`，不再额外混入 beam_score/logprob。

### Step 2：重新生成一版「干净」的 RL 数据

- [ ] 调 `generate_rl_data.py` 时传入正确的 `--val_ques_path` / `--val_ann_path`；  
- [ ] 检查日志，确认大部分 query 都是用官方 metric 打分的；  
- [ ] 保存 RL JSON（如 `rl_data_qwen2.5_vl_3B_hard_soft.json`）。

### Step 3：跑一个 RCE-only baseline

- [ ] 用新的 RL JSON + 新的 `RLBeamDatasetWithEmbedding` 训练 pointer：
  - RCE_EPOCHS：例如 5；  
  - GRPO_EPOCHS：0；  
- [ ] 在 100 / 200 / 400 / full dev set 上评估：
  - 2‑shot：是否 ≥ v2；  
  - 1/3/4‑shot：是否不显著退化。

### Step 4：在 RCE-only 基础上加入轻量 GRPO

- [ ] 在保持 RCE 配置不变的前提下，增加 1~2 个 epoch 的 GRPO：
  - GRPO_LR 较小；  
  - KL 约束稍强；  
- [ ] 再次评估 1/2/3/4‑shot，比较相对 RCE-only 的变化。

### Step 5：做 gen_method 消融

- [ ] 使用 `filter_gen_methods=["beam"]` 训练一版，记录 1/2/3/4‑shot；  
- [ ] 使用 `["beam", "sample"]` 再训练一版；  
- [ ] 使用全量（beam+sample+random）做对照；  
- [ ] 分析不同设置下 RL 效果的差异，确认噪声主要来源。

### Step 6（可选）：可视化 reward 分布

- [ ] 随机抽几个 query，画出不同 pointer 的：
  - `vqa_correct` 分布；  
  - `vqa_acc_score` 分布；  
  - `reward = hard+soft` 分布；  
- [ ] 观察：
  - 有多少 query 所有 pointer 都错；  
  - group 内 reward 差异有多大；  
  - 正/负样本比例。

这些数据会帮助你判断：

> “是在 reward 设计本身就难以区分 pointer 质量，还是在已有的 RL 数据上，其实已经没有太多提升空间了？”

---

## 6. 总结

1. **实现上，你已经基本完成了从 InfoScore 增益 → 整条 pointer correctness（hard+soft）的迁移**；  
2. 再往前走，瓶颈已经不再是“代码写错”，而是：
   - 训练目标只优化 2‑shot，1/3/4‑shot 未被显式纳入；  
   - correctness reward 在许多 query 上信号弱、噪声大；  
   - RL 数据量与覆盖有限，offline RL 上限不高。

3. **短期优先级建议：**
   - 修掉 pointer 映射与 VQA metric 的小坑；  
   - 重跑一版“干净”的 RL 数据；  
   - 先在 RCE-only & 轻量 GRPO 的框架下，把 2‑shot 表现稳定做到不输 v2，并尽量减少 1/3/4‑shot 的退化。

4. **中长期可以考虑的方向：**
   - 多 shot RL（R1+R2+R3+R4 的综合 reward）；  
   - 更精细的 reward 设计（例如对 soft score 做非线性拉伸、区分 easy/medium/hard query 等）；  
   - 扩大 RL 数据覆盖（更多 query / 更丰富的 pointer 轨迹）。

你可以直接把这份 md 放到仓库的 `docs/` 目录，比如命名为：

```text
docs/LeverPlus_v3_RL_实现评估与改进计划.md
```

然后按第 5 节的 checklist 逐项推进。  
如果后面你在某一阶段（比如“只用 beam 的 RL”）得到新的实验结果，也可以在这份文档下继续追加“实验记录”小节。