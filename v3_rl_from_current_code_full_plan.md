# Lever-Plus v3 强化学习现状总结 & 下一阶段改造计划

> 基于当前 GitHub 仓库实际代码（`dataset_v3.py`, `reward_utils.py`, `generate_rl_data.py` 等）整理  
>
> - 现在 **已经实现了什么**、有哪些地方是对的；  
> - 哪些地方存在“结构性问题”（不是 bug，但影响效果）；  
> - 下一阶段 **具体要改哪些代码、怎么改**；  
> - 推荐的实验路线。

---

## 0. 一页纸总结

### 现在的状态（简版）

1. **RL 数据生成已经是“整条 pointer 序列 correctness”驱动的**  
   - 使用 `lever_lm/models/v3/generate_rl_data.py`：  
     - 用 v2/v3 SFT pointer 模型重新做 **beam + 温度采样 + 随机 pointer**；  
     - 用 VQA 模型对每条 pointer `[i1,i2]` 真实推理一次，得到：
       - `vqa_correct`（硬 0/1）
       - `vqa_acc_score`（软得分 [0,1]）
     - 写入 `rl_data_*.json` 的 `pointer_candidates`。
   - 此阶段 **不再使用 InfoScore/增益作为 RL reward 的主信号**。

2. **RL 训练（RCE + GRPO）用的是“hard+soft correctness reward”**
   - `lever_lm/utils/reward_utils.compute_reward_for_candidate`：
     - 默认 `reward_mode="hard_plus_soft"`，`hard_weight=soft_weight=1.0`；
     - `reward = vqa_correct + vqa_acc_score ∈ [0, 2]`；
   - `lever_lm/models/v3/dataset_v3.RLBeamDatasetWithEmbedding`：
     - 从 RL JSON 读取 `vqa_correct` / `vqa_acc_score`；
     - 调 `compute_reward_for_candidate(...)` 生成 `beam_rewards_raw`；
     - GRPO 用 group 内 Z-score 的 `beam_rewards`。

3. **实现层面已经基本对齐了“新 RL 方案”的设计**，没有大 bug。

### 效果不理想的主要原因

1. **训练目标只优化了 2‑shot**：  
   pointer 序列长度是 2，reward 也是纯 2‑shot correctness；  
   但评估看的是 1/2/3/4‑shot → 目标错位，1/3/4‑shot 没被优化。

2. **correctness reward 信号在很多 query 上比较“稀 + noisy”**：  
   - 很多 query 下所有 pointer 都是错的，`hard=0`；
   - `soft` 多靠 string match/contains 打分，噪声不小，组内差异很弱；
   - group 内 Z-score 之后，优势信号进一步被削弱。

3. **RL 数据量有限 + offline RL 的上限**：  
   - 只在 ~800 条 query 上生成 RL 数据；
   - pointer 原本已经不错，上限不高，稍有噪声就可能抵消/反向。

4. 再叠加一些细节（索引 fallback、VQA 文件路径没配好导致 fallback 到简单匹配），进一步削弱 RL 效果。

---

## 1. 现有实现：从代码层面精准描述

### 1.1 奖励工具：`lever_lm/utils/reward_utils.py`

**核心函数：**

```python
def compute_reward_for_candidate(
    beam_score: Optional[float] = None,
    logprob_score: Optional[float] = None,
    vqa_correct: Optional[int] = None,
    vqa_acc_score: Optional[float] = None,
    # 新增：reward 模式参数
    reward_mode: str = "hard_plus_soft",
    hard_weight: float = 1.0,
    soft_weight: float = 1.0,
    # 兼容旧接口的参数（默认不启用）
    alpha: float = 0.0,
    beta: float = 0.0,
    correctness_mode: str = "01",
    use_logprob: bool = False,
    reward_clip: Tuple[float, float] = (-5.0, 5.0)
) -> float:
    ...
```

**默认行为（关键）：**

```python
if reward_mode == "hard_plus_soft":
    hard = float(vqa_correct) if vqa_correct is not None else 0.0
    soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
    reward = hard_weight * hard + soft_weight * soft  # 默认 1+1
...
reward = max(reward_clip[0], min(reward_clip[1], reward))
```

> 只要外层没改参数，reward = `vqa_correct + vqa_acc_score`，范围 [0,2]，与设计一致。  
> legacy 分支（alpha/beta）默认不会用到。

### 1.2 RL 数据集：`lever_lm/models/v3/dataset_v3.py`

#### `RLBeamDatasetWithEmbedding`

构造函数（关键部分）：

```python
class RLBeamDatasetWithEmbedding(Dataset):
    def __init__(
        self,
        rl_data: Dict,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_indices: List[int],
        shot_num: int = 2,
        normalize_rewards: bool = True,
        reward_mode: str = "hard_plus_soft",
        hard_weight: float = 1.0,
        soft_weight: float = 1.0,
        reward_alpha: float = 0.0,
        reward_beta: float = 0.0,
        reward_correctness_mode: str = "01",
        use_logprob: bool = False,
        filter_gen_methods: Optional[List[str]] = None
    ):
        ...
        self.cand_idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}
        ...
        for query_id_str, query_data in rl_data.items():
            query_id = int(query_id_str)
            pointer_candidates = query_data.get("pointer_candidates", [])
            ...
            for c in pointer_candidates:
                pointer = c["pointer"]
                ...
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

`__getitem__`：

```python
beam_labels_tensor = torch.tensor(beam_labels, dtype=torch.long)
beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)

# 组内Z-score
beam_rewards_normalized = beam_rewards_raw.clone()
mean = beam_rewards_raw.mean()
std = beam_rewards_raw.std()
if std > 1e-12:
    beam_rewards_normalized = (beam_rewards_raw - mean) / std

result = {
    "query_id": query_id,
    "query_emb": query_emb,              # [d]
    "cand_emb": cand_emb,                # [K, d]
    "beam_labels": beam_labels_tensor,   # [num_candidates, shot_num]
    "beam_rewards": beam_rewards_normalized if self.normalize_rewards else beam_rewards_raw,
    "beam_rewards_raw": beam_rewards_raw
}
if beam_logprobs and all(lp is not None for lp in beam_logprobs):
    result["beam_logprobs"] = torch.tensor(beam_logprobs, dtype=torch.float32)
```

**现状结论：**

- RL reward 来源：`compute_reward_for_candidate(vqa_correct, vqa_acc_score, ...)`；
- 默认 `reward_mode="hard_plus_soft"`：  
  → `reward = vqa_correct + vqa_acc_score`；
- GRPO 看到的是 `beam_rewards`（Z-score 后的 group 内标准化 reward）；  
- RCE 看到的是 `beam_rewards_raw`（原始硬+软得分）。

**唯一隐性问题：**

```python
mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]
```

- 如果 `pointer` 里的某个 index 不在 `candidate_indices` 中，会默默用 `idx` 自己作为 embedding 下标，导致“错行 embedding”，但不会报错。

---

### 1.3 RL 数据生成：`lever_lm/models/v3/generate_rl_data.py`

#### 1）生成 pointer 候选

```python
pointer_candidates = generate_pointer_candidates_for_query(
    model=sft_model,
    query_emb=query_emb,
    cand_emb=cand_emb,
    num_beams=num_beams,
    temps=temps,
    num_samples_per_temp=num_samples_per_temp,
    num_random=num_random,
    beam_search_fn=None
)
```

这是在 **embedding 空间** 用当前 pointer SFT 模型重新做：

- beam search；
- 温度采样；
- 随机 pointer 组合。

**不再直接依赖旧的 InfoScore JSON 的 beam/score 作为 RL 数据来源**。

#### 2）用 VQA 计算 correctness

```python
query_item = dataset[query_id]
image = query_item.get("image")
question = query_item.get("question")
gt_answers = query_item.get("answers", [])

candidate_pool = [dataset[idx] for idx in candidate_indices]

for c in pointer_candidates:
    pointer = c["pointer"]
    original_pointer = [candidate_indices[p] for p in pointer]
    ex1 = candidate_pool[original_pointer[0]]
    ex2 = candidate_pool[original_pointer[1]]

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

`compute_vqa_accuracy` 的行为：

- 如果 `val_ques_path`, `val_ann_path`, `question_id` 都提供且匹配，  
  → 调用 `open_mmicl.metrics.vqa_metrics.compute_vqa_accuracy`，使用官方 VQA 评测（score 可是 0/0.3/0.6/1 等）；  
- 如果缺少或出错，则 fallback 到字符串匹配（exact match + contains），返回 `(correct, acc_score)`，其中部分匹配给 0.5 分。

**结论：**

- RL 数据确实是“在整条 pointer 序列下跑 VQA，拿软/硬得分”；  
- 如果你命令行没传 `val_ques_path`/`val_ann_path`，reward 会基于比较粗糙的字符串匹配，噪声会比较大。

#### 3）写出 RL JSON

```python
rl_data[query_id_str] = {
    "pointer_candidates": pointer_candidates_with_correctness
}
```

结构满足 `RLBeamDatasetWithEmbedding` 的预期：

```jsonc
{
  "12345": {
    "pointer_candidates": [
      {
        "pointer": [7,22],
        "gen_method": "beam",
        "beam_score": ...,
        "logprob_score": ...,
        "vqa_pred_answer": "...",
        "vqa_correct": 1,
        "vqa_acc_score": 1.0
      },
      ...
    ]
  }
}
```

---

## 2. “为什么效果没那么好？”——从实现 & 目标两层分析

### 2.1 不是“实现写挂了”，而是目标设计的问题

**现在的 RL 训练目标：**

- pointer_len = 2（shot_num=2）；
- reward = `vqa_correct(2-shot) + vqa_acc_score(2-shot)`；
- RCE + GRPO 全流程都是围绕这个 reward 优化。

**而评估时你关心：**

- 1‑shot：只用 pointer 第一条 ICD；
- 2‑shot：用前两条；
- 3/4‑shot：在 pointer score 排序后取前 3/4 条 ICD 做 VQA。

=> **只有 2‑shot 真正与训练目标对齐**；  
1/3/4‑shot 完全没有进入 reward，属于“副作用”。

这会导致：

- 1‑shot：  
  模型完全可能选那种“pair 很好，但第一条单独用一般”的 ICD → 一旦 pointer 策略向 2‑shot 优化，1‑shot 变差是正常的。
- 3/4‑shot：  
  pointer 模型只学“选出对 2‑shot 有利的 pair”，第三、第四条 ICD 对 reward 不起作用 → 3/4‑shot 性能只能靠运气和 side-effect，很难系统性提升。

### 2.2 correctness reward 本身的“稀 + noisy”

当前 reward：

```python
hard = vqa_correct ∈ {0,1}
soft = vqa_acc_score ∈ [0,1]  # 实际多为 0, 0.5, 1.0
reward = hard + soft
```

如果 `compute_vqa_accuracy` 走 fallback 分支（字符串匹配），则大致规则是：

- 完全匹配 → `(correct=1, acc_score=1.0)`；
- 包含关系 → `(correct=1, acc_score=0.5)`；
- 其他 → `(correct=0, acc_score=0.0)`。

在许多 query 下可能出现：

- 所有 pointer 都是错的 → `reward` 全是 0 或 0.5；
- 或者都半对半错 → `reward` 全是 0.5 或 1.5；
- group 内 Z-score 后，差异非常小，advantage 接近 0；
- 再考虑文本匹配本身的噪声，很容易把“看起来部分对”的错误样本 reward 拉高。

**结果：**

- RL 在许多 query 上拿到的信号很弱，甚至有方向噪声；
- 相比之下，旧 InfoScore 虽然理论不好，但在 group 内的数值差异有时更大，优势信号更鲜明，所以 shot=2 的提升能更稳定地体现出来。

### 2.3 offline RL 的硬上限（数据少 + 探索空间有限）

- RL 数据只在固定子集（比如 ~800 条 RandSampler）上生成；  
- pointer 原本由 v2 SFT 训练，表现已经不错；  
- offline RL 在“已有 SFT 轨迹的局部邻域”做 reweight，**理论能带来的提升就不会太大**；
- 再加上 reward 噪声，就很容易出现：
  - 小数据上似乎有提升（100 条上略好）；  
  - 数据量一旦变大（400/800 条），噪声平均后反而难以稳定超 v2。

---

## 3. 下一阶段要改什么 & 怎么改（代码级具体建议）

### 3.1 修掉 pointer→embedding 的隐形坑

**问题位置：**

`RLBeamDatasetWithEmbedding.__init__`：

```python
mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]
```

**潜在问题：**

- 如果 `idx` 不在 `candidate_indices` 里，会直接 fallback 用 `idx` 作为行号；
- 这意味着 pointer “指错行”，但不会 crash，debug 很难发现。

**建议改法：**

```python
# 原来：
mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]

# 建议改成：
mapped_pointer = []
for idx in pointer:
    if idx not in self.cand_idx_to_pos:
        raise KeyError(
            f"Pointer index {idx} not in candidate_indices "
            f"(query_id={query_id})"
        )
    mapped_pointer.append(self.cand_idx_to_pos[idx])
```

**带来的好处：**

- 一旦 RL JSON / embedding 映射有问题，会第一时间 crash，而不是训练一堆“错 ICD”；  
- 便于确认目前 RL 数据和 embedding 是完全对齐的。

---

### 3.2 强制/强烈依赖“官方 VQA 文件评测”而不是简单字符串匹配

**问题位置：**

`compute_vqa_accuracy(...)`：

```python
if val_ques_path and val_ann_path and question_id:
    try:
        # 调用 compute_vqa_accuracy_metric(...)
        ...
        return correct, acc_score
    except Exception:
        print("警告：使用文件方式计算准确率失败，回退到简单匹配")
# 否则走简单匹配：pred in gt / gt in pred，给 1.0 或 0.5 或 0.0
```

如果你在命令行生成 RL 数据时 **没有提供 `--val_ques_path` / `--val_ann_path`**，或者 question_id 对不上，就会走 fallback 简单匹配分支。

**建议改法 A（推荐）——在生成数据脚本层强约束：**

在 `main()` 里：

```python
# 如果是 VQA 类数据集，强制要求提供 val_ques_path/val_ann_path
if "vqa" in args.dataset.lower():
    if not args.val_ques_path or not args.val_ann_path:
        raise ValueError(
            "For VQA datasets, you must provide --val_ques_path and --val_ann_path "
            "to compute official VQA accuracy as RL reward."
        )
```

这样保证 RL 数据一定基于官方 metric，而不是简单字符串匹配。

**建议改法 B（轻一点）——在 compute_vqa_accuracy 内增加提示/开关：**

```python
def compute_vqa_accuracy(..., allow_fallback: bool = False):
    ...
    if val_ques_path and val_ann_path and question_id:
        try:
            ...
            return correct, acc_score
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(
                    f"VQA file-based accuracy failed for question_id={question_id}: {e}"
                )
            print("警告：使用文件方式计算准确率失败，回退到简单匹配: {e}")
    ...
```

同时在 `generate_rl_data(...)` 调用时显式地：

```python
correct, acc_score = compute_vqa_accuracy(
    ...,
    val_ques_path=val_ques_path,
    val_ann_path=val_ann_path,
    allow_fallback=False,   # 强制报错而不是悄悄 fallback
)
```

> 这样可以确保：  
>
> - 要么用官方 VQA metric，reward 更可信；  
> - 要么整个过程直接报错，不会悄悄变成 noisy 的字符串匹配。

---

### 3.3 明确训练脚本中 RL 配置，便于 “RCE-only/轻GRPO” 实验

**目标：**  
下一阶段不要立刻尝试“大力 GRPO 重塑策略”，而是先用“RCE-only + 轻量 GRPO”的方式，稳定 2‑shot 表现。

**你可以在 v3 训练 config / 脚本中显式增加：**

```yaml
# configs/train_v3_rl.yaml 示例
rce:
  epochs: 5        # 可以先设长一点，比如 5~10
  lr: 1e-4

grpo:
  epochs: 0        # 先做 RCE-only baseline
  lr: 5e-5
  kl_beta_init: 0.1
  kl_target_min: 0.01
  kl_target_max: 0.05
```

训练时：

1. 先跑一版 **RCE-only**（`grpo.epochs=0`）：
   - 看 2‑shot 是否持平或略优于 v2；  
   - 看 1/3/4‑shot 是否没有大崩。

2. 如果 RCE-only 看起来有一点点提升且不伤整体，  
   再加上一版 **轻 GRPO**：
   - `grpo.epochs=1`；
   - 小 LR（比如原来的一半）；
   - 稍大一点 KL β（限制策略偏离）。

在 `GRPOTrainer` 中，可以增加 log：

```python
logger.info(f"[GRPO] epochs={self.num_grpo_epochs}, lr={self.grpo_lr}, kl_beta_init={self.kl_beta}")
```

便于复现实验。

---

### 3.4 利用 `filter_gen_methods` 做一次 beam/sample/random 的 ablation（只需改配置）

你的 `RLBeamDatasetWithEmbedding` 已经支持：

```python
filter_gen_methods: Optional[List[str]] = None

...

if self.filter_gen_methods is not None:
    pointer_candidates = [
        c for c in pointer_candidates
        if c.get("gen_method") in self.filter_gen_methods
    ]
```

所以你不需要改代码，只要在构建 dataset 时设置不同组合即可：

```python
# 只用 beam pointer 做 RL
rl_ds_beam_only = RLBeamDatasetWithEmbedding(
    rl_data=rl_data,
    ...,
    filter_gen_methods=["beam"],
)

# 用 beam + sample
rl_ds_beam_sample = RLBeamDatasetWithEmbedding(
    rl_data=rl_data,
    ...,
    filter_gen_methods=["beam", "sample"],
)

# 全部（beam+sample+random）
rl_ds_all = RLBeamDatasetWithEmbedding(
    rl_data=rl_data,
    ...,
    filter_gen_methods=None,
)
```

**推荐实验顺序：**

1. beam-only；
2. beam+sample；
3. beam+sample+random；

对比各自的：

- 2‑shot 精度相对于 v2 的变化；  
- reward 分布（例如每个 query 内 reward variance）。

**目标：**  
如果发现 random 或高温 sample 的 pointer 带来明显噪声（某些 query 上 reward 全是垃圾），就可以考虑在正式 RL 中禁用它们或降低比例。

---

### 3.5 更长远：多 shot RL（可放到以后）

这一条你现在不用立刻做，但可以作为“下一代 v4 方案”的核心方向：

- Pointer 模型输出 `[i1, i2, i3, i4]`（固定 4 shot）；  
- 对每条 pointer 序列，分别构造：
  - 1‑shot prompt（只用 ICD1）；  
  - 2‑shot prompt（ICD1+ICD2）；  
  - 3‑shot prompt；  
  - 4‑shot prompt；
- 分别跑 VQA，得到：
  - `R1, R2, R3, R4`（各自的 hard+soft correctness）；
- 综合 reward：
  \[
  R_\text{total} = w_1 R1 + w_2 R2 + w_3 R3 + w_4 R4
  \]
- 用 `R_total` 去做 RCE + GRPO。

这样 1/2/3/4‑shot 的表现就都进入了目标函数，而不是寄希望于 2‑shot 的副作用。

> 这一条改动比较重（数据生成时间 ×4，代码也要改 pointer head），建议在当前 v3 方案稳定后，再做版本升级。

---

## 4. 推荐执行顺序（极简 checklist）

1. **立即修改的代码（容易 & 必要）**
   - `RLBeamDatasetWithEmbedding`：
     - 把 `mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) ...]` 改为严格检查版本，避免隐藏索引错误。
   - `generate_rl_data.py`：
     - 在 `main()` 或 `compute_vqa_accuracy` 调用处 **禁止 silent fallback**：
       - 没提供 `val_ques_path` / `val_ann_path` 或 question_id 不匹配时，直接报错，不走简单匹配。
2. **重新生成一版 RL 数据**
   - 确保这次生成：
     - 使用正确的 VQA 官方评估（不依赖简单匹配）；  
     - pointer → candidate_indices 映射无异常。
3. **跑一版 RCE-only 实验**
   - `grpo.epochs=0`；
   - 记录 v2 vs v3_new 在 1/2/3/4‑shot 上的差异；
   - 如果 2‑shot 至少不比 v2 差太多、3/4‑shot 有一点点改善，说明方向没错。
4. **在 RCE-only 基础上，加 1 epoch 轻 GRPO**
   - 小 lr + 强 KL（保持策略靠近 v2）；
   - 看是否能带来额外 0.x 个点的提升。
5. **用 `filter_gen_methods` 做 ablation**
   - beam-only vs beam+sample vs all；
   - 找出噪声来源（是否 random / 高温 sample 指南带偏了 RL）。

---

## 5. 文件命名建议

你可以把这份文档保存到仓库，例如：

```text
docs/v3_rl_status_and_next_steps.md
```

后续在 PR 或 issue 里直接引用其中的小节（比如 “见 §3.1 pointer→embedding 映射修改”、“见 §3.2 VQA 文件评估强制开启”），方便和别人讨论与复现。  
如果你愿意，我们也可以再为“多 shot RL 设计”单独写一份设计文档，当 v3 稳定之后做 v4 的升级方案。