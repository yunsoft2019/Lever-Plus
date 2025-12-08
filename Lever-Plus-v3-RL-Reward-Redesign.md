# Lever-Plus v3 强化学习方案改进计划（从增益得分改为绝对正确率）

> **目的：**  
> 在你当前仓库代码的基础上，系统性地梳理所有和强化学习 reward 相关的改动点，解释：
> - 之前的设计在哪里、做了什么；
> - 为何会导致性能提升有限，尤其是在 Qwen 上的“全是负增益”现象；
> - 现在应该怎么改，改完之后理论上避免哪些问题；
> - 给出尽量贴近你工程结构的伪代码，方便你直接实现。

---

## 0. 总体思路一页纸

- **旧思路（已经在代码里的部分）：**
  - 使用 `utils.get_info_score` 计算每个 ICD 的 **信息增益**：  
    \[
    \mathrm{InfoScore} = P(y|x, c) - P(y|x)
    \]
    这里 `x` 是 query，`c` 是新加的 ICD；对 Qwen 来说这个值常常是负数。  
  - 在 v3（RCE + GRPO）中，把这个 **InfoScore / beam 分数** 作为 `beam_rewards`，进入：
    - `PointerSelectorV3.compute_rce_loss(...)`
    - `PointerSelectorV3.compute_grpo_loss(...)`  
  - Correctness（是否回答正确）要么没有显式使用，要么只是作为辅助。

- **核心问题：**
  - RL 的目标其实是：  
    > 在给定 `shot=2` 的前提下，选一条 ICD 序列，让 **最终 VQA 正确率最高**。  
  - 但现在 reward 用的是“**增益**”：
    - \(\mathrm{gain} = P(y|x,c) - P(y|x)\)
    - 它衡量的是“比**上一条 ICD** 好多少”，不是“在 **2-shot 整体** 下答对的概率有多高”。
  - 在 Qwen 上，常见情况是：
    - 1-shot 已经很好；
    - 加第二条 ICD 只会让结果变差 → 所有 2-shot 增益都是负的；
    - 训练就变成了“在 1-shot 基线下，找一个最不拉胯的第二条 ICD”，而不是“在 2-shot 里找到 **全局最优组合**”。

- **改进思路：**
  1. **数据层面：** 对每一条 pointer 序列（两个 ICD）**必须记录** VQA 的绝对正确率 `vqa_acc_score`（或 0/1 correctness）。
  2. **reward 层面：** 把 `beam_rewards` 改成：
     - **主 reward：** 整条 pointer 序列下的绝对正确率（例如 VQAv2 acc ∈ [0,1]）；  
     - InfoScore / 增益分数如果保留，只作为 *辅助 shaping* 或分析用，不再是主目标。
  3. **RCE + GRPO：**
     - RCE 阶段：`beam_rewards` 使用 correctness，权重更直观；
     - GRPO 阶段：`beam_rewards` 直接等于 correctness，`compute_advantage` 只做组内排序 / 标准化。

---

## 1. 当前实现回顾：关键代码位置 & 逻辑

### 1.1 InfoScore（增益得分）的计算位置

文件：`utils.py`

```python
@torch.inference_mode()
def get_info_score(interface, choosed_icd_seq_list, candidate_set, batch_size, split_token=None, construct_order="left"):
    # ...
    # 1. 计算 P(y | x)
    cond_prob = interface.get_cond_prob(x_input, mask_length=[mask_length])

    # 2. 计算 P(y | x, c) 对每个候选 ICD
    # ...
    new_cond_prob = interface.get_cond_prob(add_new_icd_input, mask_length=mask_length_list)

    sub_info_score = new_cond_prob - cond_prob
    info_score_list.append(sub_info_score)

    return torch.cat(info_score_list)
```

- **本质：**  
  - 对于每个新增 ICD，计算 “加 ICD 前后的 cond_prob 差值（信息增益）”；  
  - 这是一个 **局部增量信号**，描述“加了这个示例之后答案好/坏了多少”。

- **Qwen 特殊性：**
  - 注释里已经写明：Qwen 这边 `new_cond_prob - cond_prob` 多数是负数；
  - 束搜索时用 `topk(largest=False)` 选“负数绝对值最小”的那条 beam。

### 1.2 PointerSelectorV3 的 RL 训练接口

文件：`lever_lm/models/v3/pointer_selector_v3.py`

关键接口：

```python
class PointerSelectorV3(PointerSelectorV2):
    # ...

    def compute_rce_loss(self, query_emb, cand_emb, beam_labels, beam_rewards, ...):
        # beam_rewards: [B, num_beams]
        # 通过 softmax(reward/τ) 得到权重 w_i
        # 再对所有 beam 的 CE loss 加权求和

    def compute_grpo_loss(self, query_emb, cand_emb, beam_labels, beam_rewards, old_log_probs, use_top_k=None):
        # beam_rewards: [B, num_beams]
        # advantages = self.compute_advantage(beam_rewards, normalize=True)
        # ratio = exp(new_log_probs - old_log_probs)
        # PPO 损失 + KL 惩罚
```

以及：

```python
def compute_advantage(self, rewards, normalize: bool = True, use_rank: bool = True):
    # 默认用“基于排名的归一化”：
    # - 把 reward 排序
    # - 映射到 [-1, 1]
    # - 再做裁剪
```

- **可见：**
  - RCE 和 GRPO 都把 `beam_rewards` 视为“越大越好”的评分；
  - 当前代码并不关心这些 reward 是“增益”还是“绝对正确率”，由上游数据决定。

### 1.3 新增的 RL 数据生成工具

文件：`lever_lm/models/v3/rl_data_generation.py`

- `beam_search_pointer(...)`：基于 PointerSelector（v2 / v3）的 beam search，产出若干 pointer 序列 + `score` + `logprob`。
- `sample_pointer_with_temperature(...)`：用温度采样 pointer（你已经完成了这一步 ✅）。
- `generate_pointer_candidates_for_query(...)`：组合 beam + 采样 + 随机，形成多种 pointer 序列。
- `evaluate_pointer_candidate(...)`：

```python
def evaluate_pointer_candidate(vqa_model, image, question, candidate_pool, pointer, ground_truth_answers, build_vqa_prompt_fn, compute_vqa_accuracy_fn):
    # 1) 根据 pointer 取出两个示例 ex1, ex2
    # 2) 构造 prompt
    # 3) vqa_model.generate(prompt)
    # 4) correct, acc_score = compute_vqa_accuracy_fn(...)
    return pred_answer, int(correct), float(acc_score)
```

> 这一段非常关键：你已经有了“**整条 pointer 序列的绝对 correctness（acc_score）**”，但在后续的 RL 管道里，reward 仍然是 InfoScore / 增益，这就是可以改的核心点。

---

## 2. 核心问题：为什么“增益 correctness”会限制强化学习效果？

### 2.1 目标与 reward 不对齐

- **真正的目标：**  
  在固定 `shot=2` 设置下，选择两个 ICD，使得 **最终 VQA 正确率最高**。

- **当前 reward：**  
  多数地方使用的是类似：
  \[
  \mathrm{gain} = P(y|x, c) - P(y|x)
  \]
  或基于 InfoScore 的排序。

- **后果：**
  - 当第一条 ICD 已经很强（1-shot 准确率很高）时：
    - 几乎所有“加第二条 ICD”的组合都会让结果变差（增益为负）；
    - RL 只能在“都负”的空间里挑“谁负得没那么厉害”；
    - 理论上可能存在一种组合 `(icd1*, icd2*)`，其 **2-shot 准确率最高**，但相对于 `(icd1*)` 的增益仍然是负的。
  - 于是：
    - RL 在优化“**相对 1-shot 的增益**”，而不是“**2-shot 下的绝对正确率**”。

### 2.2 “基线依赖 action”的 reward shaping 问题

- 形式上你现在做的是：
  \[
  r'(a) = r(a) - b(a)
  \]
  其中 \(b(a)\) 是“只用前一条 ICD 时的正确率”（或者 InfoScore 的某种 baseline）。

- 在 RL 里，如果 baseline **只依赖状态**，或者是 potential-based shaping，则不会改变最优策略。  
- 但现在 baseline 和 action（选择哪条 ICD 序列）耦合：
  - 不同第一条 ICD 会有不同 baseline；
  - 回到“2-shot 的全局最优组合”时，其 reward 可能被 baseline 减掉一大块，以至于排序被颠倒。

**典型现象：**

- 组合 A：  
  - 1-shot：0.90  
  - 2-shot：0.85 → 增益 -0.05

- 组合 B：  
  - 1-shot：0.50  
  - 2-shot：0.70 → 增益 +0.20

- 对 RL 来说：B > A（增益大）；  
- 对最终 2-shot 准确率：A > B（0.85 > 0.70）。

这就是你直觉里“会丢掉很多最优解”的根源。

### 2.3 Qwen 场景下“全负增益”的特殊问题

你提到的现象：

> 第一轮 beam 选出来的是得分最高的，然后 shot=2 的时候，所有的序列增益都是负的，只能比谁负得小。

- 对 Qwen：
  - 1-shot 已经非常强；
  - 增加第二条 ICD 经常只会扰动分布，变差的幅度不同；
- 在这种设定下：
  - reward 的动态范围非常窄（都在负区间且差距很小）；
  - 组内 advantage 接近 0，RL 梯度极小；
  - 更重要的是：**学习方向偏向“保持 1-shot 策略 + 找一个最不坏的 second shot”**，而不是探索少数真正有价值的 2-shot 组合。

---

## 3. 改动总览：从“增益 correctness”到“绝对 correctness”

下面按“数据 → reward 定义 → RCE → GRPO → 调参与日志”五个层级，列出所有需要/建议的改动。

### 3.1 数据层：确保每条 pointer 都有绝对 correctness

**当前状态：**

- `rl_data_generation.evaluate_pointer_candidate(...)` 已经返回 `correct` 和 `acc_score`；
- 你需要确认：
  - 这些字段是否在 RL 数据 JSON / JSONL 中被保存；
  - 字段名是否固定，比如 `vqa_correct`, `vqa_acc_score`。

**改动要求：**

- 在 RL 数据文件中，每条 pointer 样本必须包含：

```jsonc
{
  "pointer": [7, 22],
  "gen_method": "beam" / "sample" / "random",
  "beam_rank": 0,
  "beam_score": 2.13,          // 可选
  "logprob_score": -1.56,      // 可选
  "temperature": null,         // 可选
  "vqa_pred_answer": "a book",
  "vqa_correct": 1,            // 0/1
  "vqa_acc_score": 1.0         // [0,1]，例如 VQAv2 风格
}
```

**问题 &后果（如果不做）：**

- 没有绝对 correctness，就永远只能用 InfoScore / beam 分数做 reward；
- RL 无法直接优化“最终是否答对”这个核心指标。

**伪代码（生成并保存）：**

```python
# 伪代码：在 RL 数据生成主脚本中（调用 rl_data_generation 的地方）
for each query in dataset:
    pointer_candidates = generate_pointer_candidates_for_query(
        model=pointer_selector_sft,
        query_emb=query_emb,
        cand_emb=cand_emb,
        num_beams=5,
        temps=(1.0, 1.3),
        num_samples_per_temp=2,
        num_random=1,
    )

    records = []
    for pc in pointer_candidates:
        pointer = pc["pointer"]
        pred_answer, correct, acc_score = evaluate_pointer_candidate(
            vqa_model=vqa_model,
            image=image,
            question=question,
            candidate_pool=candidate_pool,
            pointer=pointer,
            ground_truth_answers=gt_answers,
            build_vqa_prompt_fn=build_vqa_prompt,
            compute_vqa_accuracy_fn=compute_vqa_accuracy,
        )

        record = {
            **pc,  # 保留 pointer / gen_method / beam_score 等
            "vqa_pred_answer": pred_answer,
            "vqa_correct": int(correct),
            "vqa_acc_score": float(acc_score),
        }
        records.append(record)

    save_to_jsonl(query_id, records)
```

---

### 3.2 reward 定义层：用“整条序列的绝对正确率”做主 reward

**当前（推测）逻辑：**

- 在构造 RL 训练样本时，大致会有一段逻辑把原始分数（InfoScore / beam_score）转成 `beam_rewards`；例如：

```python
# 伪代码（旧版思路）
for each query:
    beams = load_beam_records(query)
    rewards = []
    for beam in beams:
        info_gain = beam["info_score"]   # 由 get_info_score 得到的增益
        rewards.append(info_gain)

    beam_rewards = torch.tensor(rewards)  # [num_beams]
```

**问题：**

- 这里的 `beam_rewards` 完全由增益构成；
- 与“最终答对概率”不对齐；
- 在 Qwen 上所有增益为负时，reward 空间塌缩。

**改动方案：**

1. 定义一个新的 reward 构造函数，显式选择“**主 reward = correctness**，增益 InfoScore 仅作为可选附加项”。

2. 推荐的主模式（默认）：

```python
def build_reward(sample, mode="correctness_only", alpha=0.0, beta=1.0):
    """
    sample: 单条 pointer 记录（已经包含 vqa_acc_score / info_score）
    mode:   reward 模式
    alpha:  InfoScore 权重
    beta:   correctness 权重
    """
    acc = sample["vqa_acc_score"]  # [0,1] 或 0/1

    if mode == "correctness_only":
        # 纯 correctness，最简单、最对齐目标
        reward = acc

    elif mode == "correctness_plus_info":
        info = sample.get("info_score", 0.0)
        # 将 info 正则化到 [0,1]，保证方向一致（大为好）
        info_norm = normalize_info_score(info)
        reward = beta * acc + alpha * info_norm

    elif mode == "binary_correctness":
        # 只看是否答对，忽略部分正确
        reward = 1.0 if sample["vqa_correct"] == 1 else 0.0

    else:
        raise ValueError(f"Unknown reward mode: {mode}")

    return reward
```

3. 推荐的配置（先用最干净的版本）：

- `mode = "correctness_only"`
- 环境变量 / Hydra 配置：
  - `REWARD_ALPHA = 0.0`
  - `REWARD_BETA = 1.0`

**改完后，构造整个 query 的 `beam_rewards`：**

```python
def build_beam_rewards_for_query(beams, cfg):
    """
    beams: 某个 query 下的所有 pointer beam 记录（List[dict]）
    cfg.reward: 包含 mode / alpha / beta 等配置
    """
    rewards = []
    for b in beams:
        r = build_reward(
            sample=b,
            mode=cfg.reward.mode,       # e.g. correctness_only
            alpha=cfg.reward.alpha,     # e.g. 0.0
            beta=cfg.reward.beta,       # e.g. 1.0
        )
        rewards.append(r)

    # [num_beams] → Tensor
    return torch.tensor(rewards, dtype=torch.float32)
```

**避免的问题：**

- 不再依赖“增益”的符号和幅度；
- 即使所有 InfoScore 为负，reward 仍然是 [0,1] 区间的有效信号；
- 直接最大化“2-shot 下是否答对”的概率，目标一致。

---

### 3.3 RCE 阶段：用 correctness 控制样本权重

**当前逻辑（参考 `PointerSelectorV3.compute_rce_loss`）：**

- 输入：`beam_labels`, `beam_rewards`；
- 内部有两种模式：
  - `use_top1_only=True` 时退化成 v2 的普通 CE；
  - 否则：
    - 先对 `beam_rewards` 进行归一化（默认基于排名）；
    - 再用 `softmax(reward / temperature)` 得到权重；
    - 对每条 beam 的 CE loss 按权重加权求和。

**问题（如果 `beam_rewards` 是增益）：**

- 增益在 Qwen 上通常差异极小，尤其是全负时；
- 排名归一化虽然缓解了数值问题，但依然是在“相对 1-shot 增益”的空间里调权重。

**改动方案：**

1. 把 RCE 阶段传入的 `beam_rewards` 换成刚才定义的 **correctness-based reward**：

```python
# 在 RCE 训练 dataloader / training loop 中
for batch in rce_dataloader:
    query_emb = batch["query_emb"]      # [B, d]
    cand_emb  = batch["cand_emb"]       # [B, K, d]
    beam_labels = batch["beam_labels"]  # [B, num_beams, S]

    # 新的 correctness-based reward
    beam_rewards = batch["beam_rewards_correctness"]  # [B, num_beams], 在构造 dataset 时就填好

    rce_loss = model.compute_rce_loss(
        query_emb=query_emb,
        cand_emb=cand_emb,
        beam_labels=beam_labels,
        beam_rewards=beam_rewards,
        temperature=current_rce_temp,
        use_rank_normalization=True,    # 可以先保持，后续再实验是否改为 False
        use_top1_only=False,
    )

    rce_loss.backward()
    optimizer.step()
```

2. 可以后续根据实验，尝试：

- 当 reward 是 [0,1] 的 correctness 时，`use_rank_normalization=False` + Z-score 可能效果更好，因为绝对差异有意义；
- 但第一步建议先只换 reward，不动归一化方式，避免同时改两件事。

**改后避免的问题：**

- RCE 权重高的 beam 真正是“最终答对概率高”的组合；
- 不再出现“1-shot 非常强但 2-shot 略弱，因此增益为负，权重被打低”的情况。

---

### 3.4 GRPO 阶段：用 correctness 做 beam_rewards + group advantage

**当前逻辑（`compute_grpo_loss`）：**

```python
def compute_grpo_loss(self, query_emb, cand_emb, beam_labels, beam_rewards, old_log_probs, use_top_k=None):
    # 1) 可选：按 beam_rewards 选择 top-k beam（课程学习）
    # 2) new_log_probs = self.compute_log_probs_per_beam(...)
    # 3) advantages = self.compute_advantage(beam_rewards, normalize=True)
    # 4) ratio = exp(new_log_probs - old_log_probs)
    # 5) PPO 损失 + KL 惩罚
```

**问题（如果 beam_rewards 是增益）：**

- 同 RCE：advantage 依赖于“相对增益”，而非“绝对正确率”；
- 在 Qwen 场景中，很容易出现所有 advantage 非常接近的情况，梯度几乎消失。

**改动方案：**

1. 和 RCE 一样，传入 correctness-based reward：

```python
for batch in grpo_dataloader:
    query_emb = batch["query_emb"]        # [B, d]
    cand_emb  = batch["cand_emb"]         # [B, K, d]
    beam_labels   = batch["beam_labels"]  # [B, num_beams, S]
    old_log_probs = batch["old_log_probs"]# [B, num_beams]

    # 新的 reward：整条序列的正确率
    beam_rewards = batch["beam_rewards_correctness"]  # [B, num_beams]

    grpo_out = model.compute_grpo_loss(
        query_emb=query_emb,
        cand_emb=cand_emb,
        beam_labels=beam_labels,
        beam_rewards=beam_rewards,
        old_log_probs=old_log_probs,
        use_top_k=cfg.grpo.use_top_k,
    )

    loss = grpo_out["loss"]
    loss.backward()
    optimizer.step()
```

2. 关于 `compute_advantage` 的两个开关：

- **第一阶段（推荐做法）：**
  - 保持默认：`normalize=True, use_rank=True`；
  - 这样 reward 只决定 beam 的排序，不依赖具体数值；
  - 对于 correctness（0 或 1），排序刚好就是“答对的 > 答错的”。

- **第二阶段（可尝试的改进）：**
  - 当你确认 correctness reward 已经跑得稳定，可以尝试传 `use_rank=False`，改用 Z-score：
    ```python
    advantages = self.compute_advantage(
        beam_rewards,
        normalize=True,
        use_rank=False,    # 让 reward 的幅度参与优势计算
    )
    ```
  - 这能利用“部分正确 vs 完全正确”之间的差异（比如 VQAv2 acc=0.3/0.7/1.0）；
  - 但需要注意优势的 scale，防止 PPO ratio 波动过大。

**改后避免的问题：**

- GRPO 直接在“最终正确率”的空间里做策略梯度；
- 对于 Qwen，全负 InfoScore 场景不再导致 reward 空间塌缩；
- `use_top_k` 的课程学习也更加直观：就是“只用 top-k 正确率最高的 beam”。

---

### 3.5 配置与日志：引入明确的 reward 模式开关

为了方便以后做 ablation 和回滚，建议在 config / 环境变量中显式添加：

```yaml
# configs/train_v3_rl.yaml 中类似结构
reward:
  mode: "correctness_only"     # ["correctness_only", "correctness_plus_info", "binary_correctness"]
  alpha: 0.0                   # InfoScore 权重
  beta: 1.0                    # correctness 权重

grpo:
  use_top_k: 3                 # 课程学习阶段的 top-k
```

训练脚本里：

```python
logger.info(f"RL reward mode = {cfg.reward.mode}, alpha = {cfg.reward.alpha}, beta = {cfg.reward.beta}")
```

---

## 4. “从增益到绝对正确率”的前后对比伪代码

### 4.1 旧版核心（示意）

```python
# 旧版：以 InfoScore 增益为主 reward

for each query:
    beams = load_beam_records(query)   # 每个 beam 里有 pointer, info_score 等

    beam_labels = []
    beam_rewards = []

    for b in beams:
        beam_labels.append(b["pointer"])             # [S]
        gain_score = b["info_score"]                 # P(y|x,c) - P(y|x)
        beam_rewards.append(gain_score)

    beam_labels = torch.tensor(beam_labels)          # [num_beams, S]
    beam_rewards = torch.tensor(beam_rewards)        # [num_beams]

    # RCE
    rce_loss = model.compute_rce_loss(
        query_emb, cand_emb,
        beam_labels.unsqueeze(0),                    # [1, num_beams, S]
        beam_rewards.unsqueeze(0),                   # [1, num_beams]
        temperature=rce_temp,
    )

    # GRPO
    grpo_out = model.compute_grpo_loss(
        query_emb, cand_emb,
        beam_labels.unsqueeze(0),
        beam_rewards.unsqueeze(0),
        old_log_probs,                               # 来自 SFT 模型
        use_top_k=cfg.grpo.use_top_k,
    )
```

### 4.2 新版核心（示意）

```python
# 新版：以整条 pointer 序列的 VQA 正确率为主 reward

for each query:
    beams = load_beam_records(query)
    # beams 中现在包含：
    # - "pointer": [i, j]
    # - "vqa_correct": 0/1
    # - "vqa_acc_score": [0,1]
    # - "info_score": 可选

    beam_labels = []
    beam_rewards_correctness = []

    for b in beams:
        beam_labels.append(b["pointer"])

        # 主 reward = correctness
        r = build_reward(
            sample=b,
            mode=cfg.reward.mode,        # 推荐 "correctness_only"
            alpha=cfg.reward.alpha,      # e.g. 0.0
            beta=cfg.reward.beta,        # e.g. 1.0
        )
        beam_rewards_correctness.append(r)

    beam_labels = torch.tensor(beam_labels)                 # [num_beams, S]
    beam_rewards_correctness = torch.tensor(beam_rewards_correctness)  # [num_beams]

    # RCE：使用 correctness-based reward
    rce_loss = model.compute_rce_loss(
        query_emb, cand_emb,
        beam_labels.unsqueeze(0),                           # [1, num_beams, S]
        beam_rewards_correctness.unsqueeze(0),              # [1, num_beams]
        temperature=rce_temp,
        use_rank_normalization=True,                        # 可配置
    )

    # GRPO：同样使用 correctness-based reward
    grpo_out = model.compute_grpo_loss(
        query_emb, cand_emb,
        beam_labels.unsqueeze(0),
        beam_rewards_correctness.unsqueeze(0),
        old_log_probs,                                      # [1, num_beams]
        use_top_k=cfg.grpo.use_top_k,
    )
```

---

## 5. 实施顺序建议（降低踩坑风险）

1. **确认数据：**
   - 检查 RL 数据文件（JSON/JSONL）中是否已经有：
     - `vqa_correct` / `vqa_acc_score`；
     - 如果没有，先用你新加的 `rl_data_generation.py` 重新生成 RL 数据。

2. **实现 reward 构造函数：**
   - 在一个单独的模块（例如 `lever_lm/models/v3/reward_builder.py` 或 RL 数据读取脚本里）加上 `build_reward(...)`；
   - 确保能从一条 beam record 生成一个 scalar reward。

3. **修改 RCE + GRPO 调用：**
   - 在构造 mini-batch 时，把 `beam_rewards` 改为 correctness-based；
   - 第一步不要改 `compute_advantage` 的 `use_rank` / `normalize` 行为，只换 reward 源；
   - 确认训练能稳定跑通。

4. **对比实验：**
   - 保持 RCE/GRPO 超参不变，做一组 baseline：
     - v2（纯 SFT）；
     - v3（旧版增益 reward）；
     - v3（新版 correctness reward）。
   - 对比：
     - pointer selector 本身的选样准确率（offline evaluation）；
     - 下游 VQA 正确率。

5. **进一步调优（可选）：**
   - 尝试：
     - RCE 使用 `use_rank_normalization=False` + Z-score；
     - GRPO 中 `use_rank=False`；
     - `reward.mode = "correctness_plus_info"`，alpha 取 0.1~0.3 做轻量 shaping。

---

## 6. 小结：每个改动点的“前因 → 后果 → 解决方式”

为了你对照代码时更方便，这里再用一张“问题 → 改法”总表：

| 层级 | 之前做法 | 潜在问题 | 改法（本方案） |
|------|----------|----------|----------------|
| 数据 | 只记录 beam 分数 / InfoScore，correctness 信息不稳定或未统一使用 | 无法直接优化“最终是否答对”，RL 只能围绕增益打转 | 在 RL 数据中为每条 pointer **必备** `vqa_correct` / `vqa_acc_score` 字段 |
| reward 定义 | `beam_rewards = InfoScore（增益）` | 目标与 reward 不对齐；Qwen 场景信息增益全负，空间塌缩 | 定义 `build_reward(...)`，主 reward = correctness（acc_score），InfoScore 仅做可选 shaping |
| RCE | 用增益做 softmax 权重 | 1-shot 极强时，高质量 2-shot 组合可能被误判为“负增益”而权重过低 | RCE 的 `beam_rewards` 换为 correctness-based，先保持 rank 归一化，确保权重与“最终正确率”对齐 |
| GRPO | 增益 → 组内 advantage | 优化的是“相对 1-shot 增益”，不是“2-shot 绝对正确率”；Qwen 全负场景 advantage 很弱 | GRPO 的 `beam_rewards` 换为 correctness-based，`compute_advantage` 仍做组内排序，必要时再尝试 Z-score 模式 |
| Qwen InfoScore | 负数、跨度小 | 作为主 reward 时难以解释、难以调参 | 保留 InfoScore 作为分析指标或轻量 shaping 信号，不再作为主 reward |

---

如果你愿意，可以把这份文件直接保存为仓库里的  
`docs/v3_rl_reward_redesign.md`（或任意你喜欢的名字），后续在代码中加注释时直接引用其中的小节编号。

---

**文件说明**

本 Markdown 已经按你的要求整理好，你可以直接下载并放入仓库使用。