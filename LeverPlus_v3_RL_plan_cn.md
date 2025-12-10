# Lever-Plus RL 改造计划（短期 & 中期）

> 说明：  
> 以下是基于你**亲自贴出来的关键代码**（包括 `dataset_v3.py`、`reward_utils.py`、`generate_rl_data.py` 等）和你仓库里公开的设计文档整理的改造计划。  
> 我没法逐行扫描整个仓库的每个文件，但对 **v3/RL 相关的主链路** 已经完整过了一遍：  
> RL 数据生成 → reward 计算 → Dataset/Collate → GRPO 训练。  
>
> 目标：  
>
> - 短期：在现有设计不大动的前提下，修掉隐蔽坑、降低噪声，让 v3 至少在 2‑shot 上稳定不输 v2；  
> - 中期：逐步对齐 1/2/3/4‑shot 目标，提升 RL 的上限。

---

## 0. 当前 RL 实现的简要现状

### 已经做对的事情

1. **Reward 设计已经从“增益得分”切到“绝对 correctness”**  
   - RL 数据通过 `generate_rl_data.py` 生成：  
     - 使用 v2/v3 SFT pointer 模型在 embedding 上做 beam + 温度采样 + 随机 pointer；
     - 对每条 pointer `[i1, i2]` 使用 VQA 模型真实推理，得到：
       - `vqa_correct ∈ {0,1}`  
       - `vqa_acc_score ∈ [0,1]`
     - 写入 `pointer_candidates` 的字段中。
   - `reward_utils.compute_reward_for_candidate(...)` 默认：
     - `reward_mode="hard_plus_soft"`，`hard_weight=soft_weight=1.0`；
     - 即：`reward = vqa_correct + vqa_acc_score ∈ [0,2]`。

2. **RL Dataset 正确地使用了 correctness 来构造 reward**  
   - `RLBeamDatasetWithEmbedding` 从 RL JSON 中读 `vqa_correct` / `vqa_acc_score`；
   - 调 `compute_reward_for_candidate` 得到 reward → `beam_rewards_raw`；
   - 再做一次组内 Z‑score，得到 `beam_rewards` 供 GRPO 使用。

3. **GRPO 训练流程整体结构合理**  
   - RCE 阶段用 `beam_rewards_raw` 做加权 CE（RCE）；
   - GRPO 阶段用 `beam_rewards` + old_log_probs 做 PPO 风格更新；
   - 同时有 KL penalty、自适应 β 等机制。

### 主要瓶颈不在“实现错了”，而在：

1. **训练目标只 optimize 了 2‑shot**，但你评估看的是 1/2/3/4‑shot → 明显目标错位；
2. **correctness reward 本身在很多 query 上比较稀 + noisy**，尤其是 fallback 到字符串匹配时；
3. **RL 数据量和覆盖有限**（通常只是几百个 query），offline RL 提升空间有限；
4. 再叠加一些隐蔽的代码习惯（如宽松 index 映射、二次 Z‑score、默默 fallback），进一步削弱信号。

---

## 1. 短期修改（1～2 个实验轮次内能完成）

这些改动基本不需要大改结构，但会显著降低隐含风险，提高 RL 收敛的可信度。

---

### ST‑1：严格化 pointer → embedding 的索引映射

**文件位置**

- `lever_lm/models/v3/dataset_v3.py`
  - `class RLBeamDatasetWithEmbedding(Dataset).__init__`

**当前代码（核心片段）**

```python
self.cand_idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}

...

for query_id_str, query_data in rl_data.items():
    query_id = int(query_id_str)
    pointer_candidates = query_data.get("pointer_candidates", [])

    ...

    for c in pointer_candidates:
        pointer = c["pointer"]
        assert len(pointer) == shot_num, f"pointer长度不匹配: {len(pointer)} vs {shot_num}"

        # 映射pointer中的索引为candidate位置
        mapped_pointer = [self.cand_idx_to_pos.get(idx, idx) for idx in pointer]
        beam_labels.append(mapped_pointer)
        ...
```

**问题**

- `self.cand_idx_to_pos.get(idx, idx)` 的含义是：
  - 如果 `idx` 在 `candidate_indices` 中 → 用映射后的位置；
  - 否则 → 直接用原值 `idx` 当作行号。
- 这意味着：
  - 一旦 RL JSON 中的 `pointer` 和 `candidate_indices` 不同步（一个是全局 id，一个是位置 index），这行代码不会报错，而是“悄悄用错行 embedding”。

**修改目标**

- 强制：pointer 里的每个 `idx` 必须在 `candidate_indices` 映射里；
- 否则直接报错（方便你发现数据/embedding mis-match）；

**推荐修改（伪代码）**

```python
for c in pointer_candidates:
    pointer = c["pointer"]
    assert len(pointer) == shot_num, f"pointer长度不匹配: {len(pointer)} vs {shot_num}"

    # 严格映射：如果 idx 不在 candidate_indices，就抛错
    mapped_pointer = []
    for idx in pointer:
        if idx not in self.cand_idx_to_pos:
            raise KeyError(
                f"[RLBeamDatasetWithEmbedding] pointer index {idx} "
                f"not in candidate_indices (query_id={query_id})"
            )
        mapped_pointer.append(self.cand_idx_to_pos[idx])

    beam_labels.append(mapped_pointer)
    ...
```

**原因与收益**

- 防止“silent bug”：数据和 embedding 对不齐却继续训练；
- 确保 pointer 指向的 embedding 行真的是你想要的 candidate；
- 如果 RL JSON 生成逻辑出错，会第一时间 crash，而不是浪费几个 GPU 小时。

---

### ST‑2：统一 reward 归一化逻辑，避免“二次 Z‑score”

**相关位置**

- Dataset 侧：
  - `lever_lm/models/v3/dataset_v3.py`
    - `RLBeamDatasetWithEmbedding.__getitem__`
- GRPO 侧：
  - `lever_lm/utils/reward_utils.py`
    - `compute_group_relative_advantage`
  - `lever_lm/workflows/grpo_post_train.py`
    - 调用上述函数并计算 GRPO loss 的地方

**当前 Dataset 行为（片段）**

```python
beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)  # [num_candidates]

# 归一化（组内Z-score）
beam_rewards_normalized = beam_rewards_raw.clone()
mean = beam_rewards_raw.mean()
std = beam_rewards_raw.std()
if std > 1e-12:
    beam_rewards_normalized = (beam_rewards_raw - mean) / std

result = {
    ...
    "beam_rewards": beam_rewards_normalized if self.normalize_rewards else beam_rewards_raw,
    "beam_rewards_raw": beam_rewards_raw,
}
```

**GRPO 典型行为（推测）**

```python
from lever_lm.utils.reward_utils import compute_group_relative_advantage

advantages = compute_group_relative_advantage(
    rewards=batch["beam_rewards"],   # [B, N]
    normalize=True,
    clip_range=5.0,
)
```

`compute_group_relative_advantage` 内部又是：

```python
if normalize:
    advantages = normalize_rewards_zscore(rewards, dim=-1)
else:
    advantages = rewards - rewards.mean(dim=-1, keepdim=True)
advantages = clip_advantages(advantages, clip_range)
```

**问题**

- Dataset 已经对每个 query 组内做了一次 Z‑score；
- GRPO 再对已经 Z‑score 的数据做一次 Z‑score；
- `reward` 本身范围 [0,2]，组内差异小，经过两次归一化后，advantage 极小、差异被进一步抹平。

**修改目标**

- **只在一个地方做 Z‑score**；
- 推荐：Dataset 只提供原始 reward，**GRPO 内统一做 group-relative advantage**。

**推荐修改方案 A（最清晰）**

1. 修改 `RLBeamDatasetWithEmbedding.__getitem__`：

   ```python
   beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)
   
   result = {
       "query_id": query_id,
       "query_emb": query_emb,
       "cand_emb": cand_emb,
       "beam_labels": beam_labels_tensor,
       # 不再提前归一化，直接返回原始 reward
       "beam_rewards": beam_rewards_raw,
       "beam_rewards_raw": beam_rewards_raw,
   }
   ```

2. 在 GRPO 训练代码中（`grpo_post_train.py`），统一处理 advantage：

   ```python
   from lever_lm.utils.reward_utils import compute_group_relative_advantage
   
   rewards = batch["beam_rewards"]  # [B, num_candidates]
   advantages = compute_group_relative_advantage(
       rewards,
       normalize=True,
       clip_range=5.0,
   )  # [B, num_candidates]
   
   ratio = torch.exp(log_probs_new - log_probs_old)  # [B, num_candidates]
   loss = - (ratio * advantages).mean()
   ```

**收益**

- 保证 reward 转 advantage 的逻辑只有一处，方便调参和 debug；
- 避免对 already Z‑scored reward 再做一次 Z‑score，削弱有效梯度。

---

### ST‑3：禁止 RL reward 静默 fallback 到“字符串匹配”

**文件位置**

- `lever_lm/models/v3/generate_rl_data.py`
  - `compute_vqa_accuracy(...)`
  - `generate_rl_data(...)` 内调用  
  - `main()` 中的参数处理

**当前 `compute_vqa_accuracy` 行为（简化）**

```python
def compute_vqa_accuracy(pred_answer, ground_truth_answers,
                         question_id=None, val_ques_path=None, val_ann_path=None):
    ...
    if val_ques_path and val_ann_path and question_id:
        try:
            accuracy = compute_vqa_accuracy_metric(temp_result_file, val_ques_path, val_ann_path)
            ...
            return correct, acc_score
        except Exception as e:
            print(f"警告：使用文件方式计算准确率失败，回退到简单匹配: {e}")

    # 简单匹配方式
    pred_answer_lower = pred_answer.lower().strip()
    gt_answers_lower = [ans.lower().strip() for ans in gt_answers_str]

    if pred_answer_lower in gt_answers_lower:
        return 1, 1.0
    for gt_ans in gt_answers_lower:
        if pred_answer_lower in gt_ans or gt_ans in pred_answer_lower:
            return 1, 0.5

    return 0, 0.0
```

**问题**

- 如果：
  - 没传 `val_ques_path/val_ann_path`，或
  - question_id 对不上，或
  - 官方评测脚本抛异常，
- 就会直接回退到 **简单字符串匹配**，即 `in / contains`，容易给错误答案正 reward；
- 这些 noisy reward 会直接喂给 RL，严重影响 RL 效果。

**修改目标**

- 生成 RL 数据时，要么：
  - 用官方 VQA metric，得到可靠的 correctness；
  - 要么直接报错，不生成这条数据；
- 不再“悄悄回退”到字符串匹配。

**推荐修改 A：在脚本入口强制要求评测文件**

在 `main()` 中加入：

```python
def main():
    ...
    # 对 VQA/OKVQA 场景强制要求传入评测文件
    if "vqa" in args.dataset.lower():
        if not args.val_ques_path or not args.val_ann_path:
            raise ValueError(
                "[generate_rl_data] For VQA RL data generation, "
                "you must provide --val_ques_path and --val_ann_path "
                "so that RL reward is computed by official VQA metric."
            )
    ...
```

**推荐修改 B：`compute_vqa_accuracy` 增加参数 `allow_fallback=False`**

```python
def compute_vqa_accuracy(
    pred_answer: str,
    ground_truth_answers,
    question_id: Optional[str] = None,
    val_ques_path: Optional[str] = None,
    val_ann_path: Optional[str] = None,
    allow_fallback: bool = False,
) -> Tuple[int, float]:
    ...
    if val_ques_path and val_ann_path and question_id:
        try:
            ...
            return correct, acc_score
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(
                    f"[compute_vqa_accuracy] VQA file-based accuracy failed "
                    f"for question_id={question_id}: {e}"
                )
            print(f"警告：使用文件方式计算准确率失败，回退到简单匹配: {e}")

    if not allow_fallback:
        # 缺失文件信息，而且不允许 fallback，直接报错
        raise RuntimeError(
            "[compute_vqa_accuracy] val_ques_path/val_ann_path/question_id "
            "not provided, and allow_fallback=False"
        )

    # 真的要 fallback 时才走下面的字符串匹配
    ...
```

在 `generate_rl_data(...)` 中调用时：

```python
correct, acc_score = compute_vqa_accuracy(
    pred_answer=pred_answer,
    ground_truth_answers=gt_answers,
    question_id=question_id_str,
    val_ques_path=val_ques_path,
    val_ann_path=val_ann_path,
    allow_fallback=False,      # 关键
)
```

**收益**

- RL reward 始终来自正式 VQA metric，而不是粗糙字符串匹配；
- 如果配置错误（路径 / question_id 等），会直接失败，而不是隐藏噪声。

---

### ST‑4：SFT 模型加载时打印结构与加载情况

**文件位置**

- `lever_lm/models/v3/generate_rl_data.py`
  - `load_sft_model(checkpoint_path, device)`

**当前实现（简化）**

```python
def load_sft_model(checkpoint_path: str, device: torch.device) -> PointerSelectorV3:
    model = PointerSelectorV3(
        d_model=512,  # 根据实际配置调整
        K=64,
        shot_num=2
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        state_dict = {k.replace('lever_lm.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
```

**问题**

- `d_model/K/shot_num` 写死，如果未来 checkpoint 的结构改变或你换了别的配置：
  - 因为 `strict=False`，不会抛错，只是 silent mismatch；
  - RL 采样用的模型和你以为的不完全一样。

**修改目标**

- 至少让 train log 能看到：
  - pointer 模型的结构；
  - 有哪些 key 没加载上，哪些是“unexpected”。

**推荐修改**

```python
def load_sft_model(checkpoint_path: str, device: torch.device) -> PointerSelectorV3:
    model = PointerSelectorV3(
        d_model=512,
        K=64,
        shot_num=2,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        state_dict = {k.replace("lever_lm.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"[load_sft_model] loaded checkpoint from {checkpoint_path}")
    print(f"[load_sft_model] PointerSelectorV3(d_model={model.d_model}, K={model.K}, shot_num={model.shot_num})")
    print(f"[load_sft_model] missing_keys: {missing_keys}")
    print(f"[load_sft_model] unexpected_keys: {unexpected_keys}")

    model.to(device)
    model.eval()
    return model
```

**收益**

- 后续如果你改了 pointer 架构或配置不一致，可以从 log 一眼看出；
- 避免“RL 数据是用错误结构的模型采出来”的情况。

---

### ST‑5：GRPO 配置与日志显式化（方便做 RCE‑only / 轻 GRPO 对比）

**文件位置**

- `lever_lm/workflows/grpo_post_train.py`  
  （或者你管理 GRPO 训练的主脚本）

**问题**

- 你已经在文档里做了多次实验（RCE‑only、新旧 v3 对比），但：
  - GRPO/RCE 的 epoch 数、LR、KL β 等参数是否在 log 里明显打印；
  - 长期回看实验结果时，很难快速对上“这轮实验具体跑的是什么配置”。

**修改目标**

- 把 GRPO/RCE 的关键配置（epoch/LR/KL β）写成结构体或清晰的配置块；
- 每轮训练前打印到 log。

**示例伪代码**

```python
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    rce_epochs: int = 5
    rce_lr: float = 1e-4
    grpo_epochs: int = 1
    grpo_lr: float = 5e-5
    kl_beta_init: float = 0.1
    kl_target_min: float = 0.01
    kl_target_max: float = 0.05

def train_with_grpo(cfg: GRPOConfig, ...):
    print(f"[GRPO] Config: {cfg}")

    # RCE stage
    for epoch in range(cfg.rce_epochs):
        print(f"[GRPO] Stage=RCE epoch={epoch+1}/{cfg.rce_epochs}, lr={rce_optimizer.param_groups[0]['lr']}")
        ...

    # GRPO stage
    kl_beta = cfg.kl_beta_init
    for epoch in range(cfg.grpo_epochs):
        print(f"[GRPO] Stage=GRPO epoch={epoch+1}/{cfg.grpo_epochs}, lr={grpo_optimizer.param_groups[0]['lr']}, kl_beta={kl_beta}")
        ...
```

**收益**

- 你在 `新强化学习结果.md` / `新方案评估.md` 里写的那些实验，可以 1:1 对上 log；
- 后面要做系统的 RCE‑only vs RCE+GRPO 对比时，复现更容易。

---

## 2. 中期修改（需要重新生成 RL 数据或做更多实验）

这些改动工程量会大一些，但都是在你现有 v3 框架基础上的增强，而不是推倒重来。

---

### MT‑1：多 shot 目标（同时优化 1/2/3/4‑shot）

> 原因：当前 RL reward 完全基于 2‑shot correctness，而你的最终指标看 1/2/3/4‑shot → 目标不对齐。

**设计思路**

- 将 pointer 模型改为输出固定 4 个 ICD（`shot_num=4`）；

- 对每条 pointer `[i1, i2, i3, i4]`：

  - 构建 1‑shot prompt（只用 `i1`）；
  - 2‑shot prompt（用 `i1 + i2`）；
  - 3‑shot prompt；
  - 4‑shot prompt；

- 分别跑 VQA，得到 `R1, R2, R3, R4`（每个都是 `hard + soft`）；

- 最终 RL reward 设置为：

  \[
  R_\text{total} = w_1 R_1 + w_2 R_2 + w_3 R_3 + w_4 R_4
  \]

  例如：`w1=0.2, w2=0.3, w3=0.2, w4=0.3`，可调。

**需要改动的模块**

1. **pointer 模型本身**

   - 文件：`lever_lm/models/v3/pointer_selector_v3.py`
   - 把 `shot_num` 从 2 改成 4，确保：
     - forward 输出 `[B, shot_num]`；
     - beam search / sample 函数都能输出长度为 4 的 pointer。

2. **RL pointer 采样**

   - 文件：`lever_lm/models/v3/rl_data_generation.py`
     - `generate_pointer_candidates_for_query(...)` 的 pointer 长度改为 4。

3. **RL 数据生成（compute 多 shot correctness）**

   - 文件：`lever_lm/models/v3/generate_rl_data.py`
     - 在 `generate_rl_data(...)` 中，对每条 pointer：

   **伪代码示例：**

   ```python
   pointer = c["pointer"]  # [i1, i2, i3, i4] (位置 index)
   original_pointer = [candidate_indices[p] for p in pointer]  # 映射回数据集中的 id
   
   R_list = []
   for k in [1, 2, 3, 4]:
       ex_list = [dataset[original_pointer[j]] for j in range(k)]  # 前 k 条 ICD
   
       pred_answer = build_vqa_prompt_and_generate(
           interface=vqa_model,
           image=image,
           question=question,
           ex_list=ex_list,    # 需要把 ex1/ex2 改成通用 ex_list 版本
           generation_kwargs=generation_kwargs,
       )
   
       correct, acc_score = compute_vqa_accuracy(
           pred_answer=pred_answer,
           ground_truth_answers=gt_answers,
           question_id=question_id_str,
           val_ques_path=val_ques_path,
           val_ann_path=val_ann_path,
           allow_fallback=False,
       )
   
       R_list.append(correct + acc_score)
   
   w = [0.2, 0.3, 0.2, 0.3]
   R_total = sum(w_i * R_i for w_i, R_i in zip(w, R_list))
   
   c["vqa_R_list"]  = R_list
   c["vqa_R_total"] = R_total
   ```

4. **RL Dataset 使用 `vqa_R_total`**

   - 文件：`lever_lm/models/v3/dataset_v3.py`
     - 在 `RLBeamDatasetWithEmbedding.__init__` 内：

   ```python
   for c in pointer_candidates:
       pointer = c["pointer"]
       ...
       if "vqa_R_total" in c:
           reward = float(c["vqa_R_total"])
       else:
           # 兼容旧格式：退回到 hard+soft（只有 2-shot）
           reward = compute_reward_for_candidate(
               vqa_correct=c.get("vqa_correct"),
               vqa_acc_score=c.get("vqa_acc_score"),
               reward_mode=self.reward_mode,
               hard_weight=self.hard_weight,
               soft_weight=self.soft_weight,
               ...
           )
       beam_rewards.append(reward)
   ```

**收益**

- RL reward 直接 encode 了你关心的 1/2/3/4‑shot 综合表现；
- 避免 2‑shot 变好但 1/3/4‑shot 被“牺牲”的情况。

---

### MT‑2：按 query 粒度过滤“无信号” RL 数据

**原因**

- 有些 query 的所有 pointer 都错（reward 全是 0/非常小）；
- 有些 query 的所有 pointer 得分都一样（没有排序意义）；
- 这些 query 在 GRPO 里基本没用，反而增加噪声。

**修改位置**

- `lever_lm/models/v3/dataset_v3.py`
  - `RLBeamDatasetWithEmbedding.__init__`

**推荐过滤逻辑（伪代码）**

```python
for query_id_str, query_data in rl_data.items():
    ...
    # 生成 beam_rewards 列表后
    if len(beam_rewards) == 0:
        continue

    rewards_tensor = torch.tensor(beam_rewards, dtype=torch.float32)
    r_min, r_max = rewards_tensor.min().item(), rewards_tensor.max().item()

    # 情况1: 全部 reward 一样 → 无排序信号
    if r_max == r_min:
        print(f"[RLBeamDataset] skip query_id={query_id} because all rewards equal={r_min:.4f}")
        continue

    # 情况2: reward 全部过小（例如都 < 0.1），几乎全错
    if r_max < 0.1:
        print(f"[RLBeamDataset] skip query_id={query_id} because rewards too small (max={r_max:.4f})")
        continue

    self.samples.append({
        "query_id": query_id,
        "beam_labels": beam_labels,
        "beam_rewards": beam_rewards,
        "beam_logprobs": beam_logprobs,
    })
```

**收益**

- GRPO 的梯度集中在真正“有 reward 差异”的 query 上；
- 训练更稳定，噪声更小。

---

### MT‑3：对 beam / sample / random 做“带权训练”

你已经设计了 `gen_method` 字段和 `filter_gen_methods` 参数。中期可以进一步：

- 不是简单地“只用 beam / 只用 beam+sample”，而是：
- 给不同 source 类型 (beam/sample/random) 赋予不同权重。

**实现思路**

1. **RL JSON：保持 `gen_method` 字段**  
   （你已实现）

2. **RL Dataset：把 `gen_method` 转成 numeric 权重**

   - 文件：`lever_lm/models/v3/dataset_v3.py`
     - `RLBeamDatasetWithEmbedding.__init__` 和 `__getitem__`：

   ```python
   method2weight = {"beam": 1.0, "sample": 0.7, "random": 0.4}
   
   self.samples.append({
       ...
       "beam_gen_methods": [c.get("gen_method", "beam") for c in pointer_candidates],
   })
   ```

   在 `__getitem__`：

   ```python
   beam_gen_methods = sample["beam_gen_methods"]
   method2weight = {"beam": 1.0, "sample": 0.7, "random": 0.4}
   beam_src_weights = torch.tensor(
       [method2weight.get(m, 0.5) for m in beam_gen_methods],
       dtype=torch.float32,
   )
   
   result = {
       ...
       "beam_src_weights": beam_src_weights,  # [num_candidates]
   }
   ```

3. **GRPO loss 中引入 source 权重**

   - 在 `grpo_post_train.py` 的 GRPO 计算中：

   ```python
   rewards = batch["beam_rewards"]          # [B, N]
   advantages = compute_group_relative_advantage(rewards, normalize=True, clip_range=5.0)
   src_w = batch["beam_src_weights"]        # [B, N]
   # 保证 src_w 形状可 broadcast
   while src_w.dim() < advantages.dim():
       src_w = src_w.unsqueeze(0)
   
   advantages = advantages * src_w
   
   ratio = torch.exp(log_probs_new - log_probs_old)
   loss = - (ratio * advantages).mean()
   ```

**收益**

- beam 通常质量更好，sample/random 有更多探索但也更 noisy；
- 通过 source 权重可以让 RL 更偏向稳定的 beam，同时保留探索能力。

---

### MT‑4：reward 形状微调 & KL 正则监控

在 `reward_utils.py` 中，你已经有：

- `clip_advantages(...)`
- `compute_kl_penalty(...)`
- `adaptive_kl_beta(...)`

**可以做的一些中期微调：**

1. **将 reward 做一次组内中心化（减去均值）后再 clip**

   在 GRPO 中：

   ```python
   rewards = batch["beam_rewards"]  # 原始 reward
   # 组内中心化
   centered = rewards - rewards.mean(dim=-1, keepdim=True)
   advantages = clip_advantages(centered, clip_range=2.0)
   ```

2. **收紧 KL 目标范围**

   调整 `adaptive_kl_beta` 的默认参数，例如：

   ```python
   kl_target_min = 0.01
   kl_target_max = 0.05
   adjustment_factor = 1.5
   ```

3. **周期性打印 KL & reward 统计**

   在 GRPO 训练循环中：

   ```python
   if global_step % 100 == 0:
       print(
           f"[GRPO] step={global_step} KL={kl.item():.4f} "
           f"kl_beta={kl_beta:.4f} reward_mean={rewards.mean().item():.4f} "
           f"reward_std={rewards.std().item():.4f}"
       )
   ```

**收益**

- 让 reward/advantage 的数量级更可控，有利于稳定 PPO；
- 通过 log 观察 RL 是否在“合理的 KL 区间”内更新。

---

## 3. 如何使用这份改造计划

一个可操作的执行顺序：

1. **先做短期修改（ST‑1～ST‑4）**
   - 确保索引没问题、reward 归一化链路清晰、RL reward 所有来源都可信；
   - 重新生成一版 RL 数据（至少一个 sampler + 一个模型）；
   - 跑一版 **RCE‑only**（GRPO_EPOCHS=0）：
     - 目标：2‑shot 至少不比 v2 差太多；  
     - 1/3/4‑shot 不出现明显恶化。
   - 在此基础上，加 1 个 epoch 的轻量 GRPO（小 LR、强 KL），观察是否带来微小提升。

2. **等短期方案稳定后，再开始做中期增强（MT‑2 / MT‑3）**
   - 先做 RL 数据质量过滤（按 query 粒度跳过无信号组）；
   - 再做 beam/sample/random 的带权训练；
   - 最后，如果你希望彻底对齐 1/2/3/4‑shot，再尝试 MT‑1 多 shot reward 设计（需要重跑 RL 数据，成本高一些）。

你可以直接把这份文档保存为：

> `docs/rl_todo_short_mid_term.md`

然后每搞定一个条目，就在相应小节后加上「✅ 已完成 / 对应实验编号」，形成一个可追踪的路线图。