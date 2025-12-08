# Lever-Plus v3 离线强化学习方案（基于当前仓库代码的完整改造说明）

> 说明：  
> 本文**只从当前 GitHub 仓库中的代码出发**，系统梳理：  
> - 现有 v3 相关代码的行为与数据流；  
> - 在这一阶段要做的所有改动；  
> - 每个改动背后的原因（之前会出现什么问题）；  
> - 需要改动的文件、关键代码位置、推荐的实现方式（含伪代码）。  
>
> 重点：  
> 1. 这一阶段 **所有 RL 数据（`rl_data_*.json`）都要重新生成**；  
> 2. 新生成的数据里只需要 **“整条 pointer 序列的软得分 + 硬得分”**（VQA correctness），**不再依赖增益 InfoScore**；  
> 3. 如需 InfoScore，后续可以在已有软得分基础上离线再算，而不是在 RL 数据生成阶段就强耦合。

---

## 1. 现有 v3 代码结构 & 数据流

先把当前仓库中跟 v3 / RL 相关的关键文件捋清楚，后面的修改都围绕这些文件展开。

### 1.1 脚本层

- `scripts/train_v3.sh`  
  - 完整 v3 流程入口：  
    1. 导出 embeddings（如果不存在）；  
    2. 生成 RL 数据（如果不存在）；  
    3. 调用 `lever_lm.workflows.grpo_post_train` 进行 RCE + GRPO 训练。  
  - 关键路径：
    - Embeddings：
      - `results/{dataset_name}/cache/query_embeddings.pt`
      - `results/{dataset_name}/cache/candidate_embeddings.pt`
    - RL 数据：
      - `results/{dataset_name}/generated_data/rl_data_{SamplerName}.json`
    - SFT v2 checkpoint：
      - `results/{dataset_name}/model_cpk/v2/{model_name}_{SamplerName}_infoscore_left_beam5_shot2_cand64_sample{sample_num}_best.ckpt`
- `scripts/generate_rl_data_for_sampler.sh`  
  - 专门用来生成某个 sampler 对应的 RL 数据：  
    - 从 `results/{dataset_name}/generated_data` 下读取 **束搜索 beam 数据 JSON**：  
      - 文件名示例：  
        `vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json`
    - 调用：
      ```bash
      python -m lever_lm.models.v3.generate_rl_data \
        --sft_ckpt "$sft_ckpt" \
        --beam_data "$beam_data_path" \
        --output_path "$rl_data_path" \
        --query_emb "$query_emb_path" \
        --cand_emb "$cand_emb_path" \
        --device "$device" \
        --vqa_model "$beam_model" \
        --dataset "$dataset"
      ```

> 结论：  
> **当前 RL 流程已经脚本化，但 RL 数据生成脚本只写了，还没有真正跑过一批“采样 + correctness”的数据。**  
> 本次改造的核心之一：**重新生成这一阶段所有 `rl_data_*.json`，用新定义的 reward 信号。**

---

### 1.2 v3 模型与数据集模块

- `lever_lm/models/v3/__init__.py`  
  暴露了 v3 相关所有组件：
  ```python
  from .pointer_selector_v3 import (
      PointerSelectorV3, PointerSelectorV3Config, build_model_v3
  )
  from .adapter_builder import (
      build_model_v3_with_adapter
  )
  from .dataset_v3 import (
      BeamDataset,
      BeamDatasetWithEmbedding,
      RLBeamDatasetWithEmbedding,
      collate_fn_v3,
      collate_fn_rl_v3,
      load_beam_data,
      split_beam_data,
  )
  from .inference_v3 import (
      load_v3_from_grpo_checkpoint,
      load_v3_from_sft_checkpoint,
      predict_with_v3
  )
  from .rl_data_generation import (
      compute_step_logits,
      sample_pointer_with_temperature,
      generate_pointer_candidates_for_query,
      evaluate_pointer_candidate,
  )
  ```

- `lever_lm/models/v3/pointer_selector_v3.py`  
  - 继承 v2 的双编码 + Cross-Attention 架构；
  - 新增：
    - `compute_rce_loss(...)`
    - `compute_grpo_loss(...)`
    - `compute_advantage(...)`
    - `compute_kl_divergence(...)`
  - 内部会调用 `lever_lm.utils.reward_utils` 里的一些工具函数（比如 group 内 advantage 计算）。

- `lever_lm/models/v3/dataset_v3.py`  
  - 定义了：
    - `BeamDataset`
    - `BeamDatasetWithEmbedding`
    - `RLBeamDatasetWithEmbedding`
    - `collate_fn_v3`, `collate_fn_rl_v3`
    - `load_beam_data`, `split_beam_data` 等。
  - **已明确区分了束搜索原始 JSON 的两种格式**：
    ```python
    # 旧格式：{query_id: {"id_list": [...], "score_list": [...]}}
    # 新格式：{query_id: {"pointer_candidates": [...]}}
    ```
  - `BeamDataset` / `BeamDatasetWithEmbedding` 主要服务于 v2 的 SFT 训练；
  - `RLBeamDatasetWithEmbedding` 专门服务于 v3 的 RL 训练，读取 `rl_data_*.json`。

> 当前 v3 训练脚本 `grpo_post_train.py` 的 DataLoader 是通过  
> `RLBeamDatasetWithEmbedding + collate_fn_rl_v3` 构建的；  
> **所以我们所有关于 reward 的修改，都必须让这个 dataset 输出正确的 `beam_rewards_raw` / `beam_rewards`。**

---

### 1.3 奖励工具与 RL 训练脚本

- `lever_lm/utils/reward_utils.py`  
  - 已经实现：
    - 组内 Z-score 归一化：`normalize_rewards_zscore`
    - 组内相对优势：`compute_group_relative_advantage`
    - RCE 用的 softmax 权重：`compute_softmax_weights`
    - KL 惩罚 & β 自适应调整：`compute_kl_penalty`, `adaptive_kl_beta`
    - **组合 reward 计算：`compute_reward_for_candidate(...)`**
      ```python
      def compute_reward_for_candidate(
          beam_score: Optional[float] = None,
          logprob_score: Optional[float] = None,
          vqa_correct: Optional[int] = None,
          vqa_acc_score: Optional[float] = None,
          alpha: float = 0.2,
          beta: float = 1.0,
          correctness_mode: str = "pm1",
          use_logprob: bool = False,
          reward_clip: Tuple[float, float] = (-5.0, 5.0),
      ) -> float:
          # correctness_val: 0/1 or 映射到 [-1,1]
          # quality: beam_score 或 logprob_score
          # reward = alpha * quality + beta * correctness_val
      ```
  - 可以确定：  
    当前 `RLBeamDatasetWithEmbedding` 在构造 `beam_rewards_raw` 时，是预期要用这个函数，组合：
    - beam_score / logprob_score（来自 pointer 选择器的分布）
    - correctness（来自下游 VQA 正确性）

- `lever_lm/workflows/grpo_post_train.py`  
  - 核心类：`GRPOTrainer`，包括：
    - `train_rce_epoch(...)`
      - 从 batch 中取：
        - `query_emb`, `cand_emb`
        - `beam_labels`
        - `beam_rewards_raw`
      - 调：
        ```python
        loss = self.model.compute_rce_loss(
            query_emb, cand_emb,
            beam_labels,
            beam_rewards_raw,
            temperature=temperature,
            use_top1_only=self.use_top1_only,
        )
        ```
    - `train_grpo_epoch(...)`
      - 从 batch 中取：
        - `beam_labels`
        - `beam_rewards`
        - `query_id` → 用于索引 `old_log_probs_dict[qid]`
      - 调：
        ```python
        result = self.model.compute_grpo_loss(
            query_emb, cand_emb,
            beam_labels,
            beam_rewards,
            old_log_probs,
            use_top_k=use_top_k,
        )
        ```
  - 结论：
    - **RCE 用的是 `beam_rewards_raw`**（原始 reward）；
    - **GRPO 用的是 `beam_rewards`**（可以是原始或提前归一化的 reward）；
    - 这两个字段都由 `RLBeamDatasetWithEmbedding` 负责构造。

---

## 2. 当前阶段的目标 & 设计原则

### 2.1 目标

1. **重新生成 RL 数据**（`rl_data_*.json`）：
   - 使用已经训练好的 v2 / v2_lora Pointer Selector；
   - 对每个 query：
     - 使用 Pointer Selector v2 做 beam + 温度采样 + 少量随机；
     - 对每条 pointer 序列调用 VQA 模型：
       - 得到 **整条序列的硬得分**：`vqa_correct ∈ {0,1}`；
       - 得到 **整条序列的软得分**：`vqa_acc_score ∈ [0,1]`（VQAv2 soft acc）。
   - 当前阶段 **不需要计算增益 InfoScore**；
     - 如果将来需要，可以在 `vqa_acc_score` 的基础上离线再算（例如相对于 0-shot 或 1-shot baseline）。

2. **重新定义 RL reward**（进入 Pointer Selector v3 的 reward）：
   - 只基于：
     - `vqa_acc_score`（软正确率）；  
     - `vqa_correct`（硬正确率，0/1，或从软正确率阈值化得到）。
   - 示例设计：
     - 硬得分：  
       `hard = 1.0`（答对）或 `0.0`（答错）；  
     - 软得分：  
       `soft = vqa_acc_score ∈ [0,1]`；
     - 最终 reward（示例）：
       \[
       r = hard + soft
       \]
       - 正样本：最小 `r = 1.0`，最大 `r = 2.0`；  
       - 负样本：`r ∈ [0.0, 1.0)`；  
       - 可以区分“对 vs 错”，也能区分“更好 vs 一般”。

   > 你如果之后要引入“模型认为答案正确的概率 p_correct（例如 `get_cond_prob`）”，可以直接把软得分替换为 p_correct 或设为 `soft = 0.5 * acc_score + 0.5 * p_correct`，但这一步可以放到后面。

3. **让所有 RL 相关代码（dataset + reward_utils + PointerSelectorV3）完全基于上述 reward 工作**，跟 InfoScore 脱钩。

---

## 3. RL 数据生成：需要修改的文件与逻辑

这一部分是本轮改造的核心：**重新生成 RL 数据**，确保里面已有“硬得分 + 软得分”。

### 3.1 `scripts/generate_rl_data_for_sampler.sh`

当前脚本已经正确地指向了 RL 数据生成模块：

```bash
python -m lever_lm.models.v3.generate_rl_data \
  --sft_ckpt "$sft_ckpt" \
  --beam_data "$beam_data_path" \
  --output_path "$rl_data_path" \
  --query_emb "$query_emb_path" \
  --cand_emb "$cand_emb_path" \
  --device "$device" \
  --vqa_model "$beam_model" \
  --dataset "$dataset"
```

> 不需要改脚本逻辑，只需要确保：  
> - 之前生成的 `rl_data_*.json` 全部删除或更名；  
> - 本轮改造完成后，通过脚本重新生成。

### 3.2 `lever_lm/models/v3/rl_data_generation.py`

当前已经实现了三件事：

1. **Beam Search + 温度采样 + 随机 pointer 生成**

   - `beam_search_pointer(...)`  
   - `sample_pointer_with_temperature(...)`  
   - `generate_pointer_candidates_for_query(...)`

2. **评估 pointer 的 correctness**

   - `evaluate_pointer_candidate(...)`：
     ```python
     def evaluate_pointer_candidate(
         vqa_model,
         image,
         question: str,
         candidate_pool: List[Dict],
         pointer: List[int],
         ground_truth_answers: List[str],
         build_vqa_prompt_fn: callable,
         compute_vqa_accuracy_fn: callable,
     ) -> Tuple[str, int, float]:
         # ex1, ex2 = candidate_pool[pointer[0]], candidate_pool[pointer[1]]
         # prompt = build_vqa_prompt_fn(image, question, ex1, ex2)
         # pred_answer = vqa_model.generate(prompt)
         # correct, acc_score = compute_vqa_accuracy_fn(pred_answer, ground_truth_answers)
         # return pred_answer, int(correct), float(acc_score)
     ```

> 结论：  
> - 这一模块已经 **能产出“硬得分 + 软得分”**：  
>   - `correct`（0/1）  
>   - `acc_score`（[0,1]）  
> - 不需要 InfoScore，不需要做改动，只要在 RL 数据生成脚本中把这两个字段写入 JSON 即可。

### 3.3 `lever_lm/models/v3/generate_rl_data.py`

这是实际生成 `rl_data_*.json` 的脚本，当前行为大致是：

1. 加载 SFT 模型（v2 checkpoint） → `PointerSelectorV3` 兼容加载。
2. 加载 VQA 模型 Interface → `load_vqa_model(...)`。
3. 加载数据集 → `load_ds(...)`（来自根目录 `utils.py`）。
4. 加载 embeddings：
   - `query_embeddings.pt`
   - `candidate_embeddings.pt`
5. 遍历每个 query：
   - 用 pointer SFT 模型：
     - `generate_pointer_candidates_for_query(...)` 生成：
       ```python
       {
         "pointer": [i, j],
         "gen_method": "beam" / "sample" / "random",
         "beam_rank": ...,
         "beam_score": ...,
         "logprob_score": ...,
         "temperature": ...
       }
       ```
   - 用 VQA 模型：
     - `evaluate_pointer_candidate(...)` 计算：
       ```python
       pred_answer, correct, acc_score
       ```

**需要你确认 / 完成的部分：**

> 你在前一个阶段其实只是“写了这个脚本，但没有真实跑过 v2 采样 + VQA”的生成过程。  
> 现在，我们需要在这个脚本里 **显式地把 `correct` / `acc_score` 写入 RL JSON**，并确保不依赖 InfoScore。

#### 3.3.1 建议的 JSON 结构（`rl_data_*.json`）

采用按 query_id 分组的结构：

```jsonc
{
  "12345": {
    "meta": {
      "image_id": "COCO_000001",
      "question": "What is the man holding?",
      "split": "train"
    },
    "candidate_pool": [
      // 和 beam_data 中一致的候选列表（便于 debug）
    ],
    "pointer_candidates": [
      {
        "pointer": [7, 22],
        "gen_method": "beam",
        "beam_rank": 0,
        "beam_score": 2.13,
        "logprob_score": -1.56,
        "temperature": null,
        "vqa_pred_answer": "a book",
        "vqa_correct": 1,
        "vqa_acc_score": 1.0
      },
      {
        "pointer": [5, 18],
        "gen_method": "sample",
        "beam_rank": null,
        "beam_score": null,
        "logprob_score": -2.03,
        "temperature": 1.0,
        "vqa_pred_answer": "a notebook",
        "vqa_correct": 1,
        "vqa_acc_score": 0.9
      },
      {
        "pointer": [3, 9],
        "gen_method": "random",
        "beam_rank": null,
        "beam_score": null,
        "logprob_score": null,
        "temperature": null,
        "vqa_pred_answer": "a phone",
        "vqa_correct": 0,
        "vqa_acc_score": 0.0
      }
    ]
  },
  "12346": {
    "...": "..."
  }
}
```

> 和 `dataset_v3.load_beam_data` 中注释的新格式完全一致：  
> `{query_id: {"pointer_candidates": [...]}}`。  
> 后续 `RLBeamDatasetWithEmbedding` 的实现就是基于这个结构。

#### 3.3.2 具体改动示例（伪代码）

在 `generate_rl_data.py` 的 `main()` 中（或等价函数），大致结构应该是这样（伪代码）：

```python
def generate_rl_data(
    sft_ckpt: str,
    beam_data_path: str,
    output_path: str,
    query_emb_path: str,
    cand_emb_path: str,
    vqa_model_name: str,
    dataset_name: str,
    device: str = "cuda:0",
    num_beams: int = 5,
    temps: Tuple[float, ...] = (1.0, 1.3),
    num_samples_per_temp: int = 2,
    num_random: int = 1,
):
    device = torch.device(device)

    # 1) 加载 pointer SFT 模型（v2 checkpoint）
    pointer_model = load_sft_model(sft_ckpt, device=device)

    # 2) 加载 VQA 模型 interface
    vqa_interface = load_vqa_model(vqa_model_name, device=device)

    # 3) 加载数据集（包含图像 / 问题 / 答案）
    ds = load_ds_for_vqa(dataset_name)  # 实际用 root_utils.load_ds

    # 4) 加载 embedding
    query_emb_dict = torch.load(query_emb_path, map_location=device)
    cand_emb_dict = torch.load(cand_emb_path, map_location=device)

    # 5) 加载 beam_data（用来获得 candidate_pool 与 query_id 对应关系）
    beam_data = load_beam_data(beam_data_path)  # 注意：使用 v3.dataset_v3.load_beam_data

    rl_data = {}

    for query_id, info in tqdm(beam_data.items()):
        # 5.1 取得 query 的 embedding
        query_emb = query_emb_dict[int(query_id)]      # [d]
        cand_emb = cand_emb_dict[int(query_id)]        # [K, d]

        # 5.2 恢复 candidate_pool（可以直接从 beam_data / 原始数据集中取）
        candidate_pool = build_candidate_pool_for_query(ds, info)

        # 5.3 生成 pointer 候选（beam + 温度采样 + 随机）
        pointer_candidates = generate_pointer_candidates_for_query(
            model=pointer_model,
            query_emb=query_emb,
            cand_emb=cand_emb,
            num_beams=num_beams,
            temps=temps,
            num_samples_per_temp=num_samples_per_temp,
            num_random=num_random,
        )

        # 5.4 对每个 pointer 候选计算 correctness
        gt_answers = get_ground_truth_answers(ds, query_id)

        enriched_candidates = []
        for pc in pointer_candidates:
            pointer = pc["pointer"]
            pred_answer, correct, acc_score = evaluate_pointer_candidate(
                vqa_model=vqa_interface,
                image=get_image_for_query(ds, query_id),
                question=get_question_for_query(ds, query_id),
                candidate_pool=candidate_pool,
                pointer=pointer,
                ground_truth_answers=gt_answers,
                build_vqa_prompt_fn=build_vqa_prompt_fn,
                compute_vqa_accuracy_fn=compute_vqa_accuracy_metric,
            )
            enriched_candidates.append({
                **pc,
                "vqa_pred_answer": pred_answer,
                "vqa_correct": int(correct),
                "vqa_acc_score": float(acc_score),
            })

        # 5.5 写入 rl_data 字典
        rl_data[str(query_id)] = {
            "meta": extract_meta_from_dataset(ds, query_id),
            "candidate_pool": candidate_pool,  # 可选，便于调试
            "pointer_candidates": enriched_candidates,
        }

    # 6) 保存到 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rl_data, f, ensure_ascii=False)
```

> 实际实现时，只需要在你现有的 `generate_rl_data.py` 中补全/核实：  
> - `enriched_candidates` 是否包含 `vqa_correct` / `vqa_acc_score` 字段；  
> - 顶层结构是否是 `{query_id: {"pointer_candidates": [...]}}`；  
> - 不再依赖 InfoScore 作为 reward，只作为 beam_data 的一部分存在。

---

## 4. RL 训练数据集：`dataset_v3.RLBeamDatasetWithEmbedding`

### 4.1 当前设计目标

`RLBeamDatasetWithEmbedding` 的职责是：

- 从 `rl_data_*.json` 中读取数据；
- 对每个 **query** 输出一条样本，包括：
  - `query_emb`: `[d]`
  - `cand_emb`: `[K, d]`
  - `beam_labels`: `[num_beams, shot_num]`（每条 pointer）  
  - `beam_rewards_raw`: `[num_beams]`（原始 reward）
  - `beam_rewards`: `[num_beams]`（供 GRPO 使用，可以等于 raw 或做简单处理）
  - `query_id`: 用于索引 `old_log_probs_dict`

然后 `collate_fn_rl_v3` 会把这些拼成 batch，给 `GRPOTrainer` 使用。

### 4.2 需要确保的行为

1. **完全基于 RL JSON 中的 `vqa_correct` / `vqa_acc_score` 构造 reward**  
   - 不再从 InfoScore / beam score 中推导 reward；  
   - beam_score / logprob_score 可以保留在 JSON 中用于分析，但不进入 reward 公式。

2. **保持 RCE / GRPO 期望的字段命名**  
   - `GRPOTrainer.train_rce_epoch` 期望 batch 中有：
     - `beam_labels`
     - `beam_rewards_raw`
   - `GRPOTrainer.train_grpo_epoch` 期望 batch 中有：
     - `beam_labels`
     - `beam_rewards`

3. **reward 公式（示例）**

   在这一阶段，可以采用简单而清晰的公式：

   ```python
   # 硬得分：0/1 correctness
   hard = float(vqa_correct)              # 1 or 0

   # 软得分：VQAv2 soft accuracy
   soft = float(vqa_acc_score)           # [0,1]

   # 最终 reward：
   reward = hard + soft                  # ∈ [0, 2]
   ```

   - 正样本：
     - 最小 `1.0`，最大 `2.0`；
   - 负样本：
     - `0.0 ~ <1.0`；
   - 正 vs 负 自动分离；  
   - 正样本内部：`soft` 还能拉开“更好 vs 一般”的差距；  
   - 负样本内部：`soft` 可以反映“接近正确 vs 完全错”。

### 4.3 推荐改法（伪代码）

在 `lever_lm/models/v3/dataset_v3.py` 中找到 `RLBeamDatasetWithEmbedding`，理想的实现逻辑类似：

```python
class RLBeamDatasetWithEmbedding(Dataset):
    def __init__(
        self,
        rl_data: Dict,                # 从 rl_data_*.json 加载
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_indices: List[int],
        shot_num: int = 2,
        max_beams: Optional[int] = None,
        reward_mode: str = "hard_plus_soft",
    ):
        self.rl_data = rl_data
        self.query_embeddings = query_embeddings
        self.candidate_embeddings = candidate_embeddings
        self.candidate_indices = candidate_indices
        self.shot_num = shot_num
        self.max_beams = max_beams
        self.reward_mode = reward_mode

        # 为了方便，将所有 query_id 排序，__len__ 和 __getitem__ 通过 index 做映射
        self.query_ids = sorted(rl_data.keys(), key=int)

    def __len__(self):
        return len(self.query_ids)

    def _build_reward(self, cand_rec: Dict) -> float:
        vqa_correct = cand_rec.get("vqa_correct", 0)
        vqa_acc = cand_rec.get("vqa_acc_score", 0.0)

        hard = float(vqa_correct)
        soft = float(vqa_acc)

        if self.reward_mode == "hard_only":
            return hard
        elif self.reward_mode == "soft_only":
            return soft
        elif self.reward_mode == "hard_plus_soft":
            return hard + soft
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def __getitem__(self, idx: int) -> Dict:
        query_id = self.query_ids[idx]
        rec = self.rl_data[query_id]

        # 1) 取 embedding
        qid_int = int(query_id)
        query_emb = self.query_embeddings[qid_int]          # [d]
        cand_emb_all = self.candidate_embeddings            # [N, d] 或已按候选池裁剪

        # 2) pointer_candidates 列表
        cand_list = rec["pointer_candidates"]               # List[Dict]

        if self.max_beams is not None:
            cand_list = cand_list[: self.max_beams]

        # 3) 构造 beam_labels 与 rewards
        beam_labels = []
        rewards_raw = []

        for c in cand_list:
            pointer = c["pointer"]                          # [shot_num]
            reward = self._build_reward(c)

            beam_labels.append(pointer)
            rewards_raw.append(reward)

        beam_labels = torch.tensor(beam_labels, dtype=torch.long)        # [num_beams, shot_num]
        rewards_raw = torch.tensor(rewards_raw, dtype=torch.float32)     # [num_beams]

        # 4) cand_emb：只取当前 query 的候选池 embedding
        #    具体实现取决于你如何构建 candidate_embeddings，常见做法是预先裁剪为 [num_queries, K, d]
        cand_emb = cand_emb_all[qid_int]                                 # [K, d]

        # 5) beam_rewards：此处可以直接用 raw，归一化留给 PointerSelectorV3 内部
        beam_rewards = rewards_raw.clone()

        return {
            "query_id": qid_int,
            "query_emb": query_emb,            # [d]
            "cand_emb": cand_emb,              # [K, d]
            "beam_labels": beam_labels,        # [num_beams, shot_num]
            "beam_rewards_raw": rewards_raw,   # [num_beams] - RCE 用
            "beam_rewards": beam_rewards,      # [num_beams] - GRPO 用
        }
```

> 说明：  
> - 这里 **没有再调用 `reward_utils.compute_reward_for_candidate`**，而是在 dataset 内部直接用 `vqa_correct` / `vqa_acc_score` 构造 reward；  
> - 如果你想复用 `reward_utils.compute_reward_for_candidate`，可以在 `_build_reward` 里转调，并在该函数中改公式。

---

## 5. 奖励工具：`lever_lm/utils/reward_utils.py` 的改动建议

目前的 `compute_reward_for_candidate` 主要逻辑是：

```python
# correctness_val: 0/1 或 [-1, 1]
# quality: beam_score 或 logprob_score
reward = alpha * quality + beta * correctness_val
reward = clip(reward, reward_clip)
```

这是假定“InfoScore / beam_score”作为 quality 的组合方式。  
在现在的阶段，我们**不打算再用 InfoScore / beam_score 作为 RL reward 的主要来源**，推荐改成一个更简单、且和 RL 数据结构强绑定的版本。

### 5.1 改造建议：加入 soft/hard correctness，并默认不使用 beam_score

把函数改造为支持新的 soft/hard 信号（注意：下面是**示例实现**）：

```python
def compute_reward_for_candidate(
    beam_score: Optional[float] = None,
    logprob_score: Optional[float] = None,
    vqa_correct: Optional[int] = None,
    vqa_acc_score: Optional[float] = None,
    # 新增：soft/hard 组合参数
    use_hard: bool = True,
    use_soft: bool = True,
    hard_weight: float = 1.0,
    soft_weight: float = 1.0,
    # 兼容旧接口的参数（可以保留但默认不启用）
    alpha: float = 0.0,                      # 默认不再依赖 beam_score/logprob
    beta: float = 0.0,                       # 默认不使用旧 correctness_mode
    correctness_mode: str = "01",           # 保守修改
    use_logprob: bool = False,
    reward_clip: Tuple[float, float] = (-5.0, 5.0),
) -> float:
    """
    计算 RL reward，当前阶段的推荐用法：
    - 只使用 vqa_correct / vqa_acc_score 构造 reward
      reward = hard_weight * hard + soft_weight * soft
    - 如需 beam_score/logprob_score，可以单独在实验时打开 alpha/beta
    """

    # 1) hard correctness
    hard = 0.0
    if use_hard and vqa_correct is not None:
        hard = float(vqa_correct)  # 0 or 1

    # 2) soft correctness
    soft = 0.0
    if use_soft and vqa_acc_score is not None:
        soft = float(vqa_acc_score)  # [0,1]

    reward = hard_weight * hard + soft_weight * soft

    # （可选）叠加旧的 quality + correctness_val 逻辑，
    # 但在当前阶段默认 alpha=beta=0，即不生效
    if beta != 0.0:
        if correctness_mode == "01":
            correctness_val = float(vqa_correct) if vqa_correct is not None else float(vqa_acc_score or 0.0)
        else:
            if vqa_correct is not None:
                correctness_val = 2.0 * float(vqa_correct) - 1.0
            else:
                correctness_val = 2.0 * float(vqa_acc_score or 0.0) - 1.0
        reward += beta * correctness_val

    if alpha != 0.0:
        if use_logprob and logprob_score is not None:
            quality = -float(logprob_score)
        elif beam_score is not None:
            quality = float(beam_score)
        else:
            quality = 0.0
        reward += alpha * quality

    # 裁剪
    reward = max(reward_clip[0], min(reward_clip[1], reward))
    return reward
```

> 注意：  
> - 这段改造是**为未来可能继续用 `compute_reward_for_candidate` 提供通用接口**；  
> - 如果你在 `RLBeamDatasetWithEmbedding` 中直接构造了 reward，不一定非要调用这个函数。

---

## 6. PointerSelectorV3：如何使用新的 reward

PointerSelectorV3 内部的 RL 逻辑大概是：

- `compute_rce_loss(...)`：
  - 输入：
    - `beam_labels`: `[B, num_beams, shot_num]`
    - `beam_rewards`: `[B, num_beams]`（在 `GRPOTrainer.train_rce_epoch` 中传入的是 `beam_rewards_raw`）
  - 步骤概念上类似：
    ```python
    # 1) 根据奖励算权重：w_i = softmax(reward_i / τ)
    weights = compute_softmax_weights(beam_rewards_raw, temperature)

    # 2) 对每个 beam 计算 CE loss
    ce_loss_per_beam = ...  # 逐条 pointer 的 cross-entropy

    # 3) 加权平均
    loss = (weights * ce_loss_per_beam).sum(dim=-1).mean()
    ```

- `compute_grpo_loss(...)`：
  - 输入：
    - `beam_rewards`: `[B, num_beams]`
    - `old_log_probs`: `[B, num_beams]`
  - 内部会做：
    1. 计算组内 advantage（通常用 `compute_group_relative_advantage`）；
    2. 计算 `ratio = exp(new_log_probs - old_log_probs)`；
    3. 计算 PPO 损失 + KL。

**只要我们在 `RLBeamDatasetWithEmbedding` 中把 reward 改成 “hard + soft correctness”，PointerSelectorV3 内部的逻辑不用动。**

---

## 7. 实际执行步骤（从头跑一遍）

最后整合成一套可执行的 pipeline，方便你落地：

1. **清理旧的 RL 数据（可选但强烈建议）**
   ```bash
   rm results/okvqa/generated_data/rl_data_*.json
   rm results/vqav2/generated_data/rl_data_*.json
   ```

2. **确保 v2 / v2_lora 已经训练完成**（已有 checkpoint）  
   - 检查：
     ```bash
     ls results/okvqa/model_cpk/v2/
     ```

3. **确保 Embeddings 已存在**
   - 可以直接依赖 `train_v3.sh` 的 Step 0，或者单独执行 `scripts/export_embeddings.sh`；

4. **修改完成下面几个文件：**
   - `lever_lm/models/v3/generate_rl_data.py`  
     - 确保 `vqa_correct` / `vqa_acc_score` 被写入 RL JSON；  
     - JSON 格式为 `{query_id: {"pointer_candidates": [...]}}`。
   - `lever_lm/models/v3/dataset_v3.py`  
     - 在 `RLBeamDatasetWithEmbedding` 中，读取 `vqa_correct` / `vqa_acc_score` 并构造 reward；  
     - 输出 `beam_rewards_raw` / `beam_rewards`。
   - （可选）`lever_lm/utils/reward_utils.py`  
     - 如果你在 dataset 中想复用 `compute_reward_for_candidate`，按上面示例改造接口与公式。

5. **重新生成 RL 数据**
   ```bash
   # 例：OKVQA + RandSampler + Qwen2.5-VL-3B
   bash scripts/generate_rl_data_for_sampler.sh rand_sampler qwen2.5_vl_3B okvqa_local cuda:0
   ```

6. **启动 v3 训练**
   ```bash
   bash scripts/train_v3.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
   ```

---

## 8. 小结（从“现有代码”到“可用 RL”的所有改动点）

- **数据生成层（generate_rl_data.py + rl_data_generation.py）**
  - 使用 Pointer v2 模型做 beam + 温度采样 + 随机；
  - 对每条 pointer 序列，用 VQA 模型跑整条序列的：
    - 硬得分：`vqa_correct ∈ {0,1}`；
    - 软得分：`vqa_acc_score ∈ [0,1]`；
  - 写入 `rl_data_*.json`，结构为 `{query_id: {"pointer_candidates": [...]}}`；
  - **不再在这个阶段计算增益得分 InfoScore**。

- **数据集层（RLBeamDatasetWithEmbedding）**
  - 从 `rl_data_*.json` 中组装：
    - `beam_labels`（pointer 序列）
    - `beam_rewards_raw` / `beam_rewards`；
  - reward 使用简单直观的公式：
    \[
    r = \text{hard}(vqa\_correct) + \text{soft}(vqa\_acc\_score)
    \]
  - 保证 batch 字段名与 `GRPOTrainer` 一致。

- **奖励工具层（reward_utils.py）**
  - `compute_reward_for_candidate` 可改为优先支持 “hard + soft correctness” 的组合形式，  
    兼容保留 beam_score / logprob_score 的选项。

- **模型与训练层（PointerSelectorV3 + GRPOTrainer）**
  - 不需要大改，只要仍然接收：
    - RCE：`beam_rewards_raw`
    - GRPO：`beam_rewards`
  - 内部用 group 内 softmax / advantage + PPO + KL 即可。

---

这份文档可以直接保存为：

> `docs/v3_rl_from_current_code_full_plan.md`

放在仓库里给自己和合作者阅读，实现时就对照文件中的每一节逐条改，就不会混淆“旧 InfoScore 逻辑”和“当前阶段只用整条序列 correctness 的 RL 逻辑”。