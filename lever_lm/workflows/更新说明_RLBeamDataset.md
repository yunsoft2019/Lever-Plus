# 更新 grpo_post_train.py 以使用 RLBeamDatasetWithEmbedding

## 问题说明

### 当前状态
`grpo_post_train.py` 目前使用的是 `BeamDatasetWithEmbedding`，它只能加载**旧格式**的数据：

```json
{
  "query_id": {
    "id_list": [[shot1, shot2, query_id], ...],
    "score_list": [0.046, 0.045, ...]
  }
}
```

### 新需求
我们刚刚实现了 `RLBeamDatasetWithEmbedding`，它可以加载**新格式**的数据（包含 correctness）：

```json
{
  "query_id": {
    "pointer_candidates": [
      {
        "pointer": [7, 22],
        "gen_method": "beam",
        "beam_score": 2.13,
        "logprob_score": -1.56,
        "vqa_correct": 1,
        "vqa_acc_score": 1.0
      },
      ...
    ]
  }
}
```

## 需要修改的地方

### 1. 数据格式检测
需要自动检测数据是旧格式还是新格式：
- 旧格式：有 `id_list` 和 `score_list`
- 新格式：有 `pointer_candidates`

### 2. 数据集类选择
- 旧格式 → 使用 `BeamDatasetWithEmbedding`
- 新格式 → 使用 `RLBeamDatasetWithEmbedding`

### 3. Collate 函数
- 旧格式 → 使用 `collate_fn_v3`
- 新格式 → 使用 `collate_fn_rl_v3`

### 4. Batch Size
- 旧格式 → 可以使用任意 batch_size（如 32）
- 新格式 → **必须使用 batch_size=1**（每个batch是一个query-group）

### 5. 参数添加
新格式需要额外的 reward 参数：
- `reward_alpha`: quality权重（默认0.2）
- `reward_beta`: correctness权重（默认1.0）
- `reward_correctness_mode`: "01" 或 "pm1"（默认"pm1"）
- `use_logprob`: 是否使用 logprob_score（默认False）

### 6. 数据维度获取
- 旧格式：从 `id_list` 获取 `num_beams` 和 `shot_num`
- 新格式：从 `pointer_candidates[0]["pointer"]` 获取 `shot_num`，`num_beams` 不固定（取决于候选数量）

### 7. Candidate索引提取
- 旧格式：从 `id_list` 中提取所有ICD索引
- 新格式：从 `pointer_candidates` 中提取所有pointer中的索引

## 修改后的效果

更新后，`grpo_post_train.py` 将能够：
1. ✅ 自动检测数据格式
2. ✅ 支持旧格式（向后兼容）
3. ✅ 支持新格式（使用 correctness 信号）
4. ✅ 使用组合 reward（beam_score + correctness）
5. ✅ 支持多样化的数据生成方法（beam + sample + random）

## 使用示例

### 使用旧格式数据（向后兼容）
```bash
python -m lever_lm.workflows.grpo_post_train \
    --beam_data results/okvqa/generated_data/beam_RandSampler.json \
    --img_emb results/okvqa/cache/img_embeddings.pt \
    --batch_size 32  # 可以使用任意batch_size
```

### 使用新格式数据（包含correctness）
```bash
python -m lever_lm.workflows.grpo_post_train \
    --beam_data results/okvqa/generated_data/rl_data_RandSampler.json \
    --img_emb results/okvqa/cache/img_embeddings.pt \
    --batch_size 1  # 必须为1
    --reward_alpha 0.2 \
    --reward_beta 1.0
```
