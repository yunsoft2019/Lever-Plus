# RL 数据生成脚本使用说明

## 命令格式

```bash
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt <SFT模型路径> \
    --beam_data <beam数据路径> \
    --output_path <输出RL数据路径> \
    --query_emb <query_embeddings.pt> \
    --cand_emb <candidate_embeddings.pt> \
    [其他可选参数...]
```

## 必需参数

### 1. `--sft_ckpt` (必需)
- **说明**: SFT模型（v2）的checkpoint路径
- **示例**: `results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_xxx.ckpt`

### 2. `--beam_data` (必需)
- **说明**: 现有的beam数据JSON文件路径
- **示例**: `results/okvqa/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-...json`

### 3. `--output_path` (必需)
- **说明**: 输出的RL数据JSON文件路径
- **示例**: `results/okvqa/generated_data/rl_data_RandSampler.json`

### 4. `--query_emb` (必需)
- **说明**: Query embeddings文件路径（.pt格式）
- **示例**: `results/okvqa/cache/query_embeddings.pt`

### 5. `--cand_emb` (必需)
- **说明**: Candidate embeddings文件路径（.pt格式）
- **示例**: `results/okvqa/cache/candidate_embeddings.pt`

## 可选参数

### 6. `--vqa_model` (可选，默认: `qwen2.5_vl_3B`)
- **说明**: VQA模型名称，用于评估correctness
- **可选值**: `qwen2.5_vl_3B`, `flamingo_3B` 等
- **示例**: `--vqa_model qwen2.5_vl_3B`

### 7. `--dataset` (可选，默认: `okvqa_local`)
- **说明**: 数据集名称
- **示例**: `--dataset okvqa_local`

### 8. `--num_beams` (可选，默认: `5`)
- **说明**: Beam搜索的beam数量
- **示例**: `--num_beams 5`

### 9. `--temps` (可选，默认: `1.0 1.3`)
- **说明**: 温度采样列表（可以多个）
- **示例**: `--temps 1.0 1.3` 或 `--temps 0.7 1.0 1.3`

### 10. `--num_samples_per_temp` (可选，默认: `2`)
- **说明**: 每个温度采样的数量
- **示例**: `--num_samples_per_temp 2`

### 11. `--num_random` (可选，默认: `1`)
- **说明**: 随机组合的数量
- **示例**: `--num_random 1`

### 12. `--device` (可选，默认: `cuda:0`)
- **说明**: 使用的设备
- **示例**: `--device cuda:0` 或 `--device cuda:1`

### 13. `--val_ques_path` (可选)
- **说明**: 验证集问题文件路径（用于准确率计算）
- **示例**: `--val_ques_path data/okvqa/val_questions.json`

### 14. `--val_ann_path` (可选)
- **说明**: 验证集标注文件路径（用于准确率计算）
- **示例**: `--val_ann_path data/okvqa/val_annotations.json`

### 15. `--config` (可选)
- **说明**: Hydra配置文件路径（用于加载数据集和VQA模型配置）
- **示例**: `--config configs/train/query_img_text_icd_img_text_v2.yaml`

## 完整示例

### 示例1：基本使用（最小参数）

```bash
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800.ckpt \
    --beam_data results/okvqa/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json \
    --output_path results/okvqa/generated_data/rl_data_RandSampler.json \
    --query_emb results/okvqa/cache/query_embeddings.pt \
    --cand_emb results/okvqa/cache/candidate_embeddings.pt \
    --device cuda:0
```

### 示例2：完整参数（推荐）

```bash
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_xxx.ckpt \
    --beam_data results/okvqa/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-...json \
    --output_path results/okvqa/generated_data/rl_data_RandSampler.json \
    --query_emb results/okvqa/cache/query_embeddings.pt \
    --cand_emb results/okvqa/cache/candidate_embeddings.pt \
    --vqa_model qwen2.5_vl_3B \
    --dataset okvqa_local \
    --num_beams 5 \
    --temps 1.0 1.3 \
    --num_samples_per_temp 2 \
    --num_random 1 \
    --device cuda:0
```

### 示例3：使用验证集文件（更准确的correctness）

```bash
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/xxx.ckpt \
    --beam_data results/okvqa/generated_data/beam_RandSampler.json \
    --output_path results/okvqa/generated_data/rl_data_RandSampler.json \
    --query_emb results/okvqa/cache/query_embeddings.pt \
    --cand_emb results/okvqa/cache/candidate_embeddings.pt \
    --val_ques_path data/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --val_ann_path data/okvqa/mscoco_val2014_annotations.json \
    --device cuda:0
```

## 输出说明

脚本会生成一个JSON文件，格式如下：

```json
{
  "query_id": {
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
      ...
    ]
  }
}
```

## 注意事项

1. **计算成本**：对每个pointer候选都会调用VQA模型，计算成本较高，建议：
   - 使用GPU加速
   - 可以先在小数据集上测试
   - 批量处理

2. **Embedding文件**：确保 `query_emb` 和 `cand_emb` 文件存在且格式正确

3. **Beam数据格式**：确保beam数据文件格式正确（包含 `id_list` 和 `score_list`）

4. **验证集文件**：如果提供 `val_ques_path` 和 `val_ann_path`，会使用标准VQA评估方式计算correctness，更准确

5. **生成时间**：根据数据量，生成RL数据可能需要较长时间（几分钟到几小时）

## 常见问题

**Q: 如果缺少embedding文件怎么办？**  
A: 需要先导出embeddings。可以使用项目中的embedding导出脚本。

**Q: 可以只生成部分数据吗？**  
A: 可以修改脚本，或者先在小数据集上测试。

**Q: 生成的RL数据可以直接用于训练吗？**  
A: 是的，生成后可以直接用于 `grpo_post_train.py` 训练。
