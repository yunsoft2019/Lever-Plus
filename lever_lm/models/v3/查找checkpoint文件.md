# 如何找到正确的 checkpoint 文件路径

## 问题
`xxx.ckpt` 是示例占位符，需要替换为实际的 checkpoint 文件路径。

## 方法1：手动查找

### Step 1: 确定文件位置
根据项目结构，checkpoint 文件通常保存在：
```
results/{dataset}/model_cpk/{version}/
```

对于你的情况：
```
results/okvqa/model_cpk/v2/
```

### Step 2: 列出所有文件
```bash
ls -lh results/okvqa/model_cpk/v2/*.ckpt
```

### Step 3: 查找匹配的文件
对于 RandSampler + Qwen2.5-VL-3B，文件名格式应该是：
```
Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800*.ckpt
```

## 方法2：使用查找脚本

我已经创建了一个帮助脚本 `find_checkpoint.sh`：

```bash
# 查找v2的RandSampler checkpoint
bash lever_lm/models/v3/find_checkpoint.sh v2 okvqa RandSampler Qwen2_5_VL_3B_Instruct

# 或者更简单（使用默认值）
bash lever_lm/models/v3/find_checkpoint.sh
```

## 方法3：使用通配符

如果文件名不确定，可以使用通配符：

```bash
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/*RandSampler*.ckpt \
    ...
```

**注意**：如果通配符匹配多个文件，shell会展开为多个参数，可能导致错误。建议先找到确切的文件名。

## 常见文件名格式

根据训练脚本，checkpoint文件名格式为：
```
{model_name}_{sampler_name}_infoscore_left_beam5_shot2_cand64_sample{sample_num}[_best][_resume].ckpt
```

示例：
- `Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800.ckpt`
- `Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt`
- `Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_resume.ckpt`

## 完整命令示例

找到文件后，使用完整路径：

```bash
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/Qwen2_5_VL_3B_Instruct_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best.ckpt \
    --beam_data results/okvqa/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:800.json \
    --output_path results/okvqa/generated_data/rl_data_RandSampler.json \
    --query_emb results/okvqa/cache/query_embeddings.pt \
    --cand_emb results/okvqa/cache/candidate_embeddings.pt \
    --device cuda:0
```

## 如果找不到文件

1. **检查是否已经训练过v2模型**：
   ```bash
   bash scripts/train_lever_lm.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v2
   ```

2. **检查目录是否存在**：
   ```bash
   ls -la results/okvqa/model_cpk/
   ```

3. **检查其他版本目录**：
   ```bash
   find results -name "*RandSampler*.ckpt" -type f
   ```
