# RL数据生成脚本 v4 使用说明

## 脚本位置
`scripts/generate_rl_data_v4.sh`

## 功能特点

✅ **严格一致性**：使用 `strict_eval` 模式，确保RL reward与最终评测完全一致
✅ **可复用性**：保存raw_generation、gt_answers、relevance等，后续无需重跑千问
✅ **完整信息**：保存pointer_pos、pointer(global)、eval信息等

## 使用方法

### 基本使用
```bash
# 使用默认参数（GPU=4, SAMPLER=RandSampler）
bash scripts/generate_rl_data_v4.sh

# 指定GPU和sampler
bash scripts/generate_rl_data_v4.sh 4 RandSampler

# 自定义输出文件名后缀
bash scripts/generate_rl_data_v4.sh 4 RandSampler v4_800queries_strictEval
```

### 参数说明
- `GPU_ID` (默认: 4): GPU设备ID
- `SAMPLER` (默认: RandSampler): sampler名称（如RandSampler, TextSimSampler等）
- `OUTPUT_SUFFIX` (默认: v4_strictEval): 输出文件名后缀

## 输出数据格式

生成的数据包含以下字段：

### _meta（元信息）
- `created_at`: 创建时间
- `vqa_model`: VQA模型名称
- `task_gen_args`: 生成参数（max_new_tokens, num_beams等）
- `strict_eval`: 是否启用严格模式
- `eval`: 评测文件路径

### query（查询级别）
- `query_id`: 查询ID
- `question_id`: 问题ID
- `image_id`: 图像ID
- `question`: 问题文本
- `gt_answers_raw`: 原始ground truth答案
- `gt_answers_norm`: 归一化后的ground truth答案

### pointer_candidates（候选级别）
- `pointer_pos`: position索引（用于candidate_pool）
- `pointer`: global索引（用于训练/分析）
- `vqa_raw_generation`: 原始生成结果（postprocess前）
- `vqa_pred_answer`: 后处理后的答案
- `vqa_acc_score`: VQA准确率分数 [0,1]
- `vqa_correct`: 是否正确 (0/1)
- `vqa_gt_prob`: GT概率 [0,1]
- `vqa_rel_token_f1`: token F1相关性
- `vqa_rel_edit_sim`: 编辑相似度
- `vqa_rel_score`: 相关性分数
- `vqa_eval_mode`: 评测模式（"vqaEval"或"fallback"）
- `eval_split_used`: 使用的数据集split（"train"或"val"）
- `eval_failed`: 是否评测失败

## 配置说明

### 数据生成配置
- Beam候选数: 5
- 温度采样: τ=1.0 x2 + τ=1.3 x2 = 4条
- 随机组合: 1条
- Retrieval: 5条
- 每个query总计: ~15条候选

### strict_eval模式
- 默认启用：`--strict_eval`
- 要求：必须提供至少一个VQA评测文件（train或val）
- 行为：禁用fallback，eval_failed的candidate会被跳过

### save_prompts选项
- 默认：不保存prompt文本（减少文件大小）
- 如需保存：在脚本中添加 `--save_prompts`

## 输出文件

- RL数据JSON: `results/okvqa/generated_data/rl_data_${SAMPLER}_${OUTPUT_SUFFIX}.json`
- 日志文件: `results/okvqa/generated_data/rl_data_${SAMPLER}_${OUTPUT_SUFFIX}.log`

## 下一步

生成完成后，使用新数据训练v3模型：
```bash
bash scripts/train_v3_with_new_rl_data.sh ${GPU_ID} ${SAMPLER} ${OUTPUT_PATH}
```

## 注意事项

1. **文件路径**：确保以下文件存在：
   - SFT checkpoint（v2格式）
   - Beam数据JSON
   - Query和Candidate embeddings
   - VQA评测文件（questions.json和annotations.json）

2. **strict_eval**：如果评测文件不存在，strict_eval模式会报错退出

3. **磁盘空间**：生成的数据文件较大，确保有足够空间

4. **GPU内存**：确保GPU有足够内存加载VQA模型

