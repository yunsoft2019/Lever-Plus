# RL 数据生成优化说明

## 优化内容

已优化 `lever_lm/models/v3/generate_rl_data.py`，避免重复加载 VQA 标注文件。

### 优化前
- 每个 pointer 候选都重新加载一次 VQA 标注文件
- 800 个 query × 10 个候选 ≈ 8000 次重复加载
- 预计耗时：**7-8 小时**

### 优化后
- 在函数开始时只加载一次 VQA 标注文件（训练集和验证集各一次）
- 后续所有计算复用已加载的 VQA 对象
- 预计耗时：**约 1 小时**（提升约 7-8 倍）

## 主要改动

1. **导入 VQA 和 VQAEval 类**：
   ```python
   from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy as compute_vqa_accuracy_metric, VQA, VQAEval
   ```

2. **在 `generate_rl_data` 函数开始时预加载 VQA 对象**：
   ```python
   # 预加载训练集和验证集 VQA 对象（只加载一次）
   vqa_train_cache = None
   vqa_val_cache = None
   if train_ques_path and train_ann_path:
       vqa_train_cache = VQA(train_ann_path, train_ques_path)
   if val_ques_path and val_ann_path:
       vqa_val_cache = VQA(val_ann_path, val_ques_path)
   ```

3. **修改 `compute_vqa_accuracy` 函数，支持 VQA 对象缓存**：
   - 新增 `vqa_cache` 参数
   - 如果提供了缓存，使用缓存的 VQA 对象计算准确率
   - 如果没有缓存，回退到原来的方式（兼容性）

4. **在调用 `compute_vqa_accuracy` 时传入缓存对象**：
   ```python
   compute_vqa_accuracy(..., vqa_cache=vqa_train_cache)
   ```

## 使用方法

优化后的代码会自动使用缓存机制，无需修改调用方式。

### 如果当前正在运行训练

**选项 1：等待当前运行完成**（推荐）
- 当前运行虽然慢，但功能正确
- 预计还需约 7 小时
- 完成后可以检查结果

**选项 2：中断并重新运行**（如果时间紧急）
```bash
# 1. 中断当前进程（Ctrl+C）
# 2. 重新运行训练脚本（会自动使用优化后的代码）
bash scripts/train_v3.sh vqa okvqa_local 7 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B
```

### 验证优化效果

优化后，你应该看到：
1. 开始时只打印一次 "loading VQA annotations and questions into memory..."
2. 后续不再重复加载文件
3. 速度明显提升（从 ~33 秒/query 降到 ~5 秒/query）

## 性能对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| VQA 文件加载次数 | ~8000 次 | 2 次 | **4000x** |
| 预计总耗时 | 7-8 小时 | ~1 小时 | **7-8x** |
| 每个 query 耗时 | ~33 秒 | ~5 秒 | **6.6x** |

## 进一步优化：抑制冗余输出

已添加输出抑制功能，避免每次计算时打印：
- "Loading and preparing results..."
- "creating index..."
- "computing accuracy" 和进度条

这些信息在批量处理时会产生大量输出，干扰 tqdm 进度条显示。

**实现方式**：使用 `contextlib.redirect_stdout` 重定向 VQA 库的输出到 StringIO，不影响 tqdm 的显示。

## 注意事项

1. **向后兼容**：如果没有提供 VQA 缓存，代码会自动回退到原来的方式
2. **内存使用**：VQA 对象会占用一些内存，但对于现代服务器来说通常不是问题
3. **错误处理**：如果预加载失败，会自动回退到原来的方式，不会影响功能
4. **输出抑制**：VQA 库的内部打印已被抑制，只保留关键信息（如预加载提示）

## 技术细节

优化原理：
- VQA 类的 `__init__` 方法会加载并解析标注文件，创建索引
- 这个过程只需要执行一次，后续可以复用
- `loadRes` 方法只需要加载结果文件（很小），不需要重新加载标注文件
- `VQAEval` 可以只评估指定的问题 ID，不需要评估所有问题

