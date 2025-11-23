# Pointer Network V1 推理状态说明

## 📋 当前状态

✅ **推理脚本已完成**
- `workflows/pointer_inference_vqa.py` - V1推理主程序（集成VQA评估）
- `scripts/pointer_inference.sh` - 推理执行脚本

⚠️ **缺少V1模型检查点**
- 现有检查点 (`results/model_cpk/vqa/.../last.pt`) 是**V0模型**
- V1推理需要使用V1训练脚本生成的检查点

## 🚀 使用步骤

### 第1步：训练V1模型（生成检查点）

```bash
cd /mnt/share/yiyun/Projects/VLM/Lever-Plus/Lever-Plus-04

# 运行V1模型训练（约10-20分钟）
./scripts/pointer_train.sh
```

**训练配置：**
- hidden_dim: 128（降维后的隐藏层维度）
- dropout: 0.3（强正则化）
- temperature: 0.1
- label_smoothing: 0.2
- 训练样本: 4949个
- Epoch: 2

**预期输出：**
训练完成后，会在以下目录生成检查点：
```
results/model_cpk/vqa/vqa_vqav2_random_train_Qwen2.5-VL-3B_gain_samples4949_icds32_beams3_shots6/
├── last.pt              # 最终模型
├── checkpoint.pt        # 完整检查点（包含优化器状态）
└── training_history.json # 训练历史
```

### 第2步：运行V1推理

训练完成后，直接运行：

```bash
./scripts/pointer_inference.sh
```

## 📊 推理流程

V1推理脚本执行以下步骤：

1. **加载V1模型**
   - 从检查点加载训练好的Pointer Network V1
   - 使用Bi-Encoder架构进行指针选择

2. **加载嵌入向量**
   - 查询嵌入：`pointer_embeddings/vqa/.../val_query_embeddings.npy`
   - 候选嵌入：`pointer_embeddings/vqa/.../candidate_embeddings.npy`

3. **V1指针选择**
   - 对每个查询，使用V1模型预测ICDS序列
   - 输出：每个查询对应的top-K候选ID列表

4. **VLM推理**
   - 使用预测的ICDS构建prompt
   - 调用Qwen2.5-VL-3B进行VQA推理

5. **评估打分**
   - 计算准确率
   - 保存详细的推理结果和评估指标

## 📁 输出文件

推理结果保存在：
```
results/inference/vqa/vqa_vqav2_random_train_Qwen2.5-VL-3B_gain_samples4949_icds32_beams3_shots6_v1_inferences<N>/
├── inference_results.json  # 详细推理结果
└── metrics.json            # 评估指标（准确率等）
```

## 🔍 检查当前检查点类型

可以使用以下命令检查检查点是V0还是V1：

```bash
python3 -c "
import torch
ckpt = torch.load('results/model_cpk/vqa/vqa_vqav2_random_train_Qwen2.5-VL-3B_gain_samples4949_icds32_beams3_shots6/last.pt', map_location='cpu')
print('检查点层名称（前5个）:')
for k in list(ckpt.keys())[:5]:
    print(f'  {k}')
print(f'\\n类型判断:')
print(f'  V1模型特征 (input_proj): {any(\"input_proj\" in k for k in ckpt.keys())}')
print(f'  V0模型特征 (fusion): {any(\"fusion\" in k for k in ckpt.keys())}')
"
```

**V1模型应包含：**
- `input_proj.*` - 输入降维层
- `query_proj.*` - 查询投影层
- `cand_proj.*` - 候选投影层
- `dropout.*` - Dropout层

**V0模型包含：**
- `fusion.*` - 融合层
- `transformer.*` - Transformer层

## ❓ 常见问题

### Q1: 为什么不能直接使用现有检查点？
A: 现有检查点是V0模型（基于Transformer），而V1模型使用完全不同的架构（Bi-Encoder）。两者的网络层结构完全不同，无法相互兼容。

### Q2: V1相比V0有什么改进？
A: 
- **更简单的架构**：移除复杂的Transformer，使用双编码器+点积注意力
- **更强的正则化**：添加dropout、降维、label smoothing
- **更好的泛化**：在相同数据上，V1模型收敛更稳定

### Q3: 训练需要多长时间？
A: 
- 数据加载：~1-2分钟
- Epoch 1：~5-8分钟
- Epoch 2：~5-8分钟
- 总计：约15-20分钟

### Q4: 可以修改推理参数吗？
A: 可以编辑 `scripts/pointer_inference.sh`，主要参数：
- `--shot_num`: ICDS数量（默认6）
- `--inference_num`: 推理样本数量
- `--device`: 使用的GPU（默认cuda:0）

## 📝 后续改进建议

1. **修改训练脚本保存逻辑**
   在 `workflows/pointer_train.py` 中保存完整的模型配置：
   ```python
   torch.save({
       'model_config': vars(model_config),  # 添加这一行
       'model_state_dict': model.state_dict(),
       ...
   }, checkpoint_path)
   ```

2. **添加检查点验证**
   推理脚本可以自动检测检查点类型并给出友好提示

3. **支持多种检查点格式**
   兼容不同版本的检查点格式





