# RL 数据生成实现指南

## 已完成的工作

### 1. ✅ 温度采样函数 (`rl_data_generation.py`)
- ✅ `compute_step_logits()`: 计算单步 logits（适配 PointerSelectorV2/V3）
- ✅ `sample_pointer_with_temperature()`: 温度采样实现（按照强化学习.md §3.2）
- ✅ `generate_pointer_candidates_for_query()`: 生成候选序列（beam + 温度采样 + 随机组合）
- ✅ `evaluate_pointer_candidate()`: Correctness 计算框架（按照强化学习.md §3.3）

### 2. ✅ Reward 计算工具 (`reward_utils.py`)
- ✅ `compute_reward_for_candidate()`: 组合 reward 计算（beam_score + correctness，按照强化学习.md §3.4）

### 3. ✅ 数据生成脚本框架 (`generate_rl_data.py`)
- ✅ 基础框架和接口定义
- ⚠️ 需要完成：VQA 模型加载、prompt 构建、准确率计算

## 待完成的工作

### 1. 完成 `generate_rl_data.py` 中的 TODO

#### 1.1 `load_vqa_model()` 函数
需要根据你的 VQA 模型接口实现：

```python
def load_vqa_model(model_name: str, device: torch.device):
    """
    加载 VQA 模型
    
    参考：你的项目中已有的 VQA 模型加载代码
    可能的位置：
    - open_mmicl/interface/qwen2vl_interface.py
    - open_mmicl/interface/flamingo_interface.py
    """
    # 示例（需要根据实际代码调整）：
    if model_name == "qwen2.5_vl_3B":
        from open_mmicl.interface.qwen2vl_interface import Qwen2VLInterface
        vqa_model = Qwen2VLInterface(model_name="Qwen/Qwen2.5-VL-3B-Instruct", device=device)
        return vqa_model
    elif model_name == "flamingo_3B":
        from open_mmicl.interface.flamingo_interface import FlamingoInterface
        vqa_model = FlamingoInterface(model_name="...", device=device)
        return vqa_model
    else:
        raise ValueError(f"Unknown VQA model: {model_name}")
```

#### 1.2 `build_vqa_prompt()` 函数
需要根据你的 prompt 模板实现：

```python
def build_vqa_prompt(image, question: str, ex1: Dict, ex2: Dict) -> str:
    """
    构建 VQA prompt
    
    参考：你的项目中已有的 prompt 构建代码
    可能的位置：
    - icl_inference.py 中的 prompt 构建逻辑
    - open_mmicl/interface/ 中的 prompt 模板
    """
    # 示例（需要根据实际代码调整）：
    # 根据你的 prompt 格式，例如：
    # "Question:<Q1> Short answer:<A1> Question:<Q2> Short answer:<A2> Question:<Q> Short answer:"
    
    prompt = f"Question:{ex1['question']} Short answer:{ex1['answer']} " \
             f"Question:{ex2['question']} Short answer:{ex2['answer']} " \
             f"Question:{question} Short answer:"
    
    return prompt
```

#### 1.3 `compute_vqa_accuracy()` 函数
需要根据你的 VQA 评估方式实现：

```python
def compute_vqa_accuracy(pred_answer: str, ground_truth_answers: List[str]) -> tuple:
    """
    计算 VQA 准确率
    
    参考：你的项目中已有的 VQA 评估代码
    可能的位置：
    - icl_inference.py 中的评估逻辑
    - 使用 VQAv2 评估脚本
    """
    # 示例（需要根据实际代码调整）：
    from vqa_eval import VQAEval
    
    # 使用标准 VQAv2 评估方式
    vqa_eval = VQAEval()
    acc_score = vqa_eval.compute_acc(pred_answer, ground_truth_answers)
    correct = 1 if acc_score > 0.0 else 0
    
    return correct, acc_score
```

#### 1.4 数据集和 Embedding 加载
需要根据你的数据格式实现：

```python
# 加载数据集
from datasets import load_dataset
dataset = load_dataset("your_dataset_name", split="train")

# 加载 embeddings
query_embeddings = torch.load("path/to/query_embeddings.pt")
candidate_embeddings = torch.load("path/to/candidate_embeddings.pt")
candidate_indices = list(range(len(candidate_embeddings)))
```

### 2. 更新数据格式

#### 2.1 新的数据格式（按照强化学习.md §2.1）

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
  }
}
```

### 3. 更新 `dataset_v3.py` 以支持新格式

需要修改 `BeamDataset` 和 `BeamDatasetWithEmbedding` 类，支持：
- 读取新的 `pointer_candidates` 格式
- 支持 `vqa_correct` 和 `vqa_acc_score` 字段
- 使用 `compute_reward_for_candidate()` 计算组合 reward

### 4. 更新训练脚本

在 `grpo_post_train.py` 中：
- 使用新的数据格式
- 使用组合 reward（beam_score + correctness）

## 使用流程

### Step 1: 生成 RL 数据

```bash
python -m lever_lm.models.v3.generate_rl_data \
    --sft_ckpt results/okvqa/model_cpk/v2/xxx.ckpt \
    --beam_data results/okvqa/generated_data/beam_RandSampler.json \
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

### Step 2: 使用新数据训练

```bash
python -m lever_lm.workflows.grpo_post_train \
    --beam_data results/okvqa/generated_data/rl_data_RandSampler.json \
    --img_emb results/okvqa/cache/img_embeddings.pt \
    --output_dir results/okvqa/model_cpk/v3_RandSampler \
    --rce_epochs 25 \
    --grpo_epochs 25 \
    --batch_size 32 \
    --rce_lr 5e-4 \
    --grpo_lr 5e-6 \
    --kl_beta 0.3 \
    --device cuda:0
```

## 关键改进点

1. **数据多样性**：beam + 温度采样 + 随机组合，避免"只是把 beam 再学一遍"
2. **端到端信号**：使用 correctness 作为 reward，直接优化 VQA 准确率
3. **探索能力**：温度采样和随机组合帮助发现高质量但概率低的组合

## 注意事项

1. **计算成本**：对每个 pointer 调用 VQA 模型，计算成本较高，建议：
   - 使用 GPU 加速
   - 批量处理
   - 缓存结果

2. **数据质量**：确保 correctness 计算准确，使用标准 VQAv2 评估方式

3. **Reward 设计**：根据实际效果调整 `alpha` 和 `beta` 权重

## 参考文档

- `强化学习.md`: 完整的设计文档
- `lever_lm/models/v3/rl_data_generation.py`: 核心实现
- `lever_lm/utils/reward_utils.py`: Reward 计算工具
