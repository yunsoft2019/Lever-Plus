# Lever-Plus

## 1. 环境部署

### 环境说明

1. 由于 faiss 当前最新版本是 1.13.0，而这个版本最高只支持 python 3.12，故本环境最高只能支持 python 3.12。

2. open_flamingo 支持的最高版本是 python 3.9, torch 2.0.1，通过修改 setup.py，可支持 3.12 及 torch 2.9.1，可直接安装已修改的依赖。

3. OpenICL 支持的最高版本是 python 3.10，通过修改 setup.py，可支持 3.12，可直接安装已修改的依赖。

### 安装步骤

```bash
conda create -n lever_env python=3.12 -y
conda activate lever_env
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.13.0 
git clone https://github.com/mlfoundations/open_flamingo.git
cd open_flamingo
# 修改setup.py, 使之支持python3.12, torch2.9.1,
pip install -e .
cd ..
git clone https://github.com/ForJadeForest/OpenICL.git
cd OpenICL
# 修改setup.py, 使之支持pyton3.12,faiss1.13.0
pip install -e .
cd ..

pip install hydra-core
pip install more_itertools
pip install python-dotenv
pip install pytorch-lightning
pip install omegaconf
pip install pycocotools
pip install pycocoevalcap
pip install tensorboard
pip install fsspec
pip install datasets
pip install aiohttp
pip install pyarrow
pip install loguru
pip install multiprocess
pip install -U rich
pip install qwen_vl_utils
```

## 2. 执行脚本

**重要提示**: 在执行脚本前需要下载相关模型参数。如果没有加速器，请先执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.1 束搜索

**参数说明**: `task dataset gpu_ids sampler [beam_model]`

- `beam_model` 可选值: `flamingo_3B` (默认) 或 `qwen2.5_vl_3B`

#### 使用 Flamingo-3B 模型

```bash
# 随机采样器（RandSampler）
bash scripts/generate_data.sh vqa okvqa_local "[0]" rand_sampler flamingo_3B

# 文本相似度采样器（TextSimSampler）
bash scripts/generate_data.sh vqa okvqa_local "[0]" text_sim_sampler flamingo_3B

# 图片相似度采样器（ImgSimSampler）
bash scripts/generate_data.sh vqa okvqa_local "[0]" img_sim_sampler flamingo_3B

# 混合采样器（MixSampler）
bash scripts/generate_data.sh vqa okvqa_local "[0]" mix_sampler flamingo_3B
```

#### 使用 Qwen2.5-VL-3B-Instruct 模型

```bash
# 随机采样器（RandSampler）
bash scripts/generate_data.sh vqa okvqa_local "[0]" rand_sampler qwen2.5_vl_3B

# 文本相似度采样器（TextSimSampler）
bash scripts/generate_data.sh vqa okvqa_local "[1]" text_sim_sampler qwen2.5_vl_3B

# 图片相似度采样器（ImgSimSampler）
bash scripts/generate_data.sh vqa okvqa_local "[2]" img_sim_sampler qwen2.5_vl_3B

# 混合采样器（MixSampler）
bash scripts/generate_data.sh vqa okvqa_local "[3]" mix_sampler qwen2.5_vl_3B
```

### 2.2 训练

**参数说明**: `task dataset gpu_id lever_lm sampler [beam_model] [version]`

- `gpu_id`: GPU 编号，例如 0 表示使用 GPU 0，1 表示使用 GPU 1（默认: 0）
- `beam_model` 可选值: `flamingo_3B` (默认) 或 `qwen2.5_vl_3B`
- `version` 可选值: `v0` (默认), `v1`, `v2`, `v3`, `v4` - 模型版本号，用于区分不同版本的模型代码和检查点
- **注意**: `beam_model` 必须与生成数据时使用的模型一致
- **注意**: 检查点文件保存在 `results/{dataset}/model_cpk/{version}/` 目录下

#### 使用 Flamingo-3B 生成的束搜索数据训练

```bash
# 随机采样器（使用 GPU 0，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B

# 文本相似度采样器（使用 GPU 1，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler flamingo_3B

# 图片相似度采样器（使用 GPU 2，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler flamingo_3B

# 混合采样器（使用 GPU 3，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler flamingo_3B

# v1版本训练（Bi-Encoder 指针网络架构，使用独立的编码器分别编码 query 和 candidates，通过 MLP 投影层和指针网络选择机制从候选池中选择范例）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler flamingo_3B v1

# v2版本训练（在 v1 的 Bi-Encoder 架构基础上添加了单层 Cross-Attention 机制，通过多头注意力增强 query 与 candidates 之间的细粒度交互能力，使用残差连接和 LayerNorm 提升训练稳定性）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler flamingo_3B v2

# v3版本训练（灵活基础架构 + 排序学习，可选择 v1 或 v2 作为基础架构，利用束搜索的多个 beam 进行排序学习，损失函数为交叉熵 + 排序损失，提升 Top-k、NDCG、MRR 等排序指标）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler flamingo_3B v3

# v4版本训练（V3 + 离线强化学习，在 v3 基础上新增离线强化学习阶段：先 RCE 预热，再 GRPO（PPO-clip + KL）后训练，利用束搜索的多条 beam 及分数进一步优化候选排序与端到端指标）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler flamingo_3B v4
```

#### 使用 Qwen2.5-VL-3B-Instruct 生成的束搜索数据训练

```bash
# 随机采样器（使用 GPU 0，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

# 文本相似度采样器（使用 GPU 1，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler qwen2.5_vl_3B

# 图片相似度采样器（使用 GPU 2，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler qwen2.5_vl_3B

# 混合采样器（使用 GPU 3，默认版本 v0）
bash scripts/train_lever_lm.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler qwen2.5_vl_3B

# v1版本训练（Bi-Encoder 指针网络架构，使用独立的编码器分别编码 query 和 candidates，通过 MLP 投影层和指针网络选择机制从候选池中选择范例）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v1

# v2版本训练（在 v1 的 Bi-Encoder 架构基础上添加了单层 Cross-Attention 机制，通过多头注意力增强 query 与 candidates 之间的细粒度交互能力，使用残差连接和 LayerNorm 提升训练稳定性）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v2

# v3版本训练（灵活基础架构 + 排序学习，可选择 v1 或 v2 作为基础架构，利用束搜索的多个 beam 进行排序学习，损失函数为交叉熵 + 排序损失，提升 Top-k、NDCG、MRR 等排序指标）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3

# v4版本训练（V3 + 离线强化学习，在 v3 基础上新增离线强化学习阶段：先 RCE 预热，再 GRPO（PPO-clip + KL）后训练，利用束搜索的多条 beam 及分数进一步优化候选排序与端到端指标）
bash scripts/train_lever_lm.sh vqa okvqa_local 4 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v4
```

### 2.3 基线

**参数说明**: `task dataset device model`

- `device`: GPU 编号，例如 0 表示使用 GPU 0，1 表示使用 GPU 1（默认: 0）
- `model`: 模型名称，可选值: `flamingo_3B` 或 `qwen2.5_vl_3B`
- **说明**: 使用随机范例（RandomRetriever）进行基线推理，从整个训练集中随机选择范例
- **Shot Num**: 自动测试 1, 2, 3, 4 个范例
- **结果文件**: 保存在 `results/{dataset}/icl_inference/baseline/{model}_RandomRetriever_baseline_metrics.json`

```bash
# 基线推理（Flamingo-3B）
bash scripts/baseline.sh vqa okvqa_local 0 flamingo_3B

# 基线推理（Qwen2.5-VL-3B）
bash scripts/baseline.sh vqa okvqa_local 1 qwen2.5_vl_3B
```

### 2.4 推理

**参数说明**: `task dataset device lever_lm sampler [beam_model] [version]`

- `device`: GPU 编号，例如 0 表示使用 GPU 0，1 表示使用 GPU 1（默认: 0）
- `beam_model` 可选值: `flamingo_3B` (默认) 或 `qwen2.5_vl_3B`
- `version` 可选值: `v0` (默认), `v1`, `v2`, `v3`, `v4` - 模型版本号，必须与训练时使用的版本一致
- **注意**: `beam_model` 必须与训练时使用的模型一致，用于选择对应的检查点文件
- **注意**: `version` 必须与训练时使用的版本一致，用于从正确的目录加载检查点
- **注意**: 推理时批量大小固定为1，避免批处理时的图像数量不匹配问题

**后台运行**: 推理任务通常需要较长时间，建议使用后台运行脚本 `scripts/run_inference_background.sh`，该脚本会自动激活 conda 环境并将输出保存到日志文件。

```bash
# 后台运行推理任务
bash scripts/run_inference_background.sh vqa okvqa_local 3 query_img_text_icd_img_text text_sim_sampler flamingo_3B v1

# 查看实时日志
tail -f logs/inference/inference_vqa_okvqa_local_3_text_sim_sampler_flamingo_3B_v1_*.log

# 查看进程状态
ps -p <PID>

# 停止任务（如果需要）
kill <PID>
```

#### 使用 Flamingo-3B 训练的模型进行推理

```bash
# 随机采样器（RandSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B

# 文本相似度采样器（TextSimSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text text_sim_sampler flamingo_3B

# 图片相似度采样器（ImgSimSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text img_sim_sampler flamingo_3B

# 混合采样器（MixSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text mix_sampler flamingo_3B

# v1版本推理（Bi-Encoder 指针网络架构，使用独立的编码器分别编码 query 和 candidates，通过 MLP 投影层和指针网络选择机制从候选池中选择范例）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B v1

# v2版本推理（在 v1 的 Bi-Encoder 架构基础上添加了单层 Cross-Attention 机制，通过多头注意力增强 query 与 candidates 之间的细粒度交互能力，使用残差连接和 LayerNorm 提升训练稳定性）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B v2

# v3版本推理（灵活基础架构 + 排序学习，可选择 v1 或 v2 作为基础架构，利用束搜索的多个 beam 进行排序学习，损失函数为交叉熵 + 排序损失，提升 Top-k、NDCG、MRR 等排序指标）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B v3

# v4版本推理（V3 + 离线强化学习，在 v3 基础上新增离线强化学习阶段：先 RCE 预热，再 GRPO（PPO-clip + KL）后训练，利用束搜索的多条 beam 及分数进一步优化候选排序与端到端指标）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B v4
```

#### 使用 Qwen2.5-VL-3B-Instruct 训练的模型进行推理

```bash
# 随机采样器（RandSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

# 文本相似度采样器（TextSimSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler qwen2.5_vl_3B

# 图片相似度采样器（ImgSimSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler qwen2.5_vl_3B

# 混合采样器（MixSampler，默认版本 v0）
bash scripts/inference.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler qwen2.5_vl_3B

# v1版本推理（Bi-Encoder 指针网络架构，使用独立的编码器分别编码 query 和 candidates，通过 MLP 投影层和指针网络选择机制从候选池中选择范例）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v1

# v2版本推理（在 v1 的 Bi-Encoder 架构基础上添加了单层 Cross-Attention 机制，通过多头注意力增强 query 与 candidates 之间的细粒度交互能力，使用残差连接和 LayerNorm 提升训练稳定性）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v2

# v3版本推理（灵活基础架构 + 排序学习，可选择 v1 或 v2 作为基础架构，利用束搜索的多个 beam 进行排序学习，损失函数为交叉熵 + 排序损失，提升 Top-k、NDCG、MRR 等排序指标）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v3

# v4版本推理（V3 + 离线强化学习，在 v3 基础上新增离线强化学习阶段：先 RCE 预热，再 GRPO（PPO-clip + KL）后训练，利用束搜索的多条 beam 及分数进一步优化候选排序与端到端指标）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B v4
```

## 3. 推理结果

### 3.1 基线推理结果（随机范例）

使用随机范例（RandomRetriever）的基线结果：

| Shot Num | Flamingo-3B | Qwen2.5-VL-3B-Instruct |
|----------|-------------|------------------------|
| 1        | 19.96       | **50.59**              |
| 2        | 20.50       | 47.04                  |
| 3        | 21.68       | 45.48                  |
| 4        | **22.33**   | 44.93                  |

**说明**: 
- **Flamingo-3B**: 最佳结果为 22.33% (shot_num=4)
- **Qwen2.5-VL-3B-Instruct**: 最佳结果为 50.59% (shot_num=1)

### 3.2 v0 推理结果

**模型说明**: v0 模型基于 GPT2 自回归语言模型架构，使用 CLIP 编码器编码 query 和 ICD（In-Context Demonstration），通过自回归生成的方式选择范例索引序列。

#### 3.2.1 Flamingo-3B 模型结果（LeverLM v0）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | **23.04**   | 19.98          | 21.05         | 20.63      |
| 2        | 20.97       | 21.89          | **23.06**     | 20.94      |
| 3        | 23.29       | 22.94          | **23.63**     | 20.31      |
| 4        | **25.28**   | 24.14          | 24.59         | 22.87      |

**最佳结果**: 25.28% (RandSampler, shot_num=4)

#### 3.2.2 Qwen2.5-VL-3B-Instruct 模型结果（LeverLM v0）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | **52.04**   | 48.91          | 48.60         | 50.19      |
| 2        | 49.76       | 44.66          | 43.98         | **46.36**  |
| 3        | 48.06       | 43.54          | 42.55         | 45.36      |
| 4        | 47.60       | 41.76          | 42.12         | 44.08      |

**最佳结果**: 52.04% (RandSampler, shot_num=1)

### 3.3 v1 推理结果

**模型说明**: v1 模型采用 Bi-Encoder 指针网络架构，使用独立的编码器分别编码 query 和 candidates，通过 MLP 投影层和指针网络选择机制从候选池中选择范例，支持 Teacher Forcing 训练。

#### 3.3.1 Flamingo-3B 模型结果（LeverLM v1）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | 21.96       | 19.71          | 21.96         | 19.71      |
| 2        | 22.03       | 22.59          | 22.03         | 22.59      |
| 3        | 22.64       | 23.32          | 22.64         | 23.32      |
| 4        | 22.76       | **24.29**      | 22.76         | **24.29**  |

**最佳结果**: 24.29% (TextSimSampler/MixSampler, shot_num=4)

#### 3.3.2 Qwen2.5-VL-3B-Instruct 模型结果（LeverLM v1）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | **48.63**   | -              | -             | -          |
| 2        | 47.26       | -              | -             | -          |
| 3        | 47.58       | -              | -             | -          |
| 4        | 47.27       | -              | -             | -          |

**最佳结果**: 48.63% (RandSampler, shot_num=1)

### 3.4 v2 推理结果

**模型说明**: v2 模型在 v1 的 Bi-Encoder 架构基础上添加了单层 Cross-Attention 机制，通过多头注意力增强 query 与 candidates 之间的细粒度交互能力，使用残差连接和 LayerNorm 提升训练稳定性，从而更准确地从候选池中选择相关范例。

#### 3.4.1 Flamingo-3B 模型结果（LeverLM v2）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | 21.96       | -              | -             | -          |
| 2        | 22.03       | -              | -             | -          |
| 3        | 22.64       | -              | -             | -          |
| 4        | **22.76**    | -              | -             | -          |

**最佳结果**: 22.76% (RandSampler, shot_num=4)

#### 3.4.2 Qwen2.5-VL-3B-Instruct 模型结果（LeverLM v2）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | **51.32**   | -              | -             | -          |
| 2        | 47.23       | -              | -             | -          |
| 3        | 46.86       | -              | -             | -          |
| 4        | 46.94       | -              | -             | -          |

**最佳结果**: 51.32% (RandSampler, shot_num=1)

### 3.5 v3 推理结果

**模型说明**: v3 模型采用灵活基础架构 + 排序学习（Ranking Learning）设计，可选择 v1（Bi-Encoder）或 v2（+ Cross-Attention）作为基础架构，利用束搜索的多个 beam 进行排序学习，损失函数为交叉熵（CE）+ 排序损失（Ranking Loss），通过 Listwise 或 Pairwise 方式提升 Top-k、NDCG、MRR 等排序指标。

#### 3.5.1 Flamingo-3B 模型结果（LeverLM v3）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | 21.98       | -              | -             | -          |
| 2        | 22.03       | -              | -             | -          |
| 3        | 22.64       | -              | -             | -          |
| 4        | **22.76**    | -              | -             | -          |

**最佳结果**: 22.76% (RandSampler, shot_num=4)

#### 3.5.2 Qwen2.5-VL-3B-Instruct 模型结果（LeverLM v3）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | **51.33**   | -              | -             | -          |
| 2        | 47.23       | -              | -             | -          |
| 3        | 46.86       | -              | -             | -          |
| 4        | 46.94       | -              | -             | -          |

**最佳结果**: 51.33% (RandSampler, shot_num=1)

### 3.6 v4 推理结果

**模型说明**: v4 模型在 v3（灵活基础架构 + 排序学习）的基础上，新增离线强化学习阶段：先 RCE（Reward-weighted Cross-Entropy）预热，再 GRPO（Group-Relative Policy Optimization with PPO-style clipping + KL 正则）后训练，利用束搜索的多条 beam 及分数进一步优化候选排序与端到端指标（VQA Acc），期望在 v3 基础上进一步提升排序质量与 VQA 准确率（通常 0.2%~1.0% 额外增益）。

#### 3.6.1 Flamingo-3B 模型结果（LeverLM v4）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | -           | -              | -             | -          |
| 2        | -           | -              | -             | -          |
| 3        | -           | -              | -             | -          |
| 4        | -           | -              | -             | -          |

**说明**: Flamingo-3B v4 推理结果待补充

#### 3.6.2 Qwen2.5-VL-3B-Instruct 模型结果（LeverLM v4）

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | -           | -              | -             | -          |
| 2        | -           | -              | -             | -          |
| 3        | -           | -              | -             | -          |
| 4        | -           | -              | -             | -          |

**说明**: Qwen2.5-VL-3B-Instruct v4 推理结果待补充

### 3.7 结果说明

- **数据集**: OKVQA
- **训练参数**: infoscore_left_beam5_shot2_cand64_sample800
- **基线结果**（从整个训练集中随机选择）:
  - Flamingo-3B: 最佳结果为 22.33% (shot_num=4)
  - Qwen2.5-VL-3B-Instruct: 最佳结果为 50.59% (shot_num=1)
- **v0 模型结果**（从64个候选范例中，通过束搜索+SFT选择）:
  - Flamingo-3B: 最佳配置为 RandSampler + shot_num=4，准确率 25.28%
  - Qwen2.5-VL-3B-Instruct: 最佳配置为 RandSampler + shot_num=1，准确率 52.04%

**结果对比分析**:
- **基线 vs 方法**: 束搜索+SFT相比基线（从整个训练集随机选择）提升了约2-3个百分点，说明束搜索+SFT方法有效
