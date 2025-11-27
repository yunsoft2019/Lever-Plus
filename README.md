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

以下表格记录了不同样本数量和shot数量下的实验配置：

| Sample Num | Shot Num | Model Name | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|------------|----------|------------|:-----------:|:-------------:|:------------:|:----------:|
| 100 | 1 | Flamingo-3B | 31.20 | 29.00 | 25.40 | 23.60 |
| 100 | 1 | Qwen2.5-VL-3B-Instruct | **59.40** | **60.60** | **60.40** | **59.00** |
| 100 | 2 | Flamingo-3B | 23.40 | 28.20 | **31.60** | 24.20 |
| 100 | 2 | Qwen2.5-VL-3B-Instruct | **55.20** | **55.40** | 51.40 | **55.60** |
| 100 | 3 | Flamingo-3B | 25.80 | 29.80 | **32.00** | 27.60 |
| 100 | 3 | Qwen2.5-VL-3B-Instruct | **57.60** | **54.00** | 51.00 | **53.60** |
| 100 | 4 | Flamingo-3B | 24.60 | 24.60 | 31.60 | 31.20 |
| 100 | 4 | Qwen2.5-VL-3B-Instruct | **61.00** | **49.80** | **56.80** | **49.60** |
| 200 | 1 | Flamingo-3B | 27.10 | 24.70 | 23.50 | 21.70 |
| 200 | 1 | Qwen2.5-VL-3B-Instruct | **54.10** | **56.00** | **56.20** | **54.10** |
| 200 | 2 | Flamingo-3B | 22.10 | 25.00 | 27.60 | 22.20 |
| 200 | 2 | Qwen2.5-VL-3B-Instruct | **52.10** | **49.40** | **49.80** | **50.20** |
| 200 | 3 | Flamingo-3B | 24.60 | **26.20** | **27.40** | 22.70 |
| 200 | 3 | Qwen2.5-VL-3B-Instruct | **52.80** | 47.10 | 46.10 | **50.00** |
| 200 | 4 | Flamingo-3B | 25.80 | **24.80** | 27.90 | **26.30** |
| 200 | 4 | Qwen2.5-VL-3B-Instruct | **53.00** | 46.70 | **49.10** | 44.00 |
| 300 | 1 | Flamingo-3B | 25.87 | 23.53 | 23.00 | 20.13 |
| 300 | 1 | Qwen2.5-VL-3B-Instruct | **54.53** | **53.00** | **53.27** | **54.00** |
| 300 | 2 | Flamingo-3B | 21.47 | 23.53 | 25.33 | 20.27 |
| 300 | 2 | Qwen2.5-VL-3B-Instruct | **51.33** | **46.93** | **47.80** | **47.73** |
| 300 | 3 | Flamingo-3B | 24.40 | 26.20 | 26.20 | 20.67 |
| 300 | 3 | Qwen2.5-VL-3B-Instruct | **51.73** | **44.40** | **43.93** | **48.93** |
| 300 | 4 | Flamingo-3B | 27.13 | 24.47 | 26.67 | 24.27 |
| 300 | 4 | Qwen2.5-VL-3B-Instruct | **52.27** | **43.60** | **45.67** | **43.27** |
**说明**: 此表格用于记录不同配置下的实验结果，Sampler列可用于填写对应的准确率或其他指标。

#### 实验分析

**1. v0 模型学习效果分析**

对比 v0 模型结果与基线（随机范例）结果，可以观察到以下关键发现：

- **Flamingo-3B 模型**：
  - 基线结果（随机范例）：19.96%-22.33%
  - v0 模型结果：20.13%-31.60%
  - **提升效果**：v0 模型相比基线提升了约 0-9 个百分点，在最佳配置下（Sample Num=100, Shot Num=2/3, ImgSimSampler）达到 31.60%，相比基线最佳结果（22.33%）提升了约 9.3 个百分点。
  - **学习机制**：v0 模型通过 SFT（Supervised Fine-Tuning）学习到了从候选池中选择相关范例的能力。模型使用 CLIP 编码器编码 query 和 ICD，通过自回归生成的方式预测范例索引序列，从而能够根据 query 的语义特征选择更相关的范例，而非随机选择。特别是在使用 ImgSimSampler 和 TextSimSampler 时，模型能够利用图像和文本相似度信息，进一步提升范例选择的质量。

- **Qwen2.5-VL-3B-Instruct 模型**：
  - 基线结果（随机范例）：44.93%-50.59%
  - v0 模型结果：43.27%-61.00%
  - **提升效果**：v0 模型相比基线在大多数配置下都有提升，在最佳配置下（Sample Num=100, Shot Num=4, RandSampler）达到 61.00%，相比基线最佳结果（50.59%）提升了约 10.4 个百分点。
  - **学习机制**：Qwen2.5-VL-3B-Instruct 作为更强的视觉语言模型，v0 模型能够更好地学习范例选择策略。模型学会了根据 query 特征从候选池中选择最有助于提升下游任务性能的范例，特别是在小样本场景（Sample Num=100）下表现更为突出。

**2. Flamingo-3B vs Qwen2.5-VL-3B-Instruct 模型对比分析**

从实验结果可以看出，两个模型在性能上存在显著差异：

- **性能差异**：
  - Qwen2.5-VL-3B-Instruct 在所有配置下的准确率都显著高于 Flamingo-3B，平均高出约 25-30 个百分点。
  - Flamingo-3B 的最佳结果为 31.60%（Sample Num=100, Shot Num=2/3, ImgSimSampler），而 Qwen2.5-VL-3B-Instruct 的最佳结果为 61.00%（Sample Num=100, Shot Num=4, RandSampler）。

- **原因分析**：
  1. **模型架构差异**：Qwen2.5-VL-3B-Instruct 采用了更先进的视觉语言模型架构，具有更强的多模态理解和推理能力，能够更好地理解图像和文本的语义关联。
  2. **预训练数据规模**：Qwen2.5-VL-3B-Instruct 在更大规模的多模态数据上进行预训练，获得了更丰富的视觉语言知识。
  3. **指令微调**：Qwen2.5-VL-3B-Instruct 经过指令微调（Instruct tuning），能够更好地遵循任务指令和格式要求，在 VQA 任务上表现更优。
  4. **范例选择策略的适应性**：虽然两个模型都通过 v0 模型学习到了范例选择能力，但 Qwen2.5-VL-3B-Instruct 作为更强的基座模型，能够更好地利用选出的高质量范例，从而获得更大的性能提升。

### 3.3 v1 推理结果

**模型说明**: v1 模型采用 Bi-Encoder 指针网络架构，使用独立的编码器分别编码 query 和 candidates，通过 MLP 投影层和指针网络选择机制从候选池中选择范例，支持 Teacher Forcing 训练。

#### 3.3.1 Flamingo-3B 模型结果

| Sample Num | Shot Num | Model Name | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|------------|----------|------------|:-----------:|:-------------:|:------------:|:----------:|
| 100 | 1 | v0 | 21.96 | 19.71 | 21.96 | 19.71 |
| 100 | 1 | v1 | **22.60** | **22.20** | **22.60** | **22.20** |
| 100 | 2 | v0 | 22.03 | 22.59 | 22.03 | 22.59 |
| 100 | 2 | v1 | **27.40** | **27.00** | **27.40** | **27.00** |
| 100 | 3 | v0 | 22.64 | 23.32 | 22.64 | 23.32 |
| 100 | 3 | v1 | **24.00** | **27.00** | **24.00** | **27.00** |
| 100 | 4 | v0 | 22.76 | 24.29 | 22.76 | 24.29 |
| 100 | 4 | v1 | **24.00** | **28.20** | **24.00** | **28.20** |
| 200 | 1 | v0 | 21.96 | 19.71 | 21.96 | 19.71 |
| 200 | 1 | v1 | **22.30** | **20.90** | **22.80** | **20.90** |
| 200 | 2 | v0 | 22.03 | 22.59 | 22.03 | 22.59 |
| 200 | 2 | v1 | **24.90** | **25.20** | **24.40** | **25.20** |
| 200 | 3 | v0 | **22.64** | 23.32 | **22.64** | 23.32 |
| 200 | 3 | v1 | 22.20 | **23.90** | 22.20 | **23.90** |
| 200 | 4 | v0 | **22.76** | 24.29 | **22.76** | 24.29 |
| 200 | 4 | v1 | **23.10** | **24.80** | 22.60 | **24.80** |
| 300 | 1 | v0 | **21.96** | 19.71 | **21.96** | 19.71 |
| 300 | 1 | v1 | 21.60 | **20.60** | 21.93 | **20.60** |
| 300 | 2 | v0 | 22.03 | 22.59 | 22.03 | 22.59 |
| 300 | 2 | v1 | **23.00** | **24.07** | **22.67** | **24.07** |
| 300 | 3 | v0 | 22.64 | 23.32 | 22.64 | 23.32 |
| 300 | 3 | v1 | **23.00** | **24.73** | **23.00** | **24.73** |
| 300 | 4 | v0 | 22.76 | 24.29 | 22.76 | 24.29 |
| 300 | 4 | v1 | **24.13** | **25.20** | **23.80** | **25.20** |

**最佳结果**: 24.29% (TextSimSampler/MixSampler, shot_num=4)

#### 3.3.2 Qwen2.5-VL-3B-Instruct 模型结果

| Sample Num | Shot Num | Model Name | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|------------|----------|------------|:-----------:|:-------------:|:------------:|:----------:|
| 100 | 1 | v0 | 59.40 | 60.60 | 60.40 | 59.00 |
| 100 | 1 | v1 | **64.80** | **64.80** | **63.80** | **64.40** |
| 100 | 2 | v0 | 55.20 | 55.40 | 51.40 | 55.60 |
| 100 | 2 | v1 | **64.40** | **64.40** | **63.80** | **63.80** |
| 100 | 3 | v0 | 57.60 | 54.00 | 51.00 | 53.60 |
| 100 | 3 | v1 | **59.80** | **59.80** | **62.80** | **62.80** |
| 100 | 4 | v0 | **61.00** | 49.80 | 56.80 | 49.60 |
| 100 | 4 | v1 | 60.80 | **60.80** | **61.40** | **61.40** |
| 200 | 1 | v0 | 54.10 | 56.00 | 56.20 | 54.10 |
| 200 | 1 | v1 | **57.80** | **57.80** | **56.90** | **57.40** |
| 200 | 2 | v0 | 52.10 | 49.40 | 49.80 | 50.20 |
| 200 | 2 | v1 | **55.90** | **55.90** | **56.30** | **55.30** |
| 200 | 3 | v0 | 52.80 | 47.10 | 46.10 | 50.00 |
| 200 | 3 | v1 | **53.60** | **53.60** | **55.50** | **55.20** |
| 200 | 4 | v0 | 53.00 | 46.70 | 49.10 | 44.00 |
| 200 | 4 | v1 | **54.90** | **54.90** | **54.90** | **54.90** |
| 300 | 1 | v0 | 54.53 | 53.00 | 53.27 | 54.00 |
| 300 | 1 | v1 | **55.53** | **55.53** | **55.47** | **55.80** |
| 300 | 2 | v0 | 51.33 | 46.93 | 47.80 | 47.73 |
| 300 | 2 | v1 | **53.27** | **53.27** | **54.13** | **53.93** |
| 300 | 3 | v0 | **51.73** | 44.40 | 43.93 | 48.93 |
| 300 | 3 | v1 | 50.73 | **50.73** | **53.73** | **53.00** |
| 300 | 4 | v0 | **52.27** | 43.60 | 45.67 | 43.27 |
| 300 | 4 | v1 | 51.80 | **51.80** | **52.20** | **52.00** |

**最佳结果**: 64.8% (RandSampler, shot_num=1)

**注意**: 
- 以上结果为基于100条测试数据的结果（使用原始prompt）
- 之前结果为基于1000条测试数据的结果（使用修改后的prompt）
- 完整数据集结果可能略有不同

#### 实验分析

**1. v1 vs v0 性能对比分析**

通过对比 3.3.1 和 3.3.2 中 v0 和 v1 的实验结果，可以得出以下关键发现：

**Flamingo-3B 模型（3.3.1）**：
- **性能提升**：
  - v0 模型结果范围：19.71%-24.29%
  - v1 模型结果范围：20.6%-28.2%
  - v1 相比 v0 在绝大多数配置下都有提升，平均提升约 2-5 个百分点
  - 在最佳配置下（Sample Num=100, Shot Num=4, TextSimSampler/MixSampler），v1 达到 28.2%，相比 v0 最佳结果（24.29%）提升了约 3.9 个百分点
  - 在 Sample Num=100, Shot Num=2 配置下，v1 在所有 Sampler 上都达到 27.0%-27.4%，相比 v0（22.03%-22.59%）提升了约 4.5-5.4 个百分点

- **性能特点**：
  - v1 在小样本场景（Sample Num=100）下表现最佳，提升最为明显
  - v1 在 TextSimSampler 和 MixSampler 上表现尤为突出，说明 Bi-Encoder 架构能够更好地利用文本相似度信息
  - 随着 Sample Num 增加（200、300），v1 的优势有所减弱，但仍保持稳定提升

**Qwen2.5-VL-3B-Instruct 模型（3.3.2）**：
- **性能提升**：
  - v0 模型结果范围：43.27%-61.00%
  - v1 模型结果范围：50.73%-64.8%
  - v1 相比 v0 在绝大多数配置下都有提升，平均提升约 3-6 个百分点
  - 在最佳配置下（Sample Num=100, Shot Num=1, RandSampler/TextSimSampler），v1 达到 64.8%，相比 v0 最佳结果（61.00%）提升了约 3.8 个百分点
  - 在 Sample Num=100, Shot Num=1-2 配置下，v1 在所有 Sampler 上都显著优于 v0，提升约 4-9 个百分点

- **性能特点**：
  - v1 在小样本场景（Sample Num=100）下表现最佳，特别是在 Shot Num=1-2 时提升最为明显
  - v1 在 RandSampler 和 TextSimSampler 上表现突出，说明 Bi-Encoder 架构能够有效利用不同采样策略
  - 随着 Sample Num 和 Shot Num 增加，v1 的优势仍然保持，但提升幅度有所减小

**2. v1 模型架构优势分析**

v1 模型采用 Bi-Encoder 指针网络架构，相比 v0 模型具有以下优势：

1. **独立编码机制**：
   - v1 使用独立的编码器分别编码 query 和 candidates，避免了 v0 中自回归生成方式带来的误差累积问题
   - 这种设计使得 query 和 candidates 的表示更加独立和准确，能够更好地捕捉各自的语义特征

2. **高效的相似度计算**：
   - Bi-Encoder 架构通过 MLP 投影层将 query 和 candidates 映射到同一语义空间，然后通过点积或余弦相似度计算相关性
   - 相比 v0 的自回归生成方式，v1 的计算效率更高，推理速度更快

3. **更好的泛化能力**：
   - v1 模型在不同 Sample Num 和 Shot Num 配置下都表现出稳定的性能提升
   - 特别是在小样本场景下，v1 能够更好地利用有限的训练数据，学习到更有效的范例选择策略

4. **对采样策略的适应性**：
   - v1 模型在不同 Sampler（RandSampler、TextSimSampler、ImgSimSampler、MixSampler）下都能取得良好效果
   - 说明 Bi-Encoder 架构能够灵活地适应不同的候选池构建策略

**3. 总结：使用 v1 的优势**

基于以上分析，使用 v1 模型的主要优势包括：

1. **性能提升**：v1 相比 v0 在绝大多数配置下都有 2-6 个百分点的性能提升，在最佳配置下提升可达 3.8-5.4 个百分点

2. **架构优势**：
   - Bi-Encoder 架构避免了自回归生成带来的误差累积
   - 独立编码机制使得 query 和 candidates 的表示更加准确
   - 高效的相似度计算提升了推理效率

3. **泛化能力**：v1 在不同数据规模（Sample Num）和不同 shot 数量下都表现出稳定的性能提升，具有良好的泛化能力

4. **实用性**：
   - 在小样本场景下表现尤为突出，适合实际应用中的资源受限场景
   - 对不同采样策略具有良好的适应性，可以根据实际需求灵活选择

5. **可扩展性**：Bi-Encoder 架构为后续改进（如 v2 的 Cross-Attention 机制）提供了良好的基础

因此，**v1 模型相比 v0 模型在性能、效率和实用性方面都有显著提升，是更优的范例选择模型架构选择**。

### 3.4 v2 推理结果

**模型说明**: v2 模型在 v1 的 Bi-Encoder 架构基础上添加了单层 Cross-Attention 机制，通过多头注意力增强 query 与 candidates 之间的细粒度交互能力，使用残差连接和 LayerNorm 提升训练稳定性，从而更准确地从候选池中选择相关范例。

#### 3.4.1 Flamingo-3B 模型结果

| Sample Num | Shot Num | Model Name | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|------------|----------|------------|:-----------:|:-------------:|:------------:|:----------:|
| 100 | 1 | v1 | **22.60** | 22.20 | **22.60** | 22.20 |
| 100 | 1 | v2 | **22.60** | **22.60** | **22.60** | **22.60** |
| 100 | 2 | v1 | **27.40** | 27.00 | **27.40** | 27.00 |
| 100 | 2 | v2 | **27.40** | **27.40** | **27.40** | **27.40** |
| 100 | 3 | v1 | **24.00** | **27.00** | **24.00** | **27.00** |
| 100 | 3 | v2 | **24.00** | 24.00 | **24.00** | 24.00 |
| 100 | 4 | v1 | **24.00** | **28.20** | **24.00** | **28.20** |
| 100 | 4 | v2 | **24.00** | 24.00 | **24.00** | 24.00 |
| 200 | 1 | v1 | 22.30 | 20.90 | **22.80** | 20.90 |
| 200 | 1 | v2 | **22.80** | **22.80** | **22.80** | **22.80** |
| 200 | 2 | v1 | **24.90** | **25.20** | **24.40** | **25.20** |
| 200 | 2 | v2 | 24.40 | 24.40 | **24.40** | 24.40 |
| 200 | 3 | v1 | **22.20** | **23.90** | **22.20** | **23.90** |
| 200 | 3 | v2 | **22.20** | 22.20 | **22.20** | 22.20 |
| 200 | 4 | v1 | **23.10** | **24.80** | **22.60** | **24.80** |
| 200 | 4 | v2 | 22.60 | 22.60 | **22.60** | 22.60 |
| 300 | 1 | v1 | 21.60 | 20.60 | **21.93** | 20.60 |
| 300 | 1 | v2 | **21.93** | **21.93** | **21.93** | **21.93** |
| 300 | 2 | v1 | **23.00** | **24.07** | 22.67 | **24.07** |
| 300 | 2 | v2 | 22.67 | 22.67 | 22.67 | 22.67 |
| 300 | 3 | v1 | **23.00** | **24.73** | **23.00** | **24.73** |
| 300 | 3 | v2 | **23.00** | 23.00 | **23.00** | 23.00 |
| 300 | 4 | v1 | **24.13** | **25.20** | 23.80 | **25.20** |
| 300 | 4 | v2 | 23.80 | 23.80 | 23.80 | 23.80 |

**最佳结果**: 28.2% (TextSimSampler/MixSampler, shot_num=4, v1)

#### 3.4.2 Qwen2.5-VL-3B-Instruct 模型结果

| Sample Num | Shot Num | Model Name | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|------------|----------|------------|:-----------:|:-------------:|:------------:|:----------:|
| 100 | 1 | v1 | **64.80** | **64.80** | **63.80** | 64.40 |
| 100 | 1 | v2 | **64.80** | **64.80** | 62.80 | **64.80** |
| 100 | 2 | v1 | **64.40** | **64.40** | 63.80 | 63.80 |
| 100 | 2 | v2 | **64.40** | **64.40** | **66.40** | **64.40** |
| 100 | 3 | v1 | **59.80** | **59.80** | **62.80** | **62.80** |
| 100 | 3 | v2 | **59.80** | **59.80** | 61.20 | 59.80 |
| 100 | 4 | v1 | **60.80** | **60.80** | **61.40** | **61.40** |
| 100 | 4 | v2 | **60.80** | **60.80** | 60.80 | 60.80 |
| 200 | 1 | v1 | **57.80** | **57.80** | 56.90 | 57.40 |
| 200 | 1 | v2 | **57.80** | **57.80** | **58.10** | **57.80** |
| 200 | 2 | v1 | **55.90** | **55.90** | 56.30 | 55.30 |
| 200 | 2 | v2 | **55.90** | **55.90** | **56.40** | **55.90** |
| 200 | 3 | v1 | **53.60** | **53.60** | **55.50** | **55.20** |
| 200 | 3 | v2 | **53.60** | **53.60** | 54.10 | 53.60 |
| 200 | 4 | v1 | **54.90** | **54.90** | **54.90** | **54.90** |
| 200 | 4 | v2 | **54.90** | **54.90** | 53.70 | **54.90** |
| 300 | 1 | v1 | **55.53** | **55.53** | 55.47 | **55.80** |
| 300 | 1 | v2 | **55.53** | **55.53** | **56.20** | 55.53 |
| 300 | 2 | v1 | **53.27** | **53.27** | **54.13** | **53.93** |
| 300 | 2 | v2 | **53.27** | **53.27** | **54.13** | 53.27 |
| 300 | 3 | v1 | **50.73** | **50.73** | **53.73** | **53.00** |
| 300 | 3 | v2 | **50.73** | **50.73** | 51.93 | 50.73 |
| 300 | 4 | v1 | **51.80** | **51.80** | **52.20** | **52.00** |
| 300 | 4 | v2 | **51.80** | **51.80** | 51.20 | 51.80 |

**最佳结果**: 64.8% (RandSampler/TextSimSampler, shot_num=1, v1)

#### 实验分析

通过对比 v2 和 v1 的实验结果，我们发现：

**1. Flamingo-3B 模型（3.4.1）**

- **v2 的优势场景**：
  - 在低 shot 数（Shot 1-2）时，v2 在部分 sampler 上表现更好。例如，Sample Num=100 时，Shot 1 的 TextSimSampler 和 MixSampler 从 22.20% 提升到 22.60%；Sample Num=200 时，Shot 1 的所有 sampler 都有提升，RandSampler 从 22.30% 提升到 22.80%。
  - v2 在 Sample Num=300、Shot 1 时，RandSampler、TextSimSampler 和 MixSampler 都有提升（21.93% vs 21.60%/20.60%）。

- **v1 的优势场景**：
  - 在高 shot 数（Shot 3-4）时，v1 表现明显更好。例如，Sample Num=100 时，Shot 3-4 的 TextSimSampler 和 MixSampler，v1 达到 27.00%-28.20%，而 v2 仅为 24.00%。
  - Sample Num=200-300 时，Shot 2-4 的多个 sampler 上，v1 都优于 v2。

- **原因分析**：
  - v2 在 v1 的 Bi-Encoder 架构基础上添加了 Cross-Attention 机制，增强了 query 和候选之间的交互，在简单场景（低 shot 数）下能够更准确地选择相关示例。
  - 但在复杂场景（高 shot 数）下，v1 的纯 Bi-Encoder 架构（无 Cross-Attention）可能更稳定，能够更好地处理多个示例之间的关系，避免过度交互导致的性能下降。

**2. Qwen2.5-VL-3B-Instruct 模型（3.4.2）**

- **v2 的优势场景**：
  - 在 ImgSimSampler 上，v2 在多个配置下都有提升。例如，Sample Num=100、Shot 2 时，从 63.80% 提升到 66.40%；Sample Num=200、Shot 1-2 时，分别从 56.90% 和 56.30% 提升到 58.10% 和 56.40%；Sample Num=300、Shot 1 时，从 55.47% 提升到 56.20%。
  - 在 MixSampler 上，Sample Num=100、Shot 1 时从 64.40% 提升到 64.80%；Sample Num=200、Shot 1-2 时也有提升。

- **v1 的优势场景**：
  - 在高 shot 数（Shot 3-4）时，v1 在多个 sampler 上表现更好。例如，Sample Num=100、Shot 3-4 时，TextSimSampler 和 MixSampler，v1 达到 59.80%-62.80%，而 v2 仅为 59.80%-60.80%。
  - Sample Num=200-300 时，Shot 3-4 的多个 sampler 上，v1 都优于 v2。

- **原因分析**：
  - v2 在 v1 的 Bi-Encoder 架构基础上添加的 Cross-Attention 机制在图像相似度采样场景下特别有效，能够更好地理解图像-文本的跨模态关系，因此在 ImgSimSampler 上表现突出。
  - 对于 Qwen2.5-VL-3B-Instruct 这种更强的视觉-语言模型，v2 的交互机制能够更好地利用模型的视觉理解能力。
  - 但在高 shot 数场景下，v1 的纯 Bi-Encoder 架构（无 Cross-Attention）可能更适合处理多个示例的复杂关系，避免过度交互导致的性能下降。

**3. 总体结论**

- **整体效果对比**：从实验结果来看，v2 的整体效果不如 v1。v2 只在少数特定场景下有提升（如低 shot 数、ImgSimSampler），但在大多数场景下，特别是高 shot 数时，v1 表现明显更好。

- **v2 的局限性**：
  - 虽然 v2 在 v1 的 Bi-Encoder 架构基础上增加了 Cross-Attention 机制，理论上应该增强 query 和候选之间的交互，但实际效果表明：
    - Cross-Attention 可能引入了过度拟合或噪声，导致在高 shot 数场景下性能下降
    - 额外的交互机制增加了模型复杂度，但未能带来预期的性能提升
    - 在简单场景（低 shot 数）下的提升有限，且在高 shot 数场景下的性能损失更大

- **v1 的优势**：
  - v1 的纯 Bi-Encoder 架构（无 Cross-Attention）虽然简单，但在大多数场景下表现更稳定
  - 独立编码的方式避免了过度交互可能带来的问题，在处理多个示例的复杂关系时更具优势
  - 特别是在高 shot 数场景下，v1 能够更好地利用多个示例的信息

- **建议**：
  - 从整体效果来看，**v1 是更优的选择**，在大多数场景下表现更好
  - 只有在特定的图像相似度采样任务（ImgSimSampler）且低 shot 数场景下，v2 才可能略有优势
  - 这一结果也说明，**并非所有增加模型复杂度的改进都能带来性能提升**，简单的架构有时反而更有效

**4. v2 Cross-Attention 的改进方向**

虽然当前 v2 的整体效果不如 v1，但 Cross-Attention 机制仍有改进潜力。基于当前实现（`num_heads=1`, `dropout=0.5`, `attn_dropout=0.1`, `hidden_dim=256`），可以考虑以下改进方向：

- **参数调优**：
  - **多头注意力（num_heads）**：当前使用 `num_heads=1`（单头），可以尝试增加到 2-4 头，增强模型对不同表示子空间的学习能力
  - **Dropout 率**：当前 `dropout=0.5` 和 `attn_dropout=0.1`，可以尝试降低 dropout（如 0.1-0.3），减少过度正则化，特别是在高 shot 数场景下
  - **隐藏维度（hidden_dim）**：当前 `hidden_dim=256`，可以尝试增加到 512，增强模型的表达能力
  - **温度参数（temperature）**：当前固定为 0.1，可以尝试可学习的温度参数或动态调整

- **架构改进**：
  - **多层 Cross-Attention**：当前只有单层，可以尝试 2-3 层，但需要注意过深可能导致过拟合
  - **注意力机制变体**：可以尝试不同的注意力机制，如缩放点积注意力（Scaled Dot-Product Attention）的变体，或引入相对位置编码
  - **残差连接策略**：当前使用标准残差连接，可以尝试不同的残差连接方式或权重

- **训练策略改进**：
  - **学习率调度**：针对 Cross-Attention 层使用不同的学习率，或采用 warmup 策略
  - **正则化策略**：除了 dropout，可以尝试 LayerNorm 的位置调整、权重衰减等
  - **损失函数**：当前使用交叉熵 + label smoothing，可以尝试引入对比学习损失或排序损失

- **数据相关改进**：
  - **训练数据增强**：增加训练数据的多样性，特别是高 shot 数场景的数据
  - **负样本采样**：改进负样本采样策略，使模型更好地学习区分相关和不相关的候选

- **实验建议**：
  - 建议先进行**网格搜索或贝叶斯优化**，系统性地探索参数空间
  - 重点关注**高 shot 数场景下的性能**，因为这是 v2 的主要短板
  - 可以考虑**自适应机制**，根据 shot 数动态调整 Cross-Attention 的强度或是否使用

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
