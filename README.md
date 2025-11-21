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

#### 使用 Qwen2.5-VL-3B-Instruct 模型（后台运行）

```bash
# 随机采样器（RandSampler）
bash scripts/run_generate_data_background.sh vqa okvqa_local "[0]" rand_sampler qwen2.5_vl_3B

# 文本相似度采样器（TextSimSampler）
bash scripts/run_generate_data_background.sh vqa okvqa_local "[1]" text_sim_sampler qwen2.5_vl_3B

# 图片相似度采样器（ImgSimSampler）
bash scripts/run_generate_data_background.sh vqa okvqa_local "[2]" img_sim_sampler qwen2.5_vl_3B

# 混合采样器（MixSampler）
bash scripts/run_generate_data_background.sh vqa okvqa_local "[3]" mix_sampler qwen2.5_vl_3B
```

#### 后台任务管理命令

```bash
# 列出所有运行中的任务
bash scripts/manage_background_tasks.sh list

# 停止指定PID的任务
bash scripts/manage_background_tasks.sh stop <pid>

# 停止所有generate_data任务
bash scripts/manage_background_tasks.sh stop-all

# 查看所有日志文件
bash scripts/manage_background_tasks.sh logs
```

### 2.2 训练

**参数说明**: `task dataset gpu_id lever_lm sampler [beam_model]`

- `gpu_id`: GPU 编号，例如 0 表示使用 GPU 0，1 表示使用 GPU 1（默认: 0）
- `beam_model` 可选值: `flamingo_3B` (默认) 或 `qwen2.5_vl_3B`
- **注意**: `beam_model` 必须与生成数据时使用的模型一致

#### 使用 Flamingo-3B 生成的束搜索数据训练

```bash
# 随机采样器（使用 GPU 0）
bash scripts/train_lever_lm.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B

# 文本相似度采样器（使用 GPU 1）
bash scripts/train_lever_lm.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler flamingo_3B

# 图片相似度采样器（使用 GPU 2）
bash scripts/train_lever_lm.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler flamingo_3B

# 混合采样器（使用 GPU 3）
bash scripts/train_lever_lm.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler flamingo_3B
```

#### 使用 Qwen2.5-VL-3B-Instruct 生成的束搜索数据训练

```bash
# 随机采样器（使用 GPU 0）
bash scripts/train_lever_lm.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

# 文本相似度采样器（使用 GPU 1）
bash scripts/train_lever_lm.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler qwen2.5_vl_3B

# 图片相似度采样器（使用 GPU 2）
bash scripts/train_lever_lm.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler qwen2.5_vl_3B

# 混合采样器（使用 GPU 3）
bash scripts/train_lever_lm.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler qwen2.5_vl_3B
```

### 2.3 推理

**参数说明**: `task dataset device lever_lm sampler [beam_model]`

- `device`: GPU 编号，例如 0 表示使用 GPU 0，1 表示使用 GPU 1（默认: 0）
- `beam_model` 可选值: `flamingo_3B` (默认) 或 `qwen2.5_vl_3B`
- **注意**: `beam_model` 必须与训练时使用的模型一致，用于选择对应的检查点文件
- **注意**: 推理时批量大小固定为1，避免批处理时的图像数量不匹配问题

#### 使用 Flamingo-3B 训练的模型进行推理

```bash
# 随机采样器（RandSampler）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler flamingo_3B

# 文本相似度采样器（TextSimSampler）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text text_sim_sampler flamingo_3B

# 图片相似度采样器（ImgSimSampler）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text img_sim_sampler flamingo_3B

# 混合采样器（MixSampler）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text mix_sampler flamingo_3B
```

#### 使用 Qwen2.5-VL-3B-Instruct 训练的模型进行推理

```bash
# 随机采样器（RandSampler）
bash scripts/inference.sh vqa okvqa_local 0 query_img_text_icd_img_text rand_sampler qwen2.5_vl_3B

# 文本相似度采样器（TextSimSampler）
bash scripts/inference.sh vqa okvqa_local 1 query_img_text_icd_img_text text_sim_sampler qwen2.5_vl_3B

# 图片相似度采样器（ImgSimSampler）
bash scripts/inference.sh vqa okvqa_local 2 query_img_text_icd_img_text img_sim_sampler qwen2.5_vl_3B

# 混合采样器（MixSampler）
bash scripts/inference.sh vqa okvqa_local 3 query_img_text_icd_img_text mix_sampler qwen2.5_vl_3B
```

## 3. 推理结果

### 3.1 Flamingo-3B 模型结果

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | **23.04**   | 19.98          | 21.05         | 20.63      |
| 2        | 20.97       | 21.89          | **23.06**     | 20.94      |
| 3        | 23.29       | 22.94          | **23.63**     | 20.31      |
| 4        | **25.28**   | 24.14          | 24.59         | 22.87      |
| 6        | **24.45**   | 24.0           | 23.93         | 23.25      |
| 8        | **24.68**   | 24.11          | 24.06         | 24.24      |

**最佳结果**: 25.28% (RandSampler, shot_num=4)

### 3.2 Qwen2.5-VL-3B-Instruct 模型结果

| Shot Num | RandSampler | TextSimSampler | ImgSimSampler | MixSampler |
|----------|-------------|----------------|---------------|------------|
| 1        | **52.04**   | 48.91          | 48.60         | 50.19      |
| 2        | 49.76       | 44.66          | 43.98         | **46.36**  |
| 3        | 48.06       | 43.54          | 42.55         | 45.36      |
| 4        | 47.60       | 41.76          | 42.12         | 44.08      |
| 6        | 46.55       | 42.90          | 42.39         | 44.26      |
| 8        | 46.52       | 42.60          | 42.53         | 43.65      |

**最佳结果**: 52.04% (RandSampler, shot_num=1)

### 3.3 结果说明

- **数据集**: OKVQA
- **训练参数**: infoscore_left_beam5_shot2_cand64_sample800
- **Flamingo-3B**: 最佳配置为 RandSampler + shot_num=4，准确率 25.28%
- **Qwen2.5-VL-3B-Instruct**: 最佳配置为 RandSampler + shot_num=1，准确率 52.04%
