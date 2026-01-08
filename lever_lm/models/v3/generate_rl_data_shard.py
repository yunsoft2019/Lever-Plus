#!/usr/bin/env python
"""
分片生成 RL 数据，支持断点续传
用于多 GPU 并行生成
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from lever_lm.models.v3.generate_rl_data import (
    load_sft_model,
    load_vqa_model,
    generate_rl_data_for_query
)
from open_mmicl.metrics.vqa_metrics import VQA

# 导入根目录的 utils.py
import importlib.util
_utils_path = os.path.join(project_root, 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
load_ds = _root_utils.load_ds


def get_dataset_config(dataset_name: str, project_root: str):
    """根据数据集名称获取配置"""
    dataset_lower = dataset_name.lower()
    
    if "vqav2" in dataset_lower:
        vqav2_dir = os.path.join(project_root, "datasets", "vqav2")
        vqav2_hf_dir = os.path.join(vqav2_dir, "vqav2_hf")
        return {
            "train_path": os.path.join(vqav2_hf_dir, "vqav2_mscoco_train2014.json"),
            "val_path": os.path.join(vqav2_hf_dir, "vqav2_mscoco_val2014.json"),
            "train_coco_root": os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "train2014"),
            "val_coco_root": os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "val2014"),
            "val_ques_path": os.path.join(vqav2_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
            "val_ann_path": os.path.join(vqav2_dir, "v2_mscoco_val2014_annotations.json"),
            "train_ques_path": os.path.join(vqav2_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
            "train_ann_path": os.path.join(vqav2_dir, "v2_mscoco_train2014_annotations.json"),
            "dataset_name": "vqav2",
            "sample_num": 5000,
        }
    else:  # OKVQA
        okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
        okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
        return {
            "train_path": os.path.join(okvqa_hf_dir, "vqav2_mscoco_train2014.json"),
            "val_path": os.path.join(okvqa_hf_dir, "vqav2_mscoco_val2014.json"),
            "train_coco_root": os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "train2014"),
            "val_coco_root": os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "val2014"),
            "val_ques_path": os.path.join(okvqa_dir, "OpenEnded_mscoco_val2014_questions.json"),
            "val_ann_path": os.path.join(okvqa_dir, "mscoco_val2014_annotations.json"),
            "train_ques_path": os.path.join(okvqa_dir, "OpenEnded_mscoco_train2014_questions.json"),
            "train_ann_path": os.path.join(okvqa_dir, "mscoco_train2014_annotations.json"),
            "dataset_name": "okvqa",
            "sample_num": 800,
        }


def main():
    parser = argparse.ArgumentParser(description="分片生成 RL 数据")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--sampler", type=str, default="rand_sampler", help="采样器")
    parser.add_argument("--beam_model", type=str, default="qwen2.5_vl_3B", help="Beam 模型")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID")
    parser.add_argument("--start_idx", type=int, required=True, help="起始 query 索引")
    parser.add_argument("--end_idx", type=int, required=True, help="结束 query 索引")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"使用设备: {device}")
    
    # 获取数据集配置
    ds_config = get_dataset_config(args.dataset, str(project_root))
    dataset_name = ds_config["dataset_name"]
    sample_num = ds_config["sample_num"]
    
    # 转换 sampler 和 model 名称
    sampler_map = {
        "rand_sampler": "RandSampler",
        "text_sim_sampler": "TextSimSampler",
        "img_sim_sampler": "ImgSimSampler",
        "mix_sampler": "MixSampler",
    }
    sampler_name = sampler_map.get(args.sampler, args.sampler)
    
    model_map = {
        "qwen2.5_vl_3B": "Qwen2_5-VL-3B-Instruct",
        "qwen2_5_vl_3B": "Qwen2_5-VL-3B-Instruct",
        "flamingo_3B": "flamingo_3B",
    }
    model_name = model_map.get(args.beam_model, args.beam_model)
    
    # 构建文件路径
    beam_data_file = f"vqa-{dataset_name}-{model_name}-{sampler_name}-scorer:infoscore-construct_order:left-beam_size:5-max_shot:2-candidate_num:64-sample_num:{sample_num}.json"
    beam_data_path = os.path.join(project_root, "results", dataset_name, "generated_data", beam_data_file)
    query_emb_path = os.path.join(project_root, "results", dataset_name, "cache", "query_embeddings.pt")
    cand_emb_path = os.path.join(project_root, "results", dataset_name, "cache", "candidate_embeddings.pt")
    
    # 查找 SFT checkpoint
    model_name_safe = model_name.replace("-", "_").replace(".", "_")
    ckpt_pattern = f"{model_name_safe}_{sampler_name}_infoscore_left_beam5_shot2_cand64_sample{sample_num}"
    ckpt_dir = os.path.join(project_root, "results", dataset_name, "model_cpk", "v2")
    
    sft_ckpt = None
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if sampler_name in f and f.endswith(".ckpt"):
                sft_ckpt = os.path.join(ckpt_dir, f)
                break
    
    if not sft_ckpt:
        print(f"错误: 未找到 SFT checkpoint in {ckpt_dir}")
        return
    
    print(f"SFT Checkpoint: {sft_ckpt}")
    print(f"Beam Data: {beam_data_path}")
    print(f"Query Embeddings: {query_emb_path}")
    print(f"Candidate Embeddings: {cand_emb_path}")
    print(f"处理范围: {args.start_idx} - {args.end_idx}")
    
    # 检查断点
    completed_queries = set()
    if os.path.exists(args.output_file):
        print(f"发现已有输出文件，加载断点...")
        with open(args.output_file, "r") as f:
            existing_data = json.load(f)
        completed_queries = set(existing_data.keys())
        print(f"已完成 {len(completed_queries)} 个 query")
    else:
        existing_data = {}
    
    # 创建配置
    cfg = OmegaConf.create({
        "dataset": {
            "name": args.dataset,
            "version": "local",
            "train_path": ds_config["train_path"],
            "val_path": ds_config["val_path"],
            "train_coco_dataset_root": ds_config["train_coco_root"],
            "val_coco_dataset_root": ds_config["val_coco_root"],
        },
        "infer_model": {
            "name": "Qwen2.5-VL-3B-Instruct",
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "load_from_local": False,
            "precision": "bf16",
            "icd_join_char": "<|endofchunk|>",
            "system_prompt": "In the upcoming task, you will see four sets of dialogues, each containing two roles: user and assistant. The user is the questioner, who provides an image and asks a question based on it; the assistant is the responder, who answers according to the image and question provided by the user. Afterward, you will receive an image and a question from the user. Please act as the assistant and answer based on the four previous dialogue sets and your own knowledge. Strictly follow the answering format: if the examples use only one or two keywords, your reply must also use only one or two keywords; if the examples contain no more than three tokens, your reply must not exceed three tokens either.",
        },
        "task": {
            "task_name": "vqa",
            "template": "Question:<Q> Short answer:<A>",
            "column_token_map": {"question": "<Q>", "answer": "<A>"},
            "instruction": "",
            "image_field": "image",
            "output_column": "answer",
        },
        "precision": "bf16",
    })
    
    # 加载模型
    print("加载 SFT 模型...")
    sft_model = load_sft_model(sft_ckpt, device)
    
    print("加载 VQA 模型...")
    vqa_model = load_vqa_model(args.beam_model, device, cfg=cfg)
    
    # 加载数据
    print("加载 beam 数据...")
    with open(beam_data_path, "r") as f:
        beam_data = json.load(f)
    
    print("加载数据集...")
    ds = load_ds(cfg)
    train_ds = ds["train"]
    
    print("加载 embeddings...")
    query_embeddings = torch.load(query_emb_path, map_location=device)
    candidate_embeddings = torch.load(cand_emb_path, map_location=device)
    
    # 构建 candidate 索引
    candidate_indices = list(range(len(candidate_embeddings)))
    candidate_pool = [train_ds[idx] for idx in candidate_indices]
    
    # 预加载 VQA 对象
    vqa_val_cache = None
    if os.path.exists(ds_config["val_ann_path"]) and os.path.exists(ds_config["val_ques_path"]):
        try:
            print("预加载 VQA 标注文件...")
            vqa_val_cache = VQA(ds_config["val_ann_path"], ds_config["val_ques_path"])
        except Exception as e:
            print(f"警告: 加载 VQA 对象失败: {e}")
    
    # 加载生成参数
    task_config_path = os.path.join(project_root, "configs/task/vqa.yaml")
    if os.path.exists(task_config_path):
        task_cfg = OmegaConf.load(task_config_path)
        generation_kwargs = OmegaConf.to_container(task_cfg.gen_args) if hasattr(task_cfg, 'gen_args') else {}
    else:
        generation_kwargs = {"max_new_tokens": 5, "num_beams": 3, "length_penalty": 0.0, "min_new_tokens": 0}
    
    print(f"生成参数: {generation_kwargs}")
    
    # 获取要处理的 query 列表
    all_query_ids = list(beam_data.keys())
    query_ids_to_process = all_query_ids[args.start_idx:args.end_idx]
    
    print(f"开始生成 RL 数据...")
    print(f"  - 总 Query 数: {len(query_ids_to_process)}")
    print(f"  - 已完成: {len(completed_queries)}")
    print(f"  - 待处理: {len(query_ids_to_process) - len(completed_queries)}")
    
    # 生成参数
    num_beams = 5
    temps = (1.0, 1.3)
    num_samples_per_temp = 2
    num_random = 1
    num_retrieval = 5
    
    rl_data = existing_data.copy()
    save_interval = 10  # 每处理 10 个 query 保存一次
    
    for i, query_id in enumerate(tqdm(query_ids_to_process, desc=f"GPU {args.gpu_id}")):
        if query_id in completed_queries:
            continue
        
        try:
            # 获取 query 数据
            query_info = beam_data[query_id]
            query_idx = int(query_id)
            
            # 生成单个 query 的 RL 数据
            query_rl_data = generate_rl_data_for_query(
                sft_model=sft_model,
                vqa_model=vqa_model,
                query_id=query_id,
                query_info=query_info,
                query_idx=query_idx,
                query_embeddings=query_embeddings,
                candidate_embeddings=candidate_embeddings,
                candidate_indices=candidate_indices,
                candidate_pool=candidate_pool,
                dataset=train_ds,
                num_beams=num_beams,
                temps=temps,
                num_samples_per_temp=num_samples_per_temp,
                num_random=num_random,
                num_retrieval=num_retrieval,
                device=device,
                generation_kwargs=generation_kwargs,
                vqa_val_cache=vqa_val_cache,
                strict_eval=True,
            )
            
            rl_data[query_id] = query_rl_data
            
            # 定期保存
            if (i + 1) % save_interval == 0:
                with open(args.output_file, "w") as f:
                    json.dump(rl_data, f, indent=2)
                print(f"  已保存 {len(rl_data)} 个 query")
                
        except Exception as e:
            print(f"警告: query {query_id} 处理失败: {e}")
            continue
    
    # 最终保存
    with open(args.output_file, "w") as f:
        json.dump(rl_data, f, indent=2)
    
    print(f"========================================")
    print(f"GPU {args.gpu_id} 完成！")
    print(f"输出文件: {args.output_file}")
    print(f"总 Query 数: {len(rl_data)}")
    print(f"========================================")


if __name__ == "__main__":
    main()
