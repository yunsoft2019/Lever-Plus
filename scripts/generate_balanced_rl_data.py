#!/usr/bin/env python3
"""
生成平衡的RL训练数据

策略：
1. 从VQAv2训练集中随机采样Query
2. 对每个Query，从全局候选池中采样多个候选组合
3. 评测这些候选组合，找出同时有正负样本的Query
4. 只保留正负样本比例在40%-60%之间的Query
5. 目标：生成足够多的平衡Query用于训练

使用方法：
    python scripts/generate_balanced_rl_data.py --gpu 0 --num_queries 10000 --eval_per_query 20
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
from omegaconf import OmegaConf
from lever_lm.load_ds_utils import load_vqav2_ds
from lever_lm.models.v3.generate_rl_data import (
    compute_vqa_accuracy,
    build_vqa_prompt_and_generate,
)
from open_mmicl.interface import Qwen2VLInterface
from open_mmicl.metrics.vqa_metrics import VQA


def load_dataset(dataset_config_path):
    """加载VQAv2数据集"""
    with open(dataset_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cfg = OmegaConf.create(config)
    OmegaConf.resolve(cfg)
    
    ds = load_vqav2_ds(
        version=cfg.version,
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        train_coco_dataset_root=cfg.train_coco_dataset_root,
        val_coco_dataset_root=cfg.val_coco_dataset_root,
        split='train'
    )
    return ds


def sample_candidate_pairs(query_idx, cand_emb, query_emb, num_pairs=20, exclude_indices=None):
    """
    从全局候选池中采样候选pair
    
    策略：从不同相似度区间采样，增加多样性
    """
    if exclude_indices is None:
        exclude_indices = set()
    exclude_indices.add(query_idx)
    
    # 计算相似度
    q_emb = query_emb[query_idx]
    q_emb_norm = F.normalize(q_emb.unsqueeze(0), p=2, dim=1)
    c_emb_norm = F.normalize(cand_emb, p=2, dim=1)
    similarities = torch.mm(q_emb_norm, c_emb_norm.t()).squeeze(0)
    
    # 排除自身
    similarities[query_idx] = -1.0
    for idx in exclude_indices:
        if idx < len(similarities):
            similarities[idx] = -1.0
    
    # 按相似度排序
    sorted_indices = torch.argsort(similarities, descending=True).tolist()
    
    # 从不同区间采样
    N = len(sorted_indices)
    num_intervals = 10
    interval_size = N // num_intervals
    
    sampled_indices = []
    for i in range(num_intervals):
        start = i * interval_size
        end = min((i + 1) * interval_size, N)
        interval = sorted_indices[start:end]
        # 从每个区间采样2个
        if len(interval) >= 2:
            sampled = random.sample(interval, min(4, len(interval)))
            sampled_indices.extend(sampled)
    
    # 生成pair
    pairs = []
    random.shuffle(sampled_indices)
    for i in range(0, len(sampled_indices) - 1, 2):
        if len(pairs) >= num_pairs:
            break
        pair = tuple(sorted([sampled_indices[i], sampled_indices[i+1]]))
        pairs.append(pair)
    
    return pairs


def evaluate_pair(vqa_interface, query_item, dataset, pointer, 
                  vqa_train_cache, vqa_val_cache,
                  train_ques_path, train_ann_path,
                  val_ques_path, val_ann_path):
    """评测一个候选pair"""
    examples = [dataset[idx] for idx in pointer]
    
    query_image = query_item.get('image')
    query_question = query_item.get('question', '')
    
    try:
        result = build_vqa_prompt_and_generate(
            interface=vqa_interface,
            image=query_image,
            question=query_question,
            ex1=examples[0],
            ex2=examples[1] if len(examples) > 1 else None,
        )
        
        if isinstance(result, dict):
            pred_answer = result.get("pred_answer", "")
        else:
            pred_answer = str(result) if result else ""
    except Exception as e:
        pred_answer = ""
    
    # 计算准确率
    gt_answers = query_item.get('answers', [])
    question_id = query_item.get('question_id', '')
    
    used_file_metric = False
    if train_ques_path and train_ann_path:
        try:
            correct, acc_score, used_file_metric = compute_vqa_accuracy(
                pred_answer=pred_answer,
                ground_truth_answers=gt_answers,
                question_id=question_id,
                val_ques_path=train_ques_path,
                val_ann_path=train_ann_path,
                vqa_cache=vqa_train_cache
            )
        except:
            used_file_metric = False
    
    if not used_file_metric and val_ques_path and val_ann_path:
        try:
            correct, acc_score, used_file_metric = compute_vqa_accuracy(
                pred_answer=pred_answer,
                ground_truth_answers=gt_answers,
                question_id=question_id,
                val_ques_path=val_ques_path,
                val_ann_path=val_ann_path,
                vqa_cache=vqa_val_cache
            )
        except:
            used_file_metric = False
    
    if not used_file_metric:
        acc_score = 0.0
    
    return {
        "pointer": list(pointer),
        "vqa_pred_answer": pred_answer,
        "vqa_acc_score": float(acc_score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0, help="起始Query索引")
    parser.add_argument("--end_idx", type=int, default=1000, help="结束Query索引")
    parser.add_argument("--eval_per_query", type=int, default=20, help="每个Query评测的候选数")
    parser.add_argument("--min_pos_ratio", type=float, default=0.40, help="最小正样本比例")
    parser.add_argument("--max_pos_ratio", type=float, default=0.60, help="最大正样本比例")
    parser.add_argument("--output_suffix", type=str, default="", help="输出文件后缀")
    args = parser.parse_args()
    
    random.seed(42)
    
    # 注意：不要在这里设置CUDA_VISIBLE_DEVICES，由shell脚本控制
    # 如果shell已经设置了CUDA_VISIBLE_DEVICES，则使用cuda:0
    # 否则使用指定的GPU
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    
    # 路径配置
    vqav2_path = os.path.join(PROJECT_ROOT, "datasets/vqav2")
    output_path = os.path.join(PROJECT_ROOT, f"results/vqav2/generated_data/rl_data_balanced_{args.start_idx}_{args.end_idx}{args.output_suffix}.json")
    
    query_emb_path = os.path.join(PROJECT_ROOT, "results/vqav2/cache/query_embeddings.pt")
    cand_emb_path = os.path.join(PROJECT_ROOT, "results/vqav2/cache/candidate_embeddings.pt")
    dataset_config = os.path.join(PROJECT_ROOT, "configs/dataset/vqav2_local.yaml")
    
    train_ques_path = os.path.join(vqav2_path, "v2_OpenEnded_mscoco_train2014_questions.json")
    train_ann_path = os.path.join(vqav2_path, "v2_mscoco_train2014_annotations.json")
    val_ques_path = os.path.join(vqav2_path, "v2_OpenEnded_mscoco_val2014_questions.json")
    val_ann_path = os.path.join(vqav2_path, "v2_mscoco_val2014_annotations.json")
    
    print("=" * 60)
    print("生成平衡的RL训练数据")
    print("=" * 60)
    print(f"Query范围: {args.start_idx} - {args.end_idx}")
    print(f"每个Query评测数: {args.eval_per_query}")
    print(f"正样本比例范围: {args.min_pos_ratio*100:.0f}%-{args.max_pos_ratio*100:.0f}%")
    print("=" * 60)
    
    # 加载数据
    print("\n加载embeddings...")
    query_emb = torch.load(query_emb_path, map_location='cpu')
    cand_emb = torch.load(cand_emb_path, map_location='cpu')
    print(f"  Query: {query_emb.shape}, Candidate: {cand_emb.shape}")
    
    print("加载数据集...")
    dataset = load_dataset(dataset_config)
    print(f"  数据集大小: {len(dataset)}")
    
    print("加载VQA模型...")
    vqa_interface = Qwen2VLInterface(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        load_from_local=False,
        precision="bf16",
        device=device,
        prompt_template="Question: {question} Short answer:",
        column_token_map={"question": "question"},
        instruction="",
        icd_join_char="\n",
        image_field="image",
        label_field="answer",
    )
    
    print("加载VQA评测缓存...")
    vqa_train_cache = VQA(train_ann_path, train_ques_path)
    vqa_val_cache = VQA(val_ann_path, val_ques_path)
    
    # 直接使用指定范围的Query索引
    total_queries = len(dataset)
    end_idx = min(args.end_idx, total_queries)
    query_indices = list(range(args.start_idx, end_idx))
    
    print(f"\n处理Query索引: {args.start_idx} 到 {end_idx}")
    print(f"共 {len(query_indices)} 个Query")
    
    balanced_data = {
        '_meta': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_idx': args.start_idx,
            'end_idx': end_idx,
            'num_queries_processed': len(query_indices),
            'eval_per_query': args.eval_per_query,
            'pos_ratio_range': [args.min_pos_ratio, args.max_pos_ratio],
        }
    }
    
    balanced_count = 0
    total_eval = 0
    
    for query_idx in tqdm(query_indices, desc="Processing"):
        # 获取Query信息
        sample = dataset[query_idx]
        query_item = {
            'image': sample.get('image'),
            'question': sample.get('question', ''),
            'answers': sample.get('answers', []),
            'question_id': sample.get('question_id', query_idx),
        }
        
        # 采样候选pair
        pairs = sample_candidate_pairs(
            query_idx, cand_emb, query_emb, 
            num_pairs=args.eval_per_query
        )
        
        if not pairs:
            continue
        
        # 评测
        candidates = []
        for pointer in pairs:
            result = evaluate_pair(
                vqa_interface, query_item, dataset, pointer,
                vqa_train_cache, vqa_val_cache,
                train_ques_path, train_ann_path,
                val_ques_path, val_ann_path
            )
            candidates.append(result)
            total_eval += 1
        
        # 检查是否平衡
        scores = [c['vqa_acc_score'] for c in candidates]
        pos_count = sum(1 for s in scores if s > 0)
        neg_count = len(scores) - pos_count
        
        if pos_count > 0 and neg_count > 0:
            pos_ratio = pos_count / len(scores)
            if args.min_pos_ratio <= pos_ratio <= args.max_pos_ratio:
                balanced_count += 1
                balanced_data[str(query_idx)] = {
                    'query': {
                        'query_id': query_idx,
                        'question_id': sample.get('question_id', query_idx),
                        'question': sample.get('question', ''),
                        'gt_answers_raw': sample.get('answers', []),
                    },
                    'pointer_candidates': candidates,
                }
        
        # 每找到一个就保存（防止任务被终止丢失数据）
        if balanced_count > 0:
            with open(output_path, 'w') as f:
                json.dump(balanced_data, f, indent=2, ensure_ascii=False)
    
    # 最终保存
    with open(output_path, 'w') as f:
        json.dump(balanced_data, f, indent=2, ensure_ascii=False)
    
    # 统计
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"处理Query数: {len(query_indices)}")
    print(f"总评测数: {total_eval}")
    print(f"平衡Query数: {balanced_count} ({100*balanced_count/len(query_indices):.1f}%)")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
