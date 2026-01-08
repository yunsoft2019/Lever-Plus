#!/usr/bin/env python3
"""
VQAv2 RL数据全局搜索补全脚本

核心思路：
1. 在全局443757个候选中搜索，而不是只在64个候选中
2. 对 All-Zero query：在全局中找不同相似度区间的候选组合
3. 对 All-One query：在全局中找相似度较低的候选组合（更容易找到负样本）
4. 目标：每个query都有正负样本，正样本比例约55%

使用方法：
    python scripts/augment_vqav2_global_search.py --gpu 0
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

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lever_lm.models.v3.generate_rl_data import (
    compute_vqa_accuracy,
    build_vqa_prompt_and_generate,
    load_vqa_model
)
from open_mmicl.interface import Qwen2VLInterface
from open_mmicl.metrics.vqa_metrics import VQA


def load_dataset(dataset_config_path):
    """加载VQAv2数据集"""
    import yaml
    from omegaconf import OmegaConf
    from lever_lm.load_ds_utils import load_vqav2_ds
    
    with open(dataset_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 解析环境变量
    cfg = OmegaConf.create(config)
    OmegaConf.resolve(cfg)
    
    ds = load_vqav2_ds(
        version=cfg.version,
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        train_coco_dataset_root=cfg.train_coco_dataset_root,
        val_coco_dataset_root=cfg.val_coco_dataset_root,
        split='train'  # 使用训练集
    )
    return ds


def compute_global_similarities(query_emb, cand_emb, device='cuda'):
    """计算query与全局所有候选的相似度"""
    # 在CPU上计算以避免GPU内存问题
    query_emb = query_emb.cpu()
    cand_emb = cand_emb.cpu()
    
    # 归一化
    query_emb_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)  # [1, d]
    cand_emb_norm = F.normalize(cand_emb, p=2, dim=1)  # [N, d]
    
    # 计算cosine相似度
    similarities = torch.mm(query_emb_norm, cand_emb_norm.t()).squeeze(0)  # [N]
    
    return similarities


def sample_candidates_from_intervals(similarities, num_intervals=10, samples_per_interval=5, exclude_indices=None):
    """
    从不同相似度区间采样候选
    
    Args:
        similarities: [N] 相似度分数
        num_intervals: 区间数量
        samples_per_interval: 每个区间采样数量
        exclude_indices: 需要排除的索引集合
    
    Returns:
        sampled_indices: 采样的候选索引列表
    """
    if exclude_indices is None:
        exclude_indices = set()
    
    N = len(similarities)
    
    # 按相似度排序
    sorted_indices = torch.argsort(similarities, descending=True).tolist()
    
    # 过滤掉需要排除的索引
    sorted_indices = [idx for idx in sorted_indices if idx not in exclude_indices]
    
    if len(sorted_indices) == 0:
        return []
    
    # 分成多个区间
    interval_size = len(sorted_indices) // num_intervals
    if interval_size == 0:
        interval_size = 1
    
    sampled_indices = []
    for i in range(num_intervals):
        start = i * interval_size
        end = min((i + 1) * interval_size, len(sorted_indices))
        if start >= len(sorted_indices):
            break
        
        interval_indices = sorted_indices[start:end]
        if len(interval_indices) > 0:
            # 从每个区间随机采样
            num_samples = min(samples_per_interval, len(interval_indices))
            sampled = random.sample(interval_indices, num_samples)
            sampled_indices.extend(sampled)
    
    return sampled_indices


def evaluate_candidate_pair(
    vqa_interface,
    query_item,
    dataset,
    pointer,
    vqa_train_cache,
    vqa_val_cache,
    train_ques_path,
    train_ann_path,
    val_ques_path,
    val_ann_path,
    generation_kwargs=None,
    debug=False
):
    """评测一个候选pair"""
    # 获取示例
    examples = [dataset[idx] for idx in pointer]
    
    if debug:
        print(f"  评测 pointer={pointer}")
        print(f"  ex1 question: {examples[0].get('question', '')[:30]}...")
    
    # 构建prompt并生成答案
    query_image = query_item.get('image')
    query_question = query_item.get('question', '')
    
    try:
        result = build_vqa_prompt_and_generate(
            interface=vqa_interface,
            image=query_image,
            question=query_question,
            ex1=examples[0],
            ex2=examples[1] if len(examples) > 1 else None,
            generation_kwargs=generation_kwargs
        )
        
        if isinstance(result, dict):
            pred_answer = result.get("pred_answer", "")
        elif isinstance(result, str):
            pred_answer = result
        else:
            pred_answer = str(result) if result is not None else ""
        
        if pred_answer is None:
            pred_answer = ""
            
        if debug:
            print(f"  pred_answer: {pred_answer}")
    except Exception as e:
        import traceback
        print(f"Warning: VQA generation failed: {e}")
        traceback.print_exc()
        pred_answer = ""
    
    # 获取ground truth
    gt_answers_raw = query_item.get('answers', [])
    question_id = query_item.get('question_id', str(query_item.get('query_id', '')))
    
    # 计算准确率
    used_file_metric = False
    
    # 先尝试训练集文件
    if train_ques_path and train_ann_path:
        try:
            correct, acc_score, used_file_metric = compute_vqa_accuracy(
                pred_answer=pred_answer,
                ground_truth_answers=gt_answers_raw,
                question_id=question_id,
                val_ques_path=train_ques_path,
                val_ann_path=train_ann_path,
                vqa_cache=vqa_train_cache
            )
        except:
            used_file_metric = False
    
    # 如果训练集失败，尝试验证集
    if not used_file_metric and val_ques_path and val_ann_path:
        try:
            correct, acc_score, used_file_metric = compute_vqa_accuracy(
                pred_answer=pred_answer,
                ground_truth_answers=gt_answers_raw,
                question_id=question_id,
                val_ques_path=val_ques_path,
                val_ann_path=val_ann_path,
                vqa_cache=vqa_val_cache
            )
        except:
            used_file_metric = False
    
    if not used_file_metric:
        correct = 0
        acc_score = 0.0
    
    return {
        "pointer": list(pointer),
        "gen_method": "global_search",
        "vqa_pred_answer": pred_answer,
        "vqa_correct": int(correct),
        "vqa_acc_score": float(acc_score),
    }


def augment_query_global(
    query_id,
    query_item,
    existing_candidates,
    query_emb,
    cand_emb,
    dataset,
    vqa_interface,
    vqa_train_cache,
    vqa_val_cache,
    train_ques_path,
    train_ann_path,
    val_ques_path,
    val_ann_path,
    aug_type,  # 'all0' or 'all1'
    max_eval_budget=50,
    target_positive_ratio=0.55,
    generation_kwargs=None,
    device='cuda'
):
    """
    全局搜索补全一个query
    
    Args:
        query_id: 数据集索引（字符串形式）
        aug_type: 'all0' (需要找正样本) 或 'all1' (需要找负样本)
        target_positive_ratio: 目标正样本比例
    """
    stats = {
        "aug_type": aug_type,
        "eval_count": 0,
        "success": False,
        "new_candidates": []
    }
    
    # 获取已有的pointer索引
    existing_pointers = set()
    existing_indices = set()
    for c in existing_candidates:
        pointer = tuple(sorted(c.get("pointer", [])))
        if pointer:
            existing_pointers.add(pointer)
            for idx in pointer:
                existing_indices.add(idx)
    
    # query_id 就是数据集索引
    query_idx = int(query_id)
    existing_indices.add(query_idx)  # 排除query自身
    
    # 计算全局相似度
    similarities = compute_global_similarities(query_emb, cand_emb, device)
    
    # 排除query自身
    similarities[query_idx] = -1.0
    
    # 根据aug_type选择采样策略
    if aug_type == 'all0':
        # All-Zero: 需要找正样本
        # 策略：从不同相似度区间采样，优先尝试高相似度区间
        sampled_indices = sample_candidates_from_intervals(
            similarities, 
            num_intervals=20,  # 更多区间
            samples_per_interval=10,
            exclude_indices=existing_indices
        )
    else:  # all1
        # All-One: 需要找负样本
        # 策略：优先从低相似度区间采样
        # 反转相似度，让低相似度的排在前面
        inverted_similarities = -similarities
        sampled_indices = sample_candidates_from_intervals(
            inverted_similarities,
            num_intervals=20,
            samples_per_interval=10,
            exclude_indices=existing_indices
        )
    
    # 生成候选pair并评测
    eval_count = 0
    found_target = False
    
    # 随机打乱采样的索引
    random.shuffle(sampled_indices)
    
    # 生成pair组合
    pairs_to_eval = []
    for i in range(len(sampled_indices)):
        for j in range(i + 1, len(sampled_indices)):
            pair = tuple(sorted([sampled_indices[i], sampled_indices[j]]))
            if pair not in existing_pointers:
                pairs_to_eval.append(pair)
    
    # 限制pair数量
    if len(pairs_to_eval) > max_eval_budget * 2:
        pairs_to_eval = random.sample(pairs_to_eval, max_eval_budget * 2)
    
    # 评测
    for pointer in pairs_to_eval:
        if eval_count >= max_eval_budget:
            break
        
        if found_target:
            break
        
        try:
            # 第一个评测时开启debug
            debug_mode = (eval_count == 0)
            candidate_dict = evaluate_candidate_pair(
                vqa_interface=vqa_interface,
                query_item=query_item,
                dataset=dataset,
                pointer=pointer,
                vqa_train_cache=vqa_train_cache,
                vqa_val_cache=vqa_val_cache,
                train_ques_path=train_ques_path,
                train_ann_path=train_ann_path,
                val_ques_path=val_ques_path,
                val_ann_path=val_ann_path,
                generation_kwargs=generation_kwargs,
                debug=debug_mode
            )
            
            candidate_dict["aug_type"] = aug_type
            candidate_dict["gen_method"] = "global_search"
            
            stats["new_candidates"].append(candidate_dict)
            stats["eval_count"] += 1
            eval_count += 1
            existing_pointers.add(pointer)
            
            # 检查是否找到目标
            score = candidate_dict["vqa_acc_score"]
            if aug_type == 'all0' and score > 0:
                found_target = True
                stats["success"] = True
            elif aug_type == 'all1' and score == 0:
                found_target = True
                stats["success"] = True
                
        except Exception as e:
            print(f"Warning: Evaluation failed for pointer {pointer}: {e}")
            continue
    
    return stats["new_candidates"], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_eval_budget_all0", type=int, default=80)
    parser.add_argument("--max_eval_budget_all1", type=int, default=100)
    parser.add_argument("--target_positive_ratio", type=float, default=0.55)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    
    # 路径配置
    vqav2_path = os.path.join(PROJECT_ROOT, "datasets/vqav2")
    input_path = os.path.join(PROJECT_ROOT, "results/vqav2/generated_data/rl_data_merged_full_diverse.json")
    output_path = os.path.join(PROJECT_ROOT, "results/vqav2/generated_data/rl_data_global_augmented.json")
    report_path = os.path.join(PROJECT_ROOT, "results/vqav2/generated_data/rl_data_global_augmented_report.json")
    
    query_emb_path = os.path.join(PROJECT_ROOT, "results/vqav2/cache/query_embeddings.pt")
    cand_emb_path = os.path.join(PROJECT_ROOT, "results/vqav2/cache/candidate_embeddings.pt")
    dataset_config = os.path.join(PROJECT_ROOT, "configs/dataset/vqav2_local.yaml")
    
    train_ques_path = os.path.join(vqav2_path, "v2_OpenEnded_mscoco_train2014_questions.json")
    train_ann_path = os.path.join(vqav2_path, "v2_mscoco_train2014_annotations.json")
    val_ques_path = os.path.join(vqav2_path, "v2_OpenEnded_mscoco_val2014_questions.json")
    val_ann_path = os.path.join(vqav2_path, "v2_mscoco_val2014_annotations.json")
    
    print("=" * 60)
    print("VQAv2 RL数据全局搜索补全")
    print("=" * 60)
    
    # 加载数据
    print("加载RL数据...")
    with open(input_path, 'r') as f:
        rl_data = json.load(f)
    print(f"  总query数: {len(rl_data)}")
    
    # 加载embeddings
    print("加载embeddings...")
    query_emb = torch.load(query_emb_path, map_location='cpu')
    cand_emb = torch.load(cand_emb_path, map_location='cpu')
    print(f"  Query embeddings: {query_emb.shape}")
    print(f"  Candidate embeddings: {cand_emb.shape}")
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset(dataset_config)
    print(f"  数据集大小: {len(dataset)}")
    
    # 加载VQA模型
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
    
    # 加载VQA评测缓存
    print("加载VQA评测缓存...")
    vqa_train_cache = VQA(train_ann_path, train_ques_path)
    vqa_val_cache = VQA(val_ann_path, val_ques_path)
    
    # 分类query
    all0_queries = []
    all1_queries = []
    diverse_queries = []
    
    for qid, qdata in rl_data.items():
        if qid == '_meta':
            continue
        candidates = qdata.get('pointer_candidates', [])
        scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
        unique_scores = set(scores)
        
        if len(unique_scores) == 1:
            score = list(unique_scores)[0]
            if abs(score) < 1e-6:
                all0_queries.append(qid)
            elif abs(score - 1.0) < 1e-6:
                all1_queries.append(qid)
            else:
                diverse_queries.append(qid)
        else:
            diverse_queries.append(qid)
    
    total_queries = len(all0_queries) + len(all1_queries) + len(diverse_queries)
    print(f"\n当前数据分布:")
    print(f"  All-Zero: {len(all0_queries)} ({100*len(all0_queries)/total_queries:.1f}%)")
    print(f"  All-One: {len(all1_queries)} ({100*len(all1_queries)/total_queries:.1f}%)")
    print(f"  Diverse: {len(diverse_queries)} ({100*len(diverse_queries)/total_queries:.1f}%)")
    
    # 统计
    stats = {
        "before": {
            "all0_count": len(all0_queries),
            "all1_count": len(all1_queries),
            "diverse_count": len(diverse_queries)
        },
        "after": {
            "all0_success": 0,
            "all1_success": 0,
            "total_eval_count": 0
        }
    }
    
    # 补全All-Zero queries
    print(f"\n补全 All-Zero queries ({len(all0_queries)}个)...")
    for i, qid in enumerate(tqdm(all0_queries, desc="All-Zero")):
        qdata = rl_data[qid]
        candidates = qdata.get('pointer_candidates', [])
        
        # qid 就是数据集索引
        qid_int = int(qid)
        
        # 获取query embedding
        if qid_int >= len(query_emb):
            print(f"Warning: qid {qid_int} 超出 query_emb 范围")
            continue
        q_emb = query_emb[qid_int]
        
        # 从RL数据中获取query信息
        query_info = qdata.get('query', {})
        
        # 构建query_item
        query_item = {
            'image': dataset[qid_int].get('image') if qid_int < len(dataset) else None,
            'question': query_info.get('question', ''),
            'answers': query_info.get('gt_answers_raw', []),
            'question_id': query_info.get('question_id', qid)
        }
        
        if i == 0:
            print(f"\n  第一个query: qid={qid}, question={query_item['question'][:50]}...")
        
        new_candidates, aug_stats = augment_query_global(
            query_id=qid,
            query_item=query_item,
            existing_candidates=candidates,
            query_emb=q_emb,
            cand_emb=cand_emb,
            dataset=dataset,
            vqa_interface=vqa_interface,
            vqa_train_cache=vqa_train_cache,
            vqa_val_cache=vqa_val_cache,
            train_ques_path=train_ques_path,
            train_ann_path=train_ann_path,
            val_ques_path=val_ques_path,
            val_ann_path=val_ann_path,
            aug_type='all0',
            max_eval_budget=args.max_eval_budget_all0,
            target_positive_ratio=args.target_positive_ratio,
            device=device
        )
        
        if new_candidates:
            rl_data[qid]['pointer_candidates'].extend(new_candidates)
        
        if aug_stats.get("success"):
            stats["after"]["all0_success"] += 1
        stats["after"]["total_eval_count"] += aug_stats.get("eval_count", 0)
        
        if i == 0:
            print(f"  第一个query完成: success={aug_stats.get('success')}, eval_count={aug_stats.get('eval_count')}")
    
    # 补全All-One queries
    print(f"\n补全 All-One queries ({len(all1_queries)}个)...")
    for qid in tqdm(all1_queries, desc="All-One"):
        qdata = rl_data[qid]
        candidates = qdata.get('pointer_candidates', [])
        
        # qid 就是数据集索引
        qid_int = int(qid)
        
        # 获取query embedding
        if qid_int >= len(query_emb):
            print(f"Warning: qid {qid_int} 超出 query_emb 范围")
            continue
        q_emb = query_emb[qid_int]
        
        # 从RL数据中获取query信息
        query_info = qdata.get('query', {})
        
        # 构建query_item
        query_item = {
            'image': dataset[qid_int].get('image') if qid_int < len(dataset) else None,
            'question': query_info.get('question', ''),
            'answers': query_info.get('gt_answers_raw', []),
            'question_id': query_info.get('question_id', qid)
        }
        
        new_candidates, aug_stats = augment_query_global(
            query_id=qid,
            query_item=query_item,
            existing_candidates=candidates,
            query_emb=q_emb,
            cand_emb=cand_emb,
            dataset=dataset,
            vqa_interface=vqa_interface,
            vqa_train_cache=vqa_train_cache,
            vqa_val_cache=vqa_val_cache,
            train_ques_path=train_ques_path,
            train_ann_path=train_ann_path,
            val_ques_path=val_ques_path,
            val_ann_path=val_ann_path,
            aug_type='all1',
            max_eval_budget=args.max_eval_budget_all1,
            target_positive_ratio=args.target_positive_ratio,
            device=device
        )
        
        if new_candidates:
            rl_data[qid]['pointer_candidates'].extend(new_candidates)
        
        if aug_stats.get("success"):
            stats["after"]["all1_success"] += 1
        stats["after"]["total_eval_count"] += aug_stats.get("eval_count", 0)
    
    # 保存结果
    print("\n保存结果...")
    with open(output_path, 'w') as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)
    
    # 统计最终结果
    final_all0 = final_all1 = final_diverse = 0
    total_cand = pos_cand = 0
    total_queries = 0
    
    for qid, qdata in rl_data.items():
        if qid == '_meta':
            continue
        total_queries += 1
        candidates = qdata.get('pointer_candidates', [])
        scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
        total_cand += len(candidates)
        pos_cand += sum(1 for s in scores if s > 0)
        
        unique_scores = set(scores)
        if len(unique_scores) == 1:
            score = list(unique_scores)[0]
            if abs(score) < 1e-6:
                final_all0 += 1
            elif abs(score - 1.0) < 1e-6:
                final_all1 += 1
            else:
                final_diverse += 1
        else:
            final_diverse += 1
    
    stats["after"]["final_all0"] = final_all0
    stats["after"]["final_all1"] = final_all1
    stats["after"]["final_diverse"] = final_diverse
    stats["after"]["positive_ratio"] = pos_cand / total_cand if total_cand > 0 else 0
    
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("补全完成！")
    print("=" * 60)
    print(f"All-Zero 成功率: {stats['after']['all0_success']}/{len(all0_queries)} ({100*stats['after']['all0_success']/max(1,len(all0_queries)):.1f}%)")
    print(f"All-One 成功率: {stats['after']['all1_success']}/{len(all1_queries)} ({100*stats['after']['all1_success']/max(1,len(all1_queries)):.1f}%)")
    print(f"\n最终数据分布:")
    print(f"  All-Zero: {final_all0} ({100*final_all0/max(1,total_queries):.1f}%)")
    print(f"  All-One: {final_all1} ({100*final_all1/max(1,total_queries):.1f}%)")
    print(f"  Diverse: {final_diverse} ({100*final_diverse/max(1,total_queries):.1f}%)")
    print(f"  正样本比例: {100*stats['after']['positive_ratio']:.1f}%")
    print(f"\n输出文件: {output_path}")


if __name__ == "__main__":
    main()
