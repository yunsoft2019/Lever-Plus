#!/usr/bin/env python3
"""
优化RL数据：确保每个query的候选池是64个索引
策略：
1. 从全局找到更好的正样本，放入候选池
2. 保持候选池大小为64
3. 移除低质量的负样本
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime
import numpy as np
from copy import deepcopy

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def analyze_current_data(data):
    """分析当前数据的统计信息"""
    queries = {k: v for k, v in data.items() if k != '_meta'}
    
    stats = {
        'total_queries': len(queries),
        'candidate_counts': [],
        'positive_counts': [],
        'negative_counts': [],
        'pool_sizes': [],
        'zero_positive_queries': []
    }
    
    for qid, qdata in queries.items():
        candidates = qdata.get('pointer_candidates', [])
        stats['candidate_counts'].append(len(candidates))
        
        # 收集索引
        indices = set()
        pos_count = 0
        neg_count = 0
        
        for c in candidates:
            for idx in c.get('pointer', []):
                indices.add(idx)
            if c.get('vqa_correct', 0) == 1:
                pos_count += 1
            else:
                neg_count += 1
        
        stats['positive_counts'].append(pos_count)
        stats['negative_counts'].append(neg_count)
        stats['pool_sizes'].append(len(indices))
        
        if pos_count == 0:
            stats['zero_positive_queries'].append(qid)
    
    return stats

def get_all_positive_samples(data):
    """收集所有正样本，按query分组"""
    queries = {k: v for k, v in data.items() if k != '_meta'}
    
    # 全局正样本库：{(idx1, idx2): [candidate_info, ...]}
    global_positives = defaultdict(list)
    
    for qid, qdata in queries.items():
        for c in qdata.get('pointer_candidates', []):
            if c.get('vqa_correct', 0) == 1:
                pointer = tuple(sorted(c.get('pointer', [])))
                global_positives[pointer].append({
                    'source_query': qid,
                    'candidate': c,
                    'score': c.get('vqa_acc_score', 0)
                })
    
    return global_positives

def optimize_query_pool(qid, qdata, global_positives, target_pool_size=64):
    """
    优化单个query的候选池
    1. 保留所有现有候选
    2. 从全局正样本中找到可以添加的
    3. 确保候选池大小为64
    """
    candidates = qdata.get('pointer_candidates', [])
    
    # 当前使用的索引
    current_indices = set()
    for c in candidates:
        for idx in c.get('pointer', []):
            current_indices.add(idx)
    
    # 当前正样本的pointer
    current_positive_pointers = set()
    for c in candidates:
        if c.get('vqa_correct', 0) == 1:
            current_positive_pointers.add(tuple(sorted(c.get('pointer', []))))
    
    # 如果已经有足够的索引，不需要扩展
    if len(current_indices) >= target_pool_size:
        return qdata, {'expanded': False, 'reason': 'already_enough'}
    
    # 需要添加的索引数量
    needed_indices = target_pool_size - len(current_indices)
    
    # 从全局正样本中找可以添加的
    # 优先找那些pointer中有一个索引已经在当前池中的
    potential_additions = []
    
    for pointer, samples in global_positives.items():
        if pointer in current_positive_pointers:
            continue  # 已经有了
        
        # 检查这个pointer是否可以添加
        idx1, idx2 = pointer if len(pointer) == 2 else (pointer[0], pointer[0])
        
        # 计算需要添加多少新索引
        new_indices_needed = 0
        if idx1 not in current_indices:
            new_indices_needed += 1
        if idx2 not in current_indices:
            new_indices_needed += 1
        
        if new_indices_needed <= needed_indices:
            # 选择得分最高的样本
            best_sample = max(samples, key=lambda x: x['score'])
            potential_additions.append({
                'pointer': pointer,
                'new_indices_needed': new_indices_needed,
                'sample': best_sample,
                'score': best_sample['score']
            })
    
    # 按得分排序，优先添加高分样本
    potential_additions.sort(key=lambda x: (-x['score'], x['new_indices_needed']))
    
    # 添加样本
    added_candidates = []
    added_indices = set()
    
    for addition in potential_additions:
        pointer = addition['pointer']
        idx1, idx2 = pointer if len(pointer) == 2 else (pointer[0], pointer[0])
        
        # 检查是否还能添加
        new_needed = 0
        if idx1 not in current_indices and idx1 not in added_indices:
            new_needed += 1
        if idx2 not in current_indices and idx2 not in added_indices:
            new_needed += 1
        
        if len(current_indices) + len(added_indices) + new_needed <= target_pool_size:
            # 可以添加
            new_candidate = deepcopy(addition['sample']['candidate'])
            new_candidate['gen_method'] = 'global_positive_transfer'
            new_candidate['source_query'] = addition['sample']['source_query']
            added_candidates.append(new_candidate)
            
            if idx1 not in current_indices:
                added_indices.add(idx1)
            if idx2 not in current_indices:
                added_indices.add(idx2)
    
    # 更新qdata
    if added_candidates:
        new_qdata = deepcopy(qdata)
        new_qdata['pointer_candidates'].extend(added_candidates)
        return new_qdata, {
            'expanded': True,
            'added_count': len(added_candidates),
            'new_indices': len(added_indices),
            'final_pool_size': len(current_indices) + len(added_indices)
        }
    
    return qdata, {'expanded': False, 'reason': 'no_suitable_additions'}

def main():
    # 加载数据
    input_path = 'results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_final_merged.json'
    print(f"加载数据: {input_path}")
    data = load_json(input_path)
    
    # 分析当前数据
    print("\n" + "=" * 70)
    print("优化前统计")
    print("=" * 70)
    stats_before = analyze_current_data(data)
    print(f"总query数: {stats_before['total_queries']}")
    print(f"候选池大小: {np.mean(stats_before['pool_sizes']):.1f} (min={min(stats_before['pool_sizes'])}, max={max(stats_before['pool_sizes'])})")
    print(f"正样本数: {np.mean(stats_before['positive_counts']):.1f}")
    print(f"零正样本query: {len(stats_before['zero_positive_queries'])} ({len(stats_before['zero_positive_queries'])/stats_before['total_queries']*100:.1f}%)")
    
    # 收集全局正样本
    print("\n收集全局正样本...")
    global_positives = get_all_positive_samples(data)
    print(f"全局unique正样本pointer数: {len(global_positives)}")
    
    # 优化每个query
    print("\n优化候选池...")
    optimized_data = {'_meta': deepcopy(data['_meta'])}
    optimized_data['_meta']['optimized_at'] = datetime.now().isoformat()
    optimized_data['_meta']['optimization'] = 'expand_pool_to_64_with_global_positives'
    
    expansion_stats = {
        'expanded': 0,
        'not_expanded': 0,
        'total_added_candidates': 0,
        'total_new_indices': 0
    }
    
    queries = {k: v for k, v in data.items() if k != '_meta'}
    for qid, qdata in queries.items():
        optimized_qdata, result = optimize_query_pool(qid, qdata, global_positives, target_pool_size=64)
        optimized_data[qid] = optimized_qdata
        
        if result['expanded']:
            expansion_stats['expanded'] += 1
            expansion_stats['total_added_candidates'] += result['added_count']
            expansion_stats['total_new_indices'] += result['new_indices']
        else:
            expansion_stats['not_expanded'] += 1
    
    print(f"\n扩展统计:")
    print(f"  扩展的query: {expansion_stats['expanded']}")
    print(f"  未扩展的query: {expansion_stats['not_expanded']}")
    print(f"  添加的候选总数: {expansion_stats['total_added_candidates']}")
    print(f"  添加的新索引总数: {expansion_stats['total_new_indices']}")
    
    # 分析优化后数据
    print("\n" + "=" * 70)
    print("优化后统计")
    print("=" * 70)
    stats_after = analyze_current_data(optimized_data)
    print(f"总query数: {stats_after['total_queries']}")
    print(f"候选池大小: {np.mean(stats_after['pool_sizes']):.1f} (min={min(stats_after['pool_sizes'])}, max={max(stats_after['pool_sizes'])})")
    print(f"正样本数: {np.mean(stats_after['positive_counts']):.1f}")
    print(f"零正样本query: {len(stats_after['zero_positive_queries'])} ({len(stats_after['zero_positive_queries'])/stats_after['total_queries']*100:.1f}%)")
    
    # 保存优化后的数据
    output_path = 'results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_pool64_optimized.json'
    print(f"\n保存优化后数据: {output_path}")
    save_json(optimized_data, output_path)
    
    print("\n完成!")

if __name__ == '__main__':
    main()
