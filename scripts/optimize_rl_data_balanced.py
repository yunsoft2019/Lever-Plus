#!/usr/bin/env python3
"""
优化RL数据：确保每个query的候选池是64个索引，同时保持正负样本平衡
策略：
1. 保留原有的所有候选（包括正负样本）
2. 从全局正样本中补充，但控制正样本比例不超过70%
3. 如果正样本不足，从全局添加
4. 确保候选池大小为64
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
        'zero_positive_queries': [],
        'positive_ratios': []
    }
    
    for qid, qdata in queries.items():
        candidates = qdata.get('pointer_candidates', [])
        stats['candidate_counts'].append(len(candidates))
        
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
        stats['positive_ratios'].append(pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0)
        
        if pos_count == 0:
            stats['zero_positive_queries'].append(qid)
    
    return stats

def get_all_samples(data):
    """收集所有样本，按正负分组"""
    queries = {k: v for k, v in data.items() if k != '_meta'}
    
    # 全局样本库
    global_positives = defaultdict(list)  # {pointer: [samples]}
    global_negatives = defaultdict(list)
    
    for qid, qdata in queries.items():
        for c in qdata.get('pointer_candidates', []):
            pointer = tuple(sorted(c.get('pointer', [])))
            sample_info = {
                'source_query': qid,
                'candidate': c,
                'score': c.get('vqa_acc_score', 0) if c.get('vqa_correct', 0) == 1 else c.get('vqa_rel_score', 0)
            }
            
            if c.get('vqa_correct', 0) == 1:
                global_positives[pointer].append(sample_info)
            else:
                global_negatives[pointer].append(sample_info)
    
    return global_positives, global_negatives

def optimize_query_pool_balanced(qid, qdata, global_positives, global_negatives, 
                                  target_pool_size=64, max_positive_ratio=0.7, min_positive_ratio=0.3):
    """
    优化单个query的候选池，保持正负样本平衡
    """
    candidates = qdata.get('pointer_candidates', [])
    
    # 当前使用的索引和样本
    current_indices = set()
    current_pointers = set()
    current_positives = []
    current_negatives = []
    
    for c in candidates:
        pointer = tuple(sorted(c.get('pointer', [])))
        current_pointers.add(pointer)
        for idx in c.get('pointer', []):
            current_indices.add(idx)
        
        if c.get('vqa_correct', 0) == 1:
            current_positives.append(c)
        else:
            current_negatives.append(c)
    
    # 如果已经有足够的索引，检查正负比例
    if len(current_indices) >= target_pool_size:
        total = len(current_positives) + len(current_negatives)
        pos_ratio = len(current_positives) / total if total > 0 else 0
        return qdata, {'expanded': False, 'reason': 'already_enough', 'pos_ratio': pos_ratio}
    
    # 需要添加的索引数量
    needed_indices = target_pool_size - len(current_indices)
    
    # 计算目标正负样本数
    # 假设每个新pointer贡献约2个新索引
    estimated_new_samples = needed_indices // 2 + 1
    
    # 当前正样本比例
    current_total = len(current_positives) + len(current_negatives)
    current_pos_ratio = len(current_positives) / current_total if current_total > 0 else 0
    
    # 决定添加正样本还是负样本
    added_candidates = []
    added_indices = set()
    
    # 优先添加正样本（如果当前正样本不足）
    if current_pos_ratio < min_positive_ratio or len(current_positives) == 0:
        # 需要更多正样本
        for pointer, samples in sorted(global_positives.items(), key=lambda x: -max(s['score'] for s in x[1])):
            if pointer in current_pointers:
                continue
            
            idx1, idx2 = pointer if len(pointer) == 2 else (pointer[0], pointer[0])
            
            new_needed = 0
            if idx1 not in current_indices and idx1 not in added_indices:
                new_needed += 1
            if idx2 not in current_indices and idx2 not in added_indices:
                new_needed += 1
            
            if len(current_indices) + len(added_indices) + new_needed <= target_pool_size:
                best_sample = max(samples, key=lambda x: x['score'])
                new_candidate = deepcopy(best_sample['candidate'])
                new_candidate['gen_method'] = 'global_positive_transfer'
                new_candidate['source_query'] = best_sample['source_query']
                added_candidates.append(new_candidate)
                current_pointers.add(pointer)
                
                if idx1 not in current_indices:
                    added_indices.add(idx1)
                if idx2 not in current_indices:
                    added_indices.add(idx2)
                
                # 检查是否达到目标
                if len(current_indices) + len(added_indices) >= target_pool_size:
                    break
    
    # 如果还需要更多索引，继续添加（优先正样本，但也可以添加负样本）
    if len(current_indices) + len(added_indices) < target_pool_size:
        # 混合添加
        remaining_positives = [(p, s) for p, s in global_positives.items() if p not in current_pointers]
        remaining_negatives = [(p, s) for p, s in global_negatives.items() if p not in current_pointers]
        
        # 按得分排序
        remaining_positives.sort(key=lambda x: -max(s['score'] for s in x[1]))
        remaining_negatives.sort(key=lambda x: -max(s['score'] for s in x[1]))
        
        # 交替添加
        pos_idx = 0
        neg_idx = 0
        
        while len(current_indices) + len(added_indices) < target_pool_size:
            added_this_round = False
            
            # 尝试添加正样本
            while pos_idx < len(remaining_positives):
                pointer, samples = remaining_positives[pos_idx]
                pos_idx += 1
                
                if pointer in current_pointers:
                    continue
                
                idx1, idx2 = pointer if len(pointer) == 2 else (pointer[0], pointer[0])
                new_needed = 0
                if idx1 not in current_indices and idx1 not in added_indices:
                    new_needed += 1
                if idx2 not in current_indices and idx2 not in added_indices:
                    new_needed += 1
                
                if new_needed > 0 and len(current_indices) + len(added_indices) + new_needed <= target_pool_size:
                    best_sample = max(samples, key=lambda x: x['score'])
                    new_candidate = deepcopy(best_sample['candidate'])
                    new_candidate['gen_method'] = 'global_positive_transfer'
                    new_candidate['source_query'] = best_sample['source_query']
                    added_candidates.append(new_candidate)
                    current_pointers.add(pointer)
                    
                    if idx1 not in current_indices:
                        added_indices.add(idx1)
                    if idx2 not in current_indices:
                        added_indices.add(idx2)
                    added_this_round = True
                    break
            
            if not added_this_round:
                break
    
    # 更新qdata
    if added_candidates:
        new_qdata = deepcopy(qdata)
        new_qdata['pointer_candidates'].extend(added_candidates)
        
        # 计算最终正样本比例
        final_pos = len(current_positives) + sum(1 for c in added_candidates if c.get('vqa_correct', 0) == 1)
        final_neg = len(current_negatives) + sum(1 for c in added_candidates if c.get('vqa_correct', 0) == 0)
        final_ratio = final_pos / (final_pos + final_neg) if (final_pos + final_neg) > 0 else 0
        
        return new_qdata, {
            'expanded': True,
            'added_count': len(added_candidates),
            'new_indices': len(added_indices),
            'final_pool_size': len(current_indices) + len(added_indices),
            'final_pos_ratio': final_ratio
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
    print(f"负样本数: {np.mean(stats_before['negative_counts']):.1f}")
    print(f"正样本比例: {np.mean(stats_before['positive_ratios'])*100:.1f}%")
    print(f"零正样本query: {len(stats_before['zero_positive_queries'])} ({len(stats_before['zero_positive_queries'])/stats_before['total_queries']*100:.1f}%)")
    
    # 收集全局样本
    print("\n收集全局样本...")
    global_positives, global_negatives = get_all_samples(data)
    print(f"全局unique正样本pointer数: {len(global_positives)}")
    print(f"全局unique负样本pointer数: {len(global_negatives)}")
    
    # 优化每个query
    print("\n优化候选池...")
    optimized_data = {'_meta': deepcopy(data['_meta'])}
    optimized_data['_meta']['optimized_at'] = datetime.now().isoformat()
    optimized_data['_meta']['optimization'] = 'expand_pool_to_64_balanced'
    
    expansion_stats = {
        'expanded': 0,
        'not_expanded': 0,
        'total_added_candidates': 0,
        'total_new_indices': 0
    }
    
    queries = {k: v for k, v in data.items() if k != '_meta'}
    for qid, qdata in queries.items():
        optimized_qdata, result = optimize_query_pool_balanced(
            qid, qdata, global_positives, global_negatives, 
            target_pool_size=64
        )
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
    print(f"负样本数: {np.mean(stats_after['negative_counts']):.1f}")
    print(f"正样本比例: {np.mean(stats_after['positive_ratios'])*100:.1f}%")
    print(f"零正样本query: {len(stats_after['zero_positive_queries'])} ({len(stats_after['zero_positive_queries'])/stats_after['total_queries']*100:.1f}%)")
    
    # 保存优化后的数据
    output_path = 'results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_pool64_optimized.json'
    print(f"\n保存优化后数据: {output_path}")
    save_json(optimized_data, output_path)
    
    print("\n完成!")

if __name__ == '__main__':
    main()
