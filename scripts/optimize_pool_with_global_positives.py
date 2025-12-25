#!/usr/bin/env python3
"""
优化候选池：从全局找正样本ICD，替换进64个候选池

核心思路：
1. 每个query有64个候选ICD
2. 从64个候选中组合长度为2的序列作为RL样本
3. 问题：从64个候选池中组合不出正样本
4. 解决：从全局（8000+条）中找到能构成正样本的ICD，替换进候选池

策略：
1. 从全局正样本中找到构成正样本的ICD对
2. 把这些ICD替换进候选池中没用到的位置
3. 保持候选池大小为64
4. 最终只需要约20个序列样本，其中有几条正样本

使用方法：
    python scripts/optimize_pool_with_global_positives.py \
        --input_path results/okvqa/generated_data/rl_data.json \
        --output_path results/okvqa/generated_data/rl_data_optimized.json \
        --target_pool_size 64 \
        --target_sequence_num 20 \
        --min_positive_sequences 3
"""

import json
import os
import argparse
import random
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
import numpy as np
from tqdm import tqdm


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def analyze_data(data):
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
        
        total = pos_count + neg_count
        if total > 0:
            stats['positive_ratios'].append(pos_count / total)
        else:
            stats['positive_ratios'].append(0)
    
    return stats


def collect_global_positive_icds(data):
    """
    收集全局所有正样本中使用的ICD
    
    返回：
    - global_positive_pointers: {(idx1, idx2): [candidate_info, ...]}
        每个正样本pointer对应的候选信息列表
    - icd_to_positive_pointers: {icd_idx: [(pointer, score), ...]}
        每个ICD参与的正样本pointer列表
    """
    queries = {k: v for k, v in data.items() if k != '_meta'}
    
    # 全局正样本库：{(idx1, idx2): [candidate_info, ...]}
    global_positive_pointers = defaultdict(list)
    
    # ICD到正样本pointer的映射：{icd_idx: [(pointer, score), ...]}
    icd_to_positive_pointers = defaultdict(list)
    
    for qid, qdata in queries.items():
        for c in qdata.get('pointer_candidates', []):
            if c.get('vqa_correct', 0) == 1:
                pointer = c.get('pointer', [])
                if len(pointer) >= 2:
                    pointer_key = tuple(sorted(pointer[:2]))
                    score = c.get('vqa_acc_score', 0)
                    
                    global_positive_pointers[pointer_key].append({
                        'source_query': qid,
                        'candidate': c,
                        'score': score
                    })
                    
                    # 记录每个ICD参与的正样本
                    for idx in pointer[:2]:
                        icd_to_positive_pointers[idx].append((pointer_key, score))
    
    return global_positive_pointers, icd_to_positive_pointers


def get_current_pool_indices(qdata):
    """获取当前候选池中的所有ICD索引"""
    indices = set()
    for c in qdata.get('pointer_candidates', []):
        for idx in c.get('pointer', []):
            indices.add(idx)
    return indices


def get_used_indices_in_sequences(qdata):
    """获取当前序列中实际使用的ICD索引"""
    used = set()
    for c in qdata.get('pointer_candidates', []):
        pointer = c.get('pointer', [])
        for idx in pointer:
            used.add(idx)
    return used


def find_best_global_icds_for_query(
    qid,
    qdata,
    global_positive_pointers,
    icd_to_positive_pointers,
    current_pool_indices,
    target_pool_size=64,
    max_new_icds=10
):
    """
    为单个query找到最佳的全局ICD来替换
    
    策略：
    1. 优先找全局正样本中，有一个ICD已经在当前池中的pointer（只需添加1个新ICD）
    2. 其次找全局正样本中，两个ICD都不在当前池中的pointer（需要添加2个新ICD）
    3. 把新ICD替换进候选池中没用到的位置
    
    返回：
    - new_icds: 需要添加的新ICD索引列表
    - new_positive_pointers: 可以构成的新正样本pointer列表
    """
    # 当前池中的ICD
    pool_set = set(current_pool_indices)
    
    # 分类：需要添加0/1/2个新ICD的pointer
    need_0_icd = []  # 两个ICD都在池中
    need_1_icd = []  # 只需要添加1个ICD
    need_2_icd = []  # 需要添加2个ICD（全局找到的）
    
    for pointer_key, samples in global_positive_pointers.items():
        idx1, idx2 = pointer_key
        
        # 检查是否有ICD已经在池中
        in_pool_count = (idx1 in pool_set) + (idx2 in pool_set)
        
        best_sample = max(samples, key=lambda x: x['score'])
        
        if in_pool_count == 2:
            # 两个ICD都在池中，检查是否已经有这个组合
            has_this_positive = False
            for c in qdata.get('pointer_candidates', []):
                if c.get('vqa_correct', 0) == 1:
                    existing_pointer = tuple(sorted(c.get('pointer', [])[:2]))
                    if existing_pointer == pointer_key:
                        has_this_positive = True
                        break
            
            if not has_this_positive:
                need_0_icd.append({
                    'new_icds': [],
                    'pointer': pointer_key,
                    'score': best_sample['score'],
                    'sample': best_sample
                })
        elif in_pool_count == 1:
            # 只需要添加1个ICD
            new_idx = idx2 if idx1 in pool_set else idx1
            need_1_icd.append({
                'new_icds': [new_idx],
                'pointer': pointer_key,
                'score': best_sample['score'],
                'sample': best_sample
            })
        else:
            # 需要添加2个ICD（全局找到的，不在当前池中）
            need_2_icd.append({
                'new_icds': [idx1, idx2],
                'pointer': pointer_key,
                'score': best_sample['score'],
                'sample': best_sample
            })
    
    # 按分数排序
    need_0_icd.sort(key=lambda x: -x['score'])
    need_1_icd.sort(key=lambda x: -x['score'])
    need_2_icd.sort(key=lambda x: -x['score'])
    
    # 选择最佳的additions
    new_icds = set()
    new_positive_pointers = []
    
    # 1. 首先添加不需要新ICD的（两个ICD都在池中）
    for addition in need_0_icd:
        new_positive_pointers.append(addition)
    
    # 2. 然后添加只需要1个新ICD的
    for addition in need_1_icd:
        if len(new_icds) < max_new_icds:
            new_icds.update(addition['new_icds'])
            new_positive_pointers.append(addition)
    
    # 3. 最后添加需要2个新ICD的（全局找到的）
    for addition in need_2_icd:
        if len(new_icds) + 2 <= max_new_icds:
            new_icds.update(addition['new_icds'])
            new_positive_pointers.append(addition)
    
    return list(new_icds), new_positive_pointers


def optimize_query_pool(
    qid,
    qdata,
    global_positive_pointers,
    icd_to_positive_pointers,
    target_pool_size=64,
    target_sequence_num=20,
    min_positive_sequences=3
):
    """
    优化单个query的候选池
    
    核心策略：
    1. 从64个候选池中组合序列，如果组合不出正样本
    2. 从全局找到能构成正样本的ICD对
    3. 把全局找到的ICD替换进候选池中没用到的位置
    4. 保持候选池大小为64
    5. 最终只保留约20个序列，其中有几条正样本
    
    关键点：
    - 正样本是由两个ICD组成的序列
    - 全局找到的ICD可能不在当前64个候选池中
    - 需要把新ICD替换进候选池，替换掉那些没用到的ICD
    """
    candidates = qdata.get('pointer_candidates', [])
    
    # 当前池中的ICD索引
    current_pool_indices = get_current_pool_indices(qdata)
    
    # 当前的正样本和负样本
    current_positives = [c for c in candidates if c.get('vqa_correct', 0) == 1]
    current_negatives = [c for c in candidates if c.get('vqa_correct', 0) == 0]
    
    # 如果已经有足够的正样本，只需要筛选序列数量
    if len(current_positives) >= min_positive_sequences:
        # 保留所有正样本 + 部分负样本
        final_candidates = current_positives.copy()
        remaining_slots = target_sequence_num - len(final_candidates)
        if remaining_slots > 0 and current_negatives:
            # 随机选择负样本
            selected_negatives = random.sample(
                current_negatives, 
                min(remaining_slots, len(current_negatives))
            )
            final_candidates.extend(selected_negatives)
        
        new_qdata = deepcopy(qdata)
        new_qdata['pointer_candidates'] = final_candidates
        return new_qdata, {
            'optimized': True,
            'reason': 'already_has_positives',
            'positive_count': len(current_positives),
            'final_sequence_count': len(final_candidates),
            'new_icds_added': 0
        }
    
    # 需要从全局找正样本
    # 计算可以添加多少新ICD（保持候选池大小为64）
    max_new_icds = max(0, target_pool_size - len(current_pool_indices))
    
    new_icds, new_positive_pointers = find_best_global_icds_for_query(
        qid=qid,
        qdata=qdata,
        global_positive_pointers=global_positive_pointers,
        icd_to_positive_pointers=icd_to_positive_pointers,
        current_pool_indices=current_pool_indices,
        target_pool_size=target_pool_size,
        max_new_icds=max_new_icds
    )
    
    # 如果候选池已满但还需要添加新ICD，需要替换掉一些没用到的ICD
    if len(new_icds) > max_new_icds and max_new_icds == 0:
        # 找出当前候选池中没有被任何序列使用的ICD
        used_in_sequences = get_used_indices_in_sequences(qdata)
        unused_icds = current_pool_indices - used_in_sequences
        
        # 可以替换掉的ICD数量
        replaceable_count = len(unused_icds)
        
        # 重新计算可以添加的新ICD数量
        max_new_icds = replaceable_count
        
        # 重新找全局正样本
        new_icds, new_positive_pointers = find_best_global_icds_for_query(
            qid=qid,
            qdata=qdata,
            global_positive_pointers=global_positive_pointers,
            icd_to_positive_pointers=icd_to_positive_pointers,
            current_pool_indices=current_pool_indices,
            target_pool_size=target_pool_size,
            max_new_icds=max_new_icds
        )
    
    # 构建新的候选序列
    final_candidates = current_positives.copy()
    
    # 添加从全局找到的正样本序列
    added_positive_count = 0
    for addition in new_positive_pointers:
        if len(final_candidates) >= target_sequence_num:
            break
        
        # 创建新的候选
        new_candidate = deepcopy(addition['sample']['candidate'])
        new_candidate['gen_method'] = 'global_positive_transfer'
        new_candidate['source_query'] = addition['sample']['source_query']
        new_candidate['pointer'] = list(addition['pointer'])
        final_candidates.append(new_candidate)
        added_positive_count += 1
    
    # 如果正样本还不够，添加一些负样本
    remaining_slots = target_sequence_num - len(final_candidates)
    if remaining_slots > 0 and current_negatives:
        selected_negatives = random.sample(
            current_negatives,
            min(remaining_slots, len(current_negatives))
        )
        final_candidates.extend(selected_negatives)
    
    # 更新候选池索引（添加新ICD）
    new_pool_indices = current_pool_indices.union(set(new_icds))
    
    new_qdata = deepcopy(qdata)
    new_qdata['pointer_candidates'] = final_candidates
    
    # 记录候选池变化
    new_qdata['_pool_optimization'] = {
        'original_pool_size': len(current_pool_indices),
        'new_pool_size': len(new_pool_indices),
        'new_icds_added': list(new_icds),
        'global_positives_added': added_positive_count
    }
    
    # 统计最终的正样本数
    final_positive_count = sum(1 for c in final_candidates if c.get('vqa_correct', 0) == 1)
    
    return new_qdata, {
        'optimized': True,
        'reason': 'added_global_positives',
        'positive_count': final_positive_count,
        'final_sequence_count': len(final_candidates),
        'new_icds_added': len(new_icds),
        'new_positive_pointers_found': len(new_positive_pointers),
        'global_positives_added': added_positive_count
    }


def main():
    parser = argparse.ArgumentParser(description="优化候选池：从全局找正样本ICD")
    parser.add_argument("--input_path", type=str, required=True, help="输入RL数据路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出优化后数据路径")
    parser.add_argument("--target_pool_size", type=int, default=64, help="目标候选池大小")
    parser.add_argument("--target_sequence_num", type=int, default=20, help="目标序列数量")
    parser.add_argument("--min_positive_sequences", type=int, default=3, help="最少正样本序列数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"加载数据: {args.input_path}")
    data = load_json(args.input_path)
    
    # 分析优化前的数据
    print("\n" + "=" * 70)
    print("优化前统计")
    print("=" * 70)
    stats_before = analyze_data(data)
    print(f"总query数: {stats_before['total_queries']}")
    print(f"候选池大小: {np.mean(stats_before['pool_sizes']):.1f} (min={min(stats_before['pool_sizes'])}, max={max(stats_before['pool_sizes'])})")
    print(f"正样本数: {np.mean(stats_before['positive_counts']):.1f}")
    print(f"负样本数: {np.mean(stats_before['negative_counts']):.1f}")
    print(f"正样本比例: {np.mean(stats_before['positive_ratios'])*100:.1f}%")
    print(f"零正样本query: {len(stats_before['zero_positive_queries'])} ({len(stats_before['zero_positive_queries'])/stats_before['total_queries']*100:.1f}%)")
    
    # 收集全局正样本
    print("\n收集全局正样本...")
    global_positive_pointers, icd_to_positive_pointers = collect_global_positive_icds(data)
    print(f"全局unique正样本pointer数: {len(global_positive_pointers)}")
    print(f"参与正样本的unique ICD数: {len(icd_to_positive_pointers)}")
    
    # 优化每个query
    print("\n优化候选池...")
    optimized_data = {}
    if '_meta' in data:
        optimized_data['_meta'] = deepcopy(data['_meta'])
        optimized_data['_meta']['optimized_at'] = datetime.now().isoformat()
        optimized_data['_meta']['optimization'] = 'global_positive_transfer'
        optimized_data['_meta']['optimization_params'] = {
            'target_pool_size': args.target_pool_size,
            'target_sequence_num': args.target_sequence_num,
            'min_positive_sequences': args.min_positive_sequences
        }
    
    optimization_stats = {
        'already_has_positives': 0,
        'added_global_positives': 0,
        'no_change': 0,
        'total_new_icds': 0,
        'total_new_positive_pointers': 0
    }
    
    queries = {k: v for k, v in data.items() if k != '_meta'}
    for qid in tqdm(queries.keys(), desc="优化query"):
        qdata = queries[qid]
        optimized_qdata, result = optimize_query_pool(
            qid=qid,
            qdata=qdata,
            global_positive_pointers=global_positive_pointers,
            icd_to_positive_pointers=icd_to_positive_pointers,
            target_pool_size=args.target_pool_size,
            target_sequence_num=args.target_sequence_num,
            min_positive_sequences=args.min_positive_sequences
        )
        optimized_data[qid] = optimized_qdata
        
        # 统计
        if result['reason'] == 'already_has_positives':
            optimization_stats['already_has_positives'] += 1
        elif result['reason'] == 'added_global_positives':
            optimization_stats['added_global_positives'] += 1
            optimization_stats['total_new_icds'] += result.get('new_icds_added', 0)
            optimization_stats['total_new_positive_pointers'] += result.get('new_positive_pointers_found', 0)
        else:
            optimization_stats['no_change'] += 1
    
    print(f"\n优化统计:")
    print(f"  已有正样本的query: {optimization_stats['already_has_positives']}")
    print(f"  添加全局正样本的query: {optimization_stats['added_global_positives']}")
    print(f"  无变化的query: {optimization_stats['no_change']}")
    print(f"  添加的新ICD总数: {optimization_stats['total_new_icds']}")
    print(f"  找到的新正样本pointer总数: {optimization_stats['total_new_positive_pointers']}")
    
    # 分析优化后的数据
    print("\n" + "=" * 70)
    print("优化后统计")
    print("=" * 70)
    stats_after = analyze_data(optimized_data)
    print(f"总query数: {stats_after['total_queries']}")
    print(f"候选池大小: {np.mean(stats_after['pool_sizes']):.1f} (min={min(stats_after['pool_sizes'])}, max={max(stats_after['pool_sizes'])})")
    print(f"正样本数: {np.mean(stats_after['positive_counts']):.1f}")
    print(f"负样本数: {np.mean(stats_after['negative_counts']):.1f}")
    print(f"正样本比例: {np.mean(stats_after['positive_ratios'])*100:.1f}%")
    print(f"零正样本query: {len(stats_after['zero_positive_queries'])} ({len(stats_after['zero_positive_queries'])/stats_after['total_queries']*100:.1f}%)")
    
    # 对比
    print("\n" + "=" * 70)
    print("优化前后对比")
    print("=" * 70)
    print(f"                    优化前      优化后      变化")
    print(f"候选序列数(平均):   {np.mean(stats_before['candidate_counts']):.1f}        {np.mean(stats_after['candidate_counts']):.1f}       {np.mean(stats_after['candidate_counts'])-np.mean(stats_before['candidate_counts']):+.1f}")
    print(f"正样本数(平均):     {np.mean(stats_before['positive_counts']):.1f}         {np.mean(stats_after['positive_counts']):.1f}        {np.mean(stats_after['positive_counts'])-np.mean(stats_before['positive_counts']):+.1f}")
    print(f"正样本比例:         {np.mean(stats_before['positive_ratios'])*100:.1f}%       {np.mean(stats_after['positive_ratios'])*100:.1f}%      {(np.mean(stats_after['positive_ratios'])-np.mean(stats_before['positive_ratios']))*100:+.1f}%")
    print(f"零正样本query:      {len(stats_before['zero_positive_queries'])}          {len(stats_after['zero_positive_queries'])}          {len(stats_after['zero_positive_queries'])-len(stats_before['zero_positive_queries']):+d}")
    
    # 保存
    print(f"\n保存优化后数据: {args.output_path}")
    save_json(optimized_data, args.output_path)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
