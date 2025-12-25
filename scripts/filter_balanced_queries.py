#!/usr/bin/env python3
"""
过滤 RL 数据，只保留同时有正负样本的 query

问题分析：
- 当前数据中有 17% 的 query 全是负样本（136个）
- 有 5.5% 的 query 全是正样本（44个）
- 这些 query 的 advantage 信号很弱，无法有效训练

解决方案：
- 只保留混合 query（同时有正负样本）
- 这样每个 query 都有明确的正负对比信号

使用方法：
    python scripts/filter_balanced_queries.py \
        --input_path results/okvqa/generated_data/rl_data_k64_v3.json \
        --output_path results/okvqa/generated_data/rl_data_k64_v3_balanced.json

作者: Lever-Plus Team
日期: 2025-12-22
"""

import json
import os
import argparse
import numpy as np
from copy import deepcopy
from datetime import datetime


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def analyze_query(qdata):
    """分析单个 query 的正负样本情况"""
    candidates = qdata.get('pointer_candidates', [])
    
    pos_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 1)
    neg_count = len(candidates) - pos_count
    
    return {
        'total': len(candidates),
        'positive': pos_count,
        'negative': neg_count,
        'is_all_positive': pos_count == len(candidates) and len(candidates) > 0,
        'is_all_negative': neg_count == len(candidates) and len(candidates) > 0,
        'is_mixed': pos_count > 0 and neg_count > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="过滤 RL 数据，只保留混合 query")
    parser.add_argument("--input_path", type=str, required=True, help="输入 RL 数据路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出过滤后数据路径")
    parser.add_argument("--min_positive", type=int, default=1, help="最少正样本数（默认 1）")
    parser.add_argument("--min_negative", type=int, default=1, help="最少负样本数（默认 1）")
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.input_path}")
    data = load_json(args.input_path)
    
    queries = {k: v for k, v in data.items() if k != '_meta'}
    
    print(f"\n{'=' * 70}")
    print("过滤前统计")
    print('=' * 70)
    print(f"总 query 数: {len(queries)}")
    
    # 分析每个 query
    all_positive_qids = []
    all_negative_qids = []
    mixed_qids = []
    
    for qid, qdata in queries.items():
        stats = analyze_query(qdata)
        
        if stats['is_all_positive']:
            all_positive_qids.append(qid)
        elif stats['is_all_negative']:
            all_negative_qids.append(qid)
        elif stats['is_mixed']:
            mixed_qids.append(qid)
    
    print(f"\n全正样本 query: {len(all_positive_qids)} ({len(all_positive_qids)/len(queries)*100:.1f}%)")
    print(f"全负样本 query: {len(all_negative_qids)} ({len(all_negative_qids)/len(queries)*100:.1f}%)")
    print(f"混合 query: {len(mixed_qids)} ({len(mixed_qids)/len(queries)*100:.1f}%)")
    
    # 进一步过滤：确保至少有 min_positive 个正样本和 min_negative 个负样本
    filtered_qids = []
    for qid in mixed_qids:
        stats = analyze_query(queries[qid])
        if stats['positive'] >= args.min_positive and stats['negative'] >= args.min_negative:
            filtered_qids.append(qid)
    
    print(f"\n满足条件的 query (正样本>={args.min_positive}, 负样本>={args.min_negative}): {len(filtered_qids)}")
    
    # 构建过滤后的数据
    filtered_data = {}
    
    # 保留 _meta
    if '_meta' in data:
        filtered_data['_meta'] = deepcopy(data['_meta'])
        filtered_data['_meta']['filtered_at'] = datetime.now().isoformat()
        filtered_data['_meta']['filter_params'] = {
            'min_positive': args.min_positive,
            'min_negative': args.min_negative,
        }
        filtered_data['_meta']['original_query_count'] = len(queries)
        filtered_data['_meta']['filtered_query_count'] = len(filtered_qids)
        filtered_data['_meta']['removed_all_positive'] = len(all_positive_qids)
        filtered_data['_meta']['removed_all_negative'] = len(all_negative_qids)
    
    for qid in filtered_qids:
        filtered_data[qid] = queries[qid]
    
    # 统计过滤后的数据
    print(f"\n{'=' * 70}")
    print("过滤后统计")
    print('=' * 70)
    print(f"保留 query 数: {len(filtered_qids)}")
    print(f"移除 query 数: {len(queries) - len(filtered_qids)}")
    
    # 统计正负样本
    total_pos = 0
    total_neg = 0
    pos_ratios = []
    
    for qid in filtered_qids:
        stats = analyze_query(queries[qid])
        total_pos += stats['positive']
        total_neg += stats['negative']
        pos_ratios.append(stats['positive'] / stats['total'])
    
    print(f"\n总正样本数: {total_pos}")
    print(f"总负样本数: {total_neg}")
    print(f"正样本比例: {total_pos / (total_pos + total_neg) * 100:.1f}%")
    print(f"\n每个 query 的正样本比例:")
    print(f"  平均: {np.mean(pos_ratios)*100:.1f}%")
    print(f"  最小: {min(pos_ratios)*100:.1f}%")
    print(f"  最大: {max(pos_ratios)*100:.1f}%")
    
    # 计算预期的 advantage 差异
    print(f"\n{'=' * 70}")
    print("预期 Advantage 分析")
    print('=' * 70)
    
    all_pos_advs = []
    all_neg_advs = []
    
    for qid in filtered_qids:
        candidates = queries[qid].get('pointer_candidates', [])
        
        rewards = []
        corrects = []
        for c in candidates:
            hard = float(c.get('vqa_correct', 0))
            soft = float(c.get('vqa_acc_score', 0))
            # hard_plus_soft_v2: reward = soft + 2*hard
            reward = soft + 2.0 * hard
            rewards.append(reward)
            corrects.append(hard)
        
        rewards = np.array(rewards)
        corrects = np.array(corrects)
        
        # Z-score 归一化
        mean = rewards.mean()
        std = max(rewards.std(), 0.1)
        advantages = (rewards - mean) / std
        advantages = np.clip(advantages, -5, 5)
        
        for adv, correct in zip(advantages, corrects):
            if correct == 1:
                all_pos_advs.append(adv)
            else:
                all_neg_advs.append(adv)
    
    print(f"正样本 advantage 平均: {np.mean(all_pos_advs):.3f}")
    print(f"负样本 advantage 平均: {np.mean(all_neg_advs):.3f}")
    print(f"差值: {np.mean(all_pos_advs) - np.mean(all_neg_advs):.3f}")
    
    # 保存
    print(f"\n保存过滤后数据: {args.output_path}")
    save_json(filtered_data, args.output_path)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
