#!/usr/bin/env python3
"""
筛选 VQAv2 RL 数据：控制正样本比例在 45% 以下

筛选条件：
1. 必须有正样本和负样本（混合样本）
2. 正样本比例 <= 45%（负样本比例 >= 55%）
3. 负样本数量 >= 3
4. 正样本数量 >= 1

使用方法：
    python scripts/filter_vqav2_low_pos_ratio.py
"""

import json
import os
import numpy as np
from copy import deepcopy
from datetime import datetime


PROJECT_ROOT = "/mnt/share/yiyun/Projects/Lever-Plus"

# 输入：VQAv2 Balanced V2 数据
INPUT_PATH = f"{PROJECT_ROOT}/results/vqav2/generated_data/rl_data_balanced_merged.json"

# 输出：筛选后的数据
OUTPUT_PATH = f"{PROJECT_ROOT}/results/vqav2/generated_data/rl_data_balanced_v3_low_pos.json"

# 筛选参数
MAX_POS_RATIO = 0.45  # 正样本比例 <= 45%
MIN_NEG_COUNT = 3     # 负样本数量 >= 3
MIN_POS_COUNT = 1     # 正样本数量 >= 1


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print(f"加载数据: {INPUT_PATH}")
    data = load_json(INPUT_PATH)
    
    queries = {k: v for k, v in data.items() if k != '_meta'}
    
    print(f"\n{'='*70}")
    print("筛选条件")
    print('='*70)
    print(f"最大正样本比例: {MAX_POS_RATIO*100:.0f}%")
    print(f"最小负样本比例: {(1-MAX_POS_RATIO)*100:.0f}%")
    print(f"最小负样本数量: {MIN_NEG_COUNT}")
    print(f"最小正样本数量: {MIN_POS_COUNT}")
    
    # 筛选
    filtered_data = {}
    if '_meta' in data:
        filtered_data['_meta'] = deepcopy(data['_meta'])
        filtered_data['_meta']['filtered_at'] = datetime.now().isoformat()
        filtered_data['_meta']['filter_params'] = {
            'max_pos_ratio': MAX_POS_RATIO,
            'min_neg_count': MIN_NEG_COUNT,
            'min_pos_count': MIN_POS_COUNT,
        }
    
    stats = {
        'total': len(queries),
        'kept': 0,
        'removed_all_pos': 0,
        'removed_all_neg': 0,
        'removed_high_pos_ratio': 0,
        'removed_low_neg_count': 0,
        'removed_low_pos_count': 0,
    }
    
    for qid, qdata in queries.items():
        candidates = qdata.get('pointer_candidates', [])
        pos_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 1)
        neg_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 0)
        total = len(candidates)
        
        if total == 0:
            continue
        
        pos_ratio = pos_count / total
        
        # 检查条件
        if neg_count == 0:
            stats['removed_all_pos'] += 1
            continue
        
        if pos_count == 0:
            stats['removed_all_neg'] += 1
            continue
        
        if pos_ratio > MAX_POS_RATIO:
            stats['removed_high_pos_ratio'] += 1
            continue
        
        if neg_count < MIN_NEG_COUNT:
            stats['removed_low_neg_count'] += 1
            continue
        
        if pos_count < MIN_POS_COUNT:
            stats['removed_low_pos_count'] += 1
            continue
        
        # 通过筛选
        filtered_data[qid] = qdata
        stats['kept'] += 1
    
    # 统计筛选后的数据
    print(f"\n{'='*70}")
    print("筛选结果")
    print('='*70)
    print(f"原始 query 数: {stats['total']}")
    print(f"保留 query 数: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"\n移除原因:")
    print(f"  全正样本: {stats['removed_all_pos']}")
    print(f"  全负样本: {stats['removed_all_neg']}")
    print(f"  正样本比例过高 (>{MAX_POS_RATIO*100:.0f}%): {stats['removed_high_pos_ratio']}")
    print(f"  负样本数量不足 (<{MIN_NEG_COUNT}): {stats['removed_low_neg_count']}")
    print(f"  正样本数量不足 (<{MIN_POS_COUNT}): {stats['removed_low_pos_count']}")
    
    # 分析筛选后的数据
    if stats['kept'] > 0:
        pos_counts = []
        neg_counts = []
        pos_ratios = []
        
        for qid, qdata in filtered_data.items():
            if qid == '_meta':
                continue
            candidates = qdata.get('pointer_candidates', [])
            pos = sum(1 for c in candidates if c.get('vqa_correct', 0) == 1)
            neg = sum(1 for c in candidates if c.get('vqa_correct', 0) == 0)
            total = len(candidates)
            
            pos_counts.append(pos)
            neg_counts.append(neg)
            if total > 0:
                pos_ratios.append(pos / total)
        
        total_pos = sum(pos_counts)
        total_neg = sum(neg_counts)
        overall_pos_ratio = total_pos / (total_pos + total_neg)
        
        print(f"\n{'='*70}")
        print("筛选后数据统计")
        print('='*70)
        print(f"总正样本数: {total_pos}")
        print(f"总负样本数: {total_neg}")
        print(f"整体正样本比例: {overall_pos_ratio*100:.1f}%")
        print(f"整体负样本比例: {(1-overall_pos_ratio)*100:.1f}%")
        
        print(f"\n每个 query 的正样本比例:")
        print(f"  平均: {np.mean(pos_ratios)*100:.1f}%")
        print(f"  中位数: {np.median(pos_ratios)*100:.1f}%")
        print(f"  最小: {min(pos_ratios)*100:.1f}%")
        print(f"  最大: {max(pos_ratios)*100:.1f}%")
        
        print(f"\n每个 query 的正样本数量:")
        print(f"  平均: {np.mean(pos_counts):.1f}")
        print(f"  最小: {min(pos_counts)}")
        print(f"  最大: {max(pos_counts)}")
        
        print(f"\n每个 query 的负样本数量:")
        print(f"  平均: {np.mean(neg_counts):.1f}")
        print(f"  最小: {min(neg_counts)}")
        print(f"  最大: {max(neg_counts)}")
    
    # 保存
    print(f"\n保存筛选后数据: {OUTPUT_PATH}")
    save_json(filtered_data, OUTPUT_PATH)
    
    print("\n✓ 完成!")


if __name__ == '__main__':
    main()
