#!/usr/bin/env python3
"""
合并多个GPU生成的平衡RL数据

用法:
    python scripts/merge_balanced_data.py --output results/vqav2/generated_data/rl_data_balanced_merged.json
"""

import os
import sys
import json
import argparse
from glob import glob
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", type=str, 
                        default="results/vqav2/generated_data/rl_data_balanced_*_*.json",
                        help="输入文件的glob模式")
    parser.add_argument("--output", type=str,
                        default="results/vqav2/generated_data/rl_data_balanced_merged.json",
                        help="合并后的输出文件")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="训练集比例")
    args = parser.parse_args()
    
    # 查找所有输入文件
    input_pattern = os.path.join(PROJECT_ROOT, args.input_pattern)
    input_files = sorted(glob(input_pattern))
    
    # 排除已合并的文件
    input_files = [f for f in input_files if 'merged' not in f]
    
    print("=" * 60)
    print("合并平衡RL数据")
    print("=" * 60)
    print(f"找到 {len(input_files)} 个文件:")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    print("=" * 60)
    
    if not input_files:
        print("错误: 没有找到输入文件!")
        return
    
    # 合并数据
    merged_data = {}
    total_queries = 0
    
    for input_file in input_files:
        print(f"\n加载: {os.path.basename(input_file)}")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # 跳过元数据
        for key, value in data.items():
            if key.startswith('_'):
                continue
            if key not in merged_data:
                merged_data[key] = value
                total_queries += 1
        
        print(f"  累计Query数: {total_queries}")
    
    # 计算统计信息
    pos_ratios = []
    for key, value in merged_data.items():
        if key.startswith('_'):
            continue
        candidates = value.get('pointer_candidates', [])
        if candidates:
            scores = [c['vqa_acc_score'] for c in candidates]
            pos_count = sum(1 for s in scores if s > 0)
            pos_ratio = pos_count / len(scores)
            pos_ratios.append(pos_ratio)
    
    avg_pos_ratio = sum(pos_ratios) / len(pos_ratios) if pos_ratios else 0
    
    # 分割训练集和验证集
    query_ids = [k for k in merged_data.keys() if not k.startswith('_')]
    import random
    random.seed(42)
    random.shuffle(query_ids)
    
    train_size = int(len(query_ids) * args.train_ratio)
    train_ids = set(query_ids[:train_size])
    val_ids = set(query_ids[train_size:])
    
    # 添加元数据
    merged_data['_meta'] = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_files': [os.path.basename(f) for f in input_files],
        'total_queries': total_queries,
        'train_queries': len(train_ids),
        'val_queries': len(val_ids),
        'avg_pos_ratio': avg_pos_ratio,
        'train_ids': list(train_ids),
        'val_ids': list(val_ids),
    }
    
    # 保存
    output_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("合并完成!")
    print("=" * 60)
    print(f"总Query数: {total_queries}")
    print(f"训练集: {len(train_ids)}")
    print(f"验证集: {len(val_ids)}")
    print(f"平均正样本比例: {avg_pos_ratio*100:.1f}%")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
