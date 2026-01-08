#!/usr/bin/env python3
"""
合并所有 balanced RL 数据文件

使用方法：
    python scripts/merge_balanced_rl_data.py
"""

import json
import os
from glob import glob
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # 找到所有 balanced 数据文件
    pattern = os.path.join(PROJECT_ROOT, 'results/vqav2/generated_data/rl_data_balanced_*_*.json')
    files = sorted(glob(pattern))
    
    print(f"找到 {len(files)} 个数据文件")
    
    # 合并数据
    merged_data = {
        '_meta': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_files': [os.path.basename(f) for f in files],
            'num_source_files': len(files),
        }
    }
    
    total_queries = 0
    total_candidates = 0
    total_pos = 0
    
    for f in files:
        try:
            d = json.load(open(f))
            for k, v in d.items():
                if k == '_meta':
                    continue
                merged_data[k] = v
                total_queries += 1
                cands = v.get('pointer_candidates', [])
                total_candidates += len(cands)
                total_pos += sum(1 for c in cands if c.get('vqa_acc_score', 0) > 0)
        except Exception as e:
            print(f"  警告: {os.path.basename(f)} 读取失败 - {e}")
    
    # 更新 meta
    merged_data['_meta']['total_queries'] = total_queries
    merged_data['_meta']['total_candidates'] = total_candidates
    merged_data['_meta']['positive_ratio'] = total_pos / total_candidates if total_candidates > 0 else 0
    
    # 保存
    output_path = os.path.join(PROJECT_ROOT, 'results/vqav2/generated_data/rl_data_balanced_merged.json')
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("合并完成!")
    print("=" * 60)
    print(f"总 queries: {total_queries}")
    print(f"总 candidates: {total_candidates}")
    print(f"正样本比例: {total_pos/total_candidates*100:.1f}%")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()
