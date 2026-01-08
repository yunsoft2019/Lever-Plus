#!/usr/bin/env python3
"""合并 RL 数据分片文件"""

import json
import argparse
from pathlib import Path


def merge_shards(shard_dir: str, output_file: str, pattern: str = "shard_*.json"):
    """合并所有分片文件"""
    shard_path = Path(shard_dir)
    shard_files = sorted(shard_path.glob(pattern))
    
    if not shard_files:
        print(f"❌ 未找到匹配的分片文件: {shard_dir}/{pattern}")
        return
    
    print(f"找到 {len(shard_files)} 个分片文件:")
    for f in shard_files:
        print(f"  - {f.name}")
    
    all_data = {}
    meta_info = None
    total_raw = 0
    
    for shard_file in shard_files:
        with open(shard_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 提取 _meta（只保留第一个）
            if '_meta' in data:
                if meta_info is None:
                    meta_info = data.pop('_meta')
                else:
                    data.pop('_meta')
            count = len(data)
            total_raw += count
            print(f"  {shard_file.name}: {count} 条数据")
            # 合并（自动去重，后面的覆盖前面的）
            all_data.update(data)
    
    duplicates = total_raw - len(all_data)
    if duplicates > 0:
        print(f"\n去重: 移除 {duplicates} 条重复数据")
    
    # 添加 _meta
    output_data = {}
    if meta_info:
        output_data['_meta'] = meta_info
    output_data.update(all_data)
    
    # 保存合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 合并完成！")
    print(f"  - 总数据量: {len(all_data)} 条")
    print(f"  - 输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 RL 数据分片")
    parser.add_argument("--shard_dir", type=str, 
                        default="./results/vqav2/generated_data/rl_data_shards",
                        help="分片文件目录")
    parser.add_argument("--output", type=str,
                        default="./results/vqav2/generated_data/rl_data_merged.json",
                        help="合并后的输出文件")
    parser.add_argument("--pattern", type=str, default="shard_*.json",
                        help="分片文件匹配模式")
    
    args = parser.parse_args()
    merge_shards(args.shard_dir, args.output, args.pattern)
