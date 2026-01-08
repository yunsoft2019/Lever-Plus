#!/usr/bin/env python3
"""
合并多 GPU 生成的 RL 数据分片

使用方法:
    python scripts/merge_rl_data_shards.py [--dataset vqav2] [--output_path <path>]

默认会合并 results/<dataset>/generated_data/rl_data_shards/shard_gpu*.json
输出到 results/<dataset>/generated_data/rl_data_v3.json
"""

import argparse
import json
import os
from glob import glob
from datetime import datetime


def merge_shards(shard_dir: str, output_path: str, dataset: str = "vqav2"):
    """
    合并所有分片文件
    
    Args:
        shard_dir: 分片目录
        output_path: 输出路径
        dataset: 数据集名称
    """
    # 查找所有分片文件
    shard_pattern = os.path.join(shard_dir, "shard_gpu*.json")
    shard_files = sorted(glob(shard_pattern))
    
    if not shard_files:
        print(f"错误：未找到分片文件: {shard_pattern}")
        return
    
    print(f"找到 {len(shard_files)} 个分片文件:")
    for f in shard_files:
        print(f"  - {f}")
    
    # 合并数据
    merged_data = {}
    meta_info = None
    total_queries = 0
    
    for shard_file in shard_files:
        print(f"\n加载: {shard_file}")
        try:
            with open(shard_file, "r") as f:
                shard_data = json.load(f)
            
            # 提取 _meta 信息（只保留第一个）
            if "_meta" in shard_data:
                if meta_info is None:
                    meta_info = shard_data.pop("_meta")
                else:
                    shard_data.pop("_meta")
            
            # 合并数据
            shard_count = len(shard_data)
            merged_data.update(shard_data)
            total_queries += shard_count
            print(f"  - 包含 {shard_count} 个 query")
            
        except Exception as e:
            print(f"  - 错误: {e}")
    
    print(f"\n合并完成:")
    print(f"  - 总 query 数: {len(merged_data)}")
    print(f"  - 预期 query 数: {total_queries}")
    
    # 检查是否有重复
    if len(merged_data) != total_queries:
        print(f"  ⚠️  警告: 有 {total_queries - len(merged_data)} 个重复的 query")
    
    # 更新 _meta 信息
    if meta_info is None:
        meta_info = {}
    
    meta_info["merged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_info["num_shards"] = len(shard_files)
    meta_info["total_queries"] = len(merged_data)
    meta_info["dataset"] = dataset
    
    # 构建输出数据
    output_data = {"_meta": meta_info, **merged_data}
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\n保存到: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("✓ 合并完成！")
    
    # 统计信息
    print(f"\n统计信息:")
    total_candidates = 0
    correct_count = 0
    for query_id, query_data in merged_data.items():
        candidates = query_data.get("pointer_candidates", [])
        total_candidates += len(candidates)
        for c in candidates:
            if c.get("vqa_correct", 0) == 1:
                correct_count += 1
    
    if total_candidates > 0:
        avg_candidates = total_candidates / len(merged_data)
        correct_ratio = correct_count / total_candidates * 100
        print(f"  - 平均每 query 候选数: {avg_candidates:.1f}")
        print(f"  - 正确候选比例: {correct_ratio:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="合并 RL 数据分片")
    parser.add_argument("--dataset", type=str, default="vqav2", help="数据集名称")
    parser.add_argument("--shard_dir", type=str, help="分片目录（默认: results/<dataset>/generated_data/rl_data_shards）")
    parser.add_argument("--output_path", type=str, help="输出路径（默认: results/<dataset>/generated_data/rl_data_v3.json）")
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.shard_dir is None:
        args.shard_dir = f"./results/{args.dataset}/generated_data/rl_data_shards"
    
    if args.output_path is None:
        args.output_path = f"./results/{args.dataset}/generated_data/rl_data_v3.json"
    
    print("=" * 50)
    print("合并 RL 数据分片")
    print("=" * 50)
    print(f"数据集: {args.dataset}")
    print(f"分片目录: {args.shard_dir}")
    print(f"输出路径: {args.output_path}")
    print("=" * 50)
    
    merge_shards(args.shard_dir, args.output_path, args.dataset)


if __name__ == "__main__":
    main()
