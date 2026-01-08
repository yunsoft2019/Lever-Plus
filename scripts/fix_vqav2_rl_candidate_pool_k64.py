#!/usr/bin/env python3
"""
修复 VQAv2 RL 候选池维度一致性：直接用轨迹中的 ICD 构建新候选池

核心思路（与 OKVQA 相同）：
- 原始 64 候选池找不到正样本，所以才去全局搜索
- 全局搜索找到了正样本 ICD，这些 ICD 不在原始 64 里
- 直接用全局搜索找到的 ICD 构建新候选池
- Pointer 是根据 embedding 选的，候选池内容换了也没问题

修复步骤：
1. 收集每个 query 所有轨迹用到的 ICD（全局索引）
2. 构建新候选池（用到的 ICD + 补齐到 64）
3. 建立全局→局部映射
4. 转换轨迹索引为局部索引 [0, 63]
5. 保存 candidate_pool_ids 字段

使用方法：
    python scripts/fix_vqav2_rl_candidate_pool_k64.py \
        --input_path results/vqav2/generated_data/rl_data_merged.json \
        --sampler_cache_path "results/vqav2/cache/vqav2-RandSampler-anchor_sample_num: 5000:64.json" \
        --output_path results/vqav2/generated_data/rl_data_k64_v3.json

作者: Lever-Plus Team
日期: 2026-01-04
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
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fix_single_query(
    query_id: str,
    qdata: dict,
    original_pool: list,
    target_pool_size: int = 64,
    all_candidate_indices: list = None
) -> tuple:
    """
    修复单个 query 的候选池和轨迹
    
    Args:
        query_id: query ID
        qdata: query 数据（包含 pointer_candidates）
        original_pool: 原始 64 候选池（sampler 缓存）
        target_pool_size: 目标候选池大小（必须为 64）
        all_candidate_indices: 所有候选索引（用于补齐）
    
    Returns:
        (new_qdata, stats)
    """
    candidates = qdata.get('pointer_candidates', [])
    
    if len(candidates) == 0:
        return None, {'error': 'no_candidates'}
    
    # Step 1: 收集所有轨迹用到的 ICD（全局索引）
    used_icds = set()
    for traj in candidates:
        pointer = traj.get('pointer', [])
        if len(pointer) >= 2:
            used_icds.update(pointer[:2])
    
    used_icds = list(used_icds)
    
    # Step 2: 构建新候选池
    # 优先使用轨迹中用到的 ICD，然后从原始 64 补齐，最后从全局补齐
    new_pool = list(used_icds)
    
    # 从原始 64 补齐（优先）
    if len(new_pool) < target_pool_size and original_pool:
        for idx in original_pool:
            if idx not in new_pool:
                new_pool.append(idx)
                if len(new_pool) >= target_pool_size:
                    break
    
    # 从全局补齐（如果还不够）
    if len(new_pool) < target_pool_size and all_candidate_indices:
        for idx in all_candidate_indices:
            if idx not in new_pool:
                new_pool.append(idx)
                if len(new_pool) >= target_pool_size:
                    break
    
    # 截断到目标大小
    new_pool = new_pool[:target_pool_size]
    
    # Step 3: 建立全局→局部映射
    global_to_local = {gid: lid for lid, gid in enumerate(new_pool)}
    
    # Step 4: 转换轨迹索引
    new_candidates = []
    skipped = 0
    
    for traj in candidates:
        pointer_global = traj.get('pointer', [])
        if len(pointer_global) < 2:
            skipped += 1
            continue
        
        # 检查所有 ICD 是否在新候选池中
        try:
            pointer_local = [global_to_local[gid] for gid in pointer_global[:2]]
        except KeyError as e:
            # 这不应该发生，因为我们已经把所有用到的 ICD 加入候选池
            print(f"警告: Query {query_id} 轨迹 ICD {e} 不在新候选池中，跳过")
            skipped += 1
            continue
        
        # 创建新轨迹
        new_traj = deepcopy(traj)
        new_traj['pointer_global'] = pointer_global[:2]  # 保存原始全局索引
        new_traj['pointer'] = pointer_local  # 使用局部索引
        new_candidates.append(new_traj)
    
    if len(new_candidates) == 0:
        return None, {'error': 'no_valid_candidates'}
    
    # Step 5: 构建新的 query 数据
    new_qdata = {
        'query': qdata.get('query', {}),
        'candidate_pool_ids': new_pool,  # 关键：保存候选池信息
        'pointer_candidates': new_candidates,
    }
    
    # 统计
    positive_count = sum(1 for t in new_candidates if t.get('vqa_correct', 0) == 1)
    negative_count = sum(1 for t in new_candidates if t.get('vqa_correct', 0) == 0)
    
    # 计算有多少 ICD 来自原始 64
    original_pool_set = set(original_pool) if original_pool else set()
    from_original = sum(1 for idx in new_pool if idx in original_pool_set)
    from_global = len(new_pool) - from_original
    
    stats = {
        'success': True,
        'pool_size': len(new_pool),
        'trajectory_count': len(new_candidates),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'skipped_trajectories': skipped,
        'icds_from_original_64': from_original,
        'icds_from_global': from_global,
        'used_icds_count': len(used_icds),
    }
    
    return new_qdata, stats


def main():
    parser = argparse.ArgumentParser(description="修复VQAv2 RL候选池维度一致性")
    parser.add_argument("--input_path", type=str, required=True, help="输入RL数据路径")
    parser.add_argument("--sampler_cache_path", type=str, default=None, help="sampler缓存路径（可选，用于补齐候选池）")
    parser.add_argument("--output_path", type=str, required=True, help="输出修复后数据路径")
    parser.add_argument("--target_pool_size", type=int, default=64, help="目标候选池大小（必须为64）")
    parser.add_argument("--total_candidates", type=int, default=443757, help="总候选数量（用于补齐）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"加载RL数据: {args.input_path}")
    rl_data = load_json(args.input_path)
    
    # 加载 sampler 缓存（可选）
    sampler_cache = {}
    if args.sampler_cache_path and os.path.exists(args.sampler_cache_path):
        print(f"加载sampler缓存: {args.sampler_cache_path}")
        sampler_cache = load_json(args.sampler_cache_path)
        print(f"  sampler缓存包含 {len(sampler_cache)} 个query的候选池")
    
    # 所有候选索引（用于补齐）
    all_candidate_indices = list(range(args.total_candidates))
    
    # 统计优化前的状态
    queries = {k: v for k, v in rl_data.items() if k != '_meta'}
    
    print(f"\n优化前统计:")
    print(f"  总query数: {len(queries)}")
    
    # 分析当前轨迹的索引范围
    all_pointer_indices = set()
    for qid, qdata in queries.items():
        for traj in qdata.get('pointer_candidates', []):
            pointer = traj.get('pointer', [])
            all_pointer_indices.update(pointer)
    
    if all_pointer_indices:
        print(f"  当前轨迹索引范围: [{min(all_pointer_indices)}, {max(all_pointer_indices)}]")
        print(f"  唯一ICD数量: {len(all_pointer_indices)}")
    
    # 修复每个 query
    print(f"\n开始修复候选池（目标K={args.target_pool_size}）...")
    fixed_data = {}
    
    # 保留 _meta
    if '_meta' in rl_data:
        fixed_data['_meta'] = deepcopy(rl_data['_meta'])
        fixed_data['_meta']['fixed_at'] = datetime.now().isoformat()
        fixed_data['_meta']['fix_version'] = 'v3'
        fixed_data['_meta']['fix_description'] = '直接用轨迹中的ICD构建新候选池，转换为局部索引'
        fixed_data['_meta']['fix_params'] = {
            'target_pool_size': args.target_pool_size,
        }
    
    # 统计
    stats_summary = {
        'success': 0,
        'failed': 0,
        'total_trajectories': 0,
        'total_positive': 0,
        'total_negative': 0,
        'pool_sizes': [],
        'icds_from_original': [],
        'icds_from_global': [],
    }
    
    for qid in tqdm(queries.keys(), desc="修复query"):
        qdata = queries[qid]
        original_pool = sampler_cache.get(qid, [])
        
        new_qdata, stats = fix_single_query(
            query_id=qid,
            qdata=qdata,
            original_pool=original_pool,
            target_pool_size=args.target_pool_size,
            all_candidate_indices=all_candidate_indices
        )
        
        if new_qdata is None:
            stats_summary['failed'] += 1
            # 保留原始数据（但标记为未修复）
            fixed_data[qid] = qdata
            fixed_data[qid]['_fix_failed'] = True
            fixed_data[qid]['_fix_error'] = stats.get('error', 'unknown')
            continue
        
        fixed_data[qid] = new_qdata
        stats_summary['success'] += 1
        stats_summary['total_trajectories'] += stats['trajectory_count']
        stats_summary['total_positive'] += stats['positive_count']
        stats_summary['total_negative'] += stats['negative_count']
        stats_summary['pool_sizes'].append(stats['pool_size'])
        stats_summary['icds_from_original'].append(stats['icds_from_original_64'])
        stats_summary['icds_from_global'].append(stats['icds_from_global'])
    
    # 打印统计
    print(f"\n" + "=" * 70)
    print("修复结果统计")
    print("=" * 70)
    print(f"成功修复: {stats_summary['success']}")
    print(f"修复失败: {stats_summary['failed']}")
    
    if stats_summary['pool_sizes']:
        print(f"\n候选池大小:")
        print(f"  平均: {np.mean(stats_summary['pool_sizes']):.1f}")
        print(f"  最小: {min(stats_summary['pool_sizes'])}")
        print(f"  最大: {max(stats_summary['pool_sizes'])}")
        equal_64 = sum(1 for s in stats_summary['pool_sizes'] if s == args.target_pool_size)
        print(f"  等于{args.target_pool_size}: {equal_64}/{len(stats_summary['pool_sizes'])}")
    
    if stats_summary['icds_from_original']:
        print(f"\nICD来源分布:")
        print(f"  来自原始64的ICD (平均): {np.mean(stats_summary['icds_from_original']):.1f}")
        print(f"  来自全局搜索的ICD (平均): {np.mean(stats_summary['icds_from_global']):.1f}")
    
    print(f"\n轨迹统计:")
    print(f"  总轨迹数: {stats_summary['total_trajectories']}")
    print(f"  正样本数: {stats_summary['total_positive']}")
    print(f"  负样本数: {stats_summary['total_negative']}")
    if stats_summary['total_trajectories'] > 0:
        pos_ratio = stats_summary['total_positive'] / stats_summary['total_trajectories'] * 100
        print(f"  正样本比例: {pos_ratio:.1f}%")
    
    # 统计 All-Zero query
    all_zero_count = 0
    for qid, qdata in fixed_data.items():
        if qid == '_meta':
            continue
        candidates = qdata.get('pointer_candidates', [])
        positive_count = sum(1 for t in candidates if t.get('vqa_correct', 0) == 1)
        if positive_count == 0:
            all_zero_count += 1
    
    print(f"\nAll-Zero query数: {all_zero_count} ({all_zero_count/len(queries)*100:.1f}%)")
    
    # 保存
    print(f"\n保存修复后数据: {args.output_path}")
    save_json(fixed_data, args.output_path)
    
    # 验证断言
    print(f"\n验证断言...")
    assertion_passed = True
    
    for qid, qdata in fixed_data.items():
        if qid == '_meta':
            continue
        if qdata.get('_fix_failed'):
            continue
        
        pool = qdata.get('candidate_pool_ids', [])
        trajectories = qdata.get('pointer_candidates', [])
        
        # 断言1：候选池大小
        if len(pool) != args.target_pool_size:
            print(f"  ⚠️ Query {qid}: 候选池大小 {len(pool)} != {args.target_pool_size}")
            assertion_passed = False
        
        # 断言2：轨迹索引在有效范围内 [0, 63]
        for traj in trajectories:
            pointer = traj.get('pointer', [])
            for pos in pointer:
                if pos < 0 or pos >= len(pool):
                    print(f"  ⚠️ Query {qid}: 轨迹索引 {pos} 超出范围 [0, {len(pool)-1}]")
                    assertion_passed = False
        
        # 断言3：pointer_global 存在且与 pointer 对应
        for traj in trajectories:
            pointer_local = traj.get('pointer', [])
            pointer_global = traj.get('pointer_global', [])
            if len(pointer_global) != len(pointer_local):
                print(f"  ⚠️ Query {qid}: pointer_global 长度不匹配")
                assertion_passed = False
            else:
                # 验证映射正确性
                for local_idx, global_idx in zip(pointer_local, pointer_global):
                    if pool[local_idx] != global_idx:
                        print(f"  ⚠️ Query {qid}: 索引映射错误 local={local_idx} -> pool[{local_idx}]={pool[local_idx]} != global={global_idx}")
                        assertion_passed = False
    
    if assertion_passed:
        print("  ✓ 所有断言通过")
    else:
        print("  ⚠️ 部分断言失败，请检查数据")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
