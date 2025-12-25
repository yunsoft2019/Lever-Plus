#!/usr/bin/env python3
"""
修复 RL 候选池维度一致性 V4：保持原始候选池顺序

核心改进（相比 V3）：
- V3 问题：新候选池的顺序与原始 64 不一致，导致 SFT 模型的先验知识失效
- V4 解决：保持原始候选池的顺序，只在末尾添加全局搜索找到的新 ICD

修复步骤：
1. 收集每个 query 所有轨迹用到的 ICD（全局索引）
2. 构建新候选池：
   a. 先放原始 64 候选池（保持原始顺序！）
   b. 再在末尾添加全局搜索找到的新 ICD（不在原始 64 中的）
   c. 截断到 64
3. 建立全局→局部映射
4. 转换轨迹索引为局部索引 [0, 63]
5. 保存 candidate_pool_ids 字段

关键：原始候选池中的 ICD 保持原始位置，SFT 模型的先验知识得以保留！

使用方法：
    python scripts/fix_rl_candidate_pool_k64_v4.py \
        --input_path results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_final_merged.json \
        --sampler_cache_path "results/okvqa/cache/okvqa-RandSampler-anchor_sample_num: 800:64.json" \
        --output_path results/okvqa/generated_data/rl_data_k64_v4.json

作者: Lever-Plus Team
日期: 2025-12-20
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


def fix_single_query_v4(
    query_id: str,
    qdata: dict,
    original_pool: list,
    target_pool_size: int = 64,
    all_candidate_indices: list = None
) -> tuple:
    """
    修复单个 query 的候选池和轨迹（V4：保持原始顺序）
    
    Args:
        query_id: query ID
        qdata: query 数据（包含 pointer_candidates）
        original_pool: 原始 64 候选池（sampler 缓存，顺序很重要！）
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
    
    # Step 2: 构建新候选池（关键改动：保持原始顺序！）
    original_pool_set = set(original_pool) if original_pool else set()
    
    # 2a. 先放原始 64 候选池（保持原始顺序！）
    new_pool = list(original_pool) if original_pool else []
    
    # 2b. 找出轨迹中用到但不在原始 64 中的 ICD
    new_icds_from_global = [icd for icd in used_icds if icd not in original_pool_set]
    
    # 2c. 在末尾添加新 ICD（替换原始 64 中未被使用的位置）
    # 策略：如果原始 64 中有些 ICD 没被任何轨迹使用，可以用新 ICD 替换
    # 但为了保持 SFT 先验，我们优先保留原始位置
    
    # 检查哪些原始 ICD 被轨迹使用了
    used_original_icds = original_pool_set & used_icds
    unused_original_icds = original_pool_set - used_icds
    
    # 如果有新 ICD 需要加入，替换未使用的原始 ICD
    if new_icds_from_global:
        # 找出原始池中未被使用的位置
        unused_positions = []
        for i, icd in enumerate(new_pool):
            if icd in unused_original_icds:
                unused_positions.append(i)
        
        # 用新 ICD 替换未使用的位置（从后往前替换，尽量保留前面的位置）
        unused_positions = sorted(unused_positions, reverse=True)
        
        for new_icd in new_icds_from_global:
            if unused_positions:
                # 替换一个未使用的位置
                pos = unused_positions.pop()
                new_pool[pos] = new_icd
            elif len(new_pool) < target_pool_size:
                # 如果没有可替换的位置，且池子还没满，追加到末尾
                new_pool.append(new_icd)
            else:
                # 池子已满且没有可替换位置，无法加入这个 ICD
                # 这种情况下，对应的轨迹会被跳过
                pass
    
    # 截断到目标大小
    new_pool = new_pool[:target_pool_size]
    
    # Step 3: 建立全局→局部映射
    global_to_local = {gid: lid for lid, gid in enumerate(new_pool)}
    new_pool_set = set(new_pool)
    
    # Step 4: 转换轨迹索引
    new_candidates = []
    skipped = 0
    skipped_reasons = defaultdict(int)
    
    for traj in candidates:
        pointer_global = traj.get('pointer', [])
        if len(pointer_global) < 2:
            skipped += 1
            skipped_reasons['pointer_too_short'] += 1
            continue
        
        # 检查所有 ICD 是否在新候选池中
        all_in_pool = all(gid in new_pool_set for gid in pointer_global[:2])
        if not all_in_pool:
            skipped += 1
            skipped_reasons['icd_not_in_pool'] += 1
            continue
        
        # 转换为局部索引
        pointer_local = [global_to_local[gid] for gid in pointer_global[:2]]
        
        # 创建新轨迹
        new_traj = deepcopy(traj)
        new_traj['pointer_global'] = pointer_global[:2]  # 保存原始全局索引
        new_traj['pointer'] = pointer_local  # 使用局部索引
        new_candidates.append(new_traj)
    
    if len(new_candidates) == 0:
        return None, {'error': 'no_valid_candidates', 'skipped_reasons': dict(skipped_reasons)}
    
    # Step 5: 构建新的 query 数据
    new_qdata = {
        'query': qdata.get('query', {}),
        'candidate_pool_ids': new_pool,  # 关键：保存候选池信息
        'pointer_candidates': new_candidates,
    }
    
    # 统计
    positive_count = sum(1 for t in new_candidates if t.get('vqa_correct', 0) == 1)
    negative_count = sum(1 for t in new_candidates if t.get('vqa_correct', 0) == 0)
    
    # 计算候选池来源统计
    from_original = sum(1 for idx in new_pool if idx in original_pool_set)
    from_global = len(new_pool) - from_original
    
    # 计算位置保持率（原始位置被保留的比例）
    position_preserved = 0
    if original_pool:
        for i, icd in enumerate(new_pool):
            if i < len(original_pool) and original_pool[i] == icd:
                position_preserved += 1
    
    stats = {
        'success': True,
        'pool_size': len(new_pool),
        'trajectory_count': len(new_candidates),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'skipped_trajectories': skipped,
        'skipped_reasons': dict(skipped_reasons),
        'icds_from_original_64': from_original,
        'icds_from_global': from_global,
        'used_icds_count': len(used_icds),
        'position_preserved': position_preserved,  # 新增：位置保持数
        'position_preserved_ratio': position_preserved / target_pool_size if target_pool_size > 0 else 0,
    }
    
    return new_qdata, stats


def main():
    parser = argparse.ArgumentParser(description="修复RL候选池维度一致性 V4（保持原始顺序）")
    parser.add_argument("--input_path", type=str, required=True, help="输入RL数据路径")
    parser.add_argument("--sampler_cache_path", type=str, required=True, help="sampler缓存路径（必需，用于保持原始顺序）")
    parser.add_argument("--output_path", type=str, required=True, help="输出修复后数据路径")
    parser.add_argument("--target_pool_size", type=int, default=64, help="目标候选池大小（必须为64）")
    parser.add_argument("--total_candidates", type=int, default=9009, help="总候选数量（用于补齐）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"加载RL数据: {args.input_path}")
    rl_data = load_json(args.input_path)
    
    # 加载 sampler 缓存（必需）
    if not os.path.exists(args.sampler_cache_path):
        raise FileNotFoundError(f"sampler缓存文件不存在: {args.sampler_cache_path}")
    
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
    print(f"\n开始修复候选池（目标K={args.target_pool_size}，保持原始顺序）...")
    fixed_data = {}
    
    # 保留 _meta
    if '_meta' in rl_data:
        fixed_data['_meta'] = deepcopy(rl_data['_meta'])
        fixed_data['_meta']['fixed_at'] = datetime.now().isoformat()
        fixed_data['_meta']['fix_version'] = 'v4'
        fixed_data['_meta']['fix_description'] = '保持原始候选池顺序，只在末尾添加新ICD（保留SFT先验）'
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
        'position_preserved': [],  # 新增
    }
    
    for qid in tqdm(queries.keys(), desc="修复query"):
        qdata = queries[qid]
        original_pool = sampler_cache.get(qid, [])
        
        if not original_pool:
            print(f"警告: Query {qid} 没有原始候选池，跳过")
            stats_summary['failed'] += 1
            fixed_data[qid] = qdata
            fixed_data[qid]['_fix_failed'] = True
            fixed_data[qid]['_fix_error'] = 'no_original_pool'
            continue
        
        new_qdata, stats = fix_single_query_v4(
            query_id=qid,
            qdata=qdata,
            original_pool=original_pool,
            target_pool_size=args.target_pool_size,
            all_candidate_indices=all_candidate_indices
        )
        
        if new_qdata is None:
            stats_summary['failed'] += 1
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
        stats_summary['position_preserved'].append(stats['position_preserved'])
    
    # 打印统计
    print(f"\n" + "=" * 70)
    print("修复结果统计（V4：保持原始顺序）")
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
    
    if stats_summary['position_preserved']:
        print(f"\n【关键】位置保持统计（SFT先验保留程度）:")
        print(f"  平均位置保持数: {np.mean(stats_summary['position_preserved']):.1f} / 64")
        print(f"  平均位置保持率: {np.mean(stats_summary['position_preserved']) / 64 * 100:.1f}%")
        print(f"  最小位置保持数: {min(stats_summary['position_preserved'])}")
        print(f"  最大位置保持数: {max(stats_summary['position_preserved'])}")
    
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
        if qdata.get('_fix_failed'):
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
