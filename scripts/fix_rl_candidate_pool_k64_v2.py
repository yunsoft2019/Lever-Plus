#!/usr/bin/env python3
"""
修复 RL 候选池维度一致性 + 全局正样本注入 64 ICD 子池 (V2)

核心问题：
- SFT阶段：每个query有固定的64个候选池（由sampler生成）
- RL数据生成阶段：使用了全局搜索，找到的ICD可能不在原始64个候选池中
- 这导致RL训练时的候选池和SFT不一致

解决方案：
1. 保持每个query的候选池严格为64（与SFT一致）
2. 将RL数据中的全局索引映射回原始64个候选池
3. 对于不在原始64中的正样本ICD，注入到候选池中（替换低质量ICD）
4. 控制轨迹数量和正负样本比例

使用方法：
    python scripts/fix_rl_candidate_pool_k64_v2.py \
        --input_path results/okvqa/generated_data/rl_data.json \
        --sampler_cache_path "results/okvqa/cache/okvqa-RandSampler-anchor_sample_num: 800:64.json" \
        --output_path results/okvqa/generated_data/rl_data_k64_fixed.json \
        --target_trajectory_num 12 \
        --max_inject_icds 8
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


def analyze_query_status(qdata):
    """分析单个query的状态"""
    candidates = qdata.get('pointer_candidates', [])
    
    positive_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 1)
    negative_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 0)
    
    # 收集当前用到的ICD索引（全局索引）
    used_indices = set()
    for c in candidates:
        for idx in c.get('pointer', []):
            used_indices.add(idx)
    
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'total_count': len(candidates),
        'used_indices': used_indices,
        'is_all_zero': positive_count == 0
    }


def collect_trajectories_by_pool_status(qdata, original_pool):
    """
    将轨迹按照是否在原始候选池中分类
    
    返回：
    - in_pool_trajectories: 两个ICD都在原始64候选池中的轨迹
    - partial_pool_trajectories: 只有一个ICD在原始64候选池中的轨迹
    - out_pool_trajectories: 两个ICD都不在原始64候选池中的轨迹
    """
    pool_set = set(original_pool)
    
    in_pool = []
    partial_pool = []
    out_pool = []
    
    for traj in qdata.get('pointer_candidates', []):
        pointer = traj.get('pointer', [])
        if len(pointer) < 2:
            continue
        
        in_pool_count = sum(1 for idx in pointer[:2] if idx in pool_set)
        
        if in_pool_count == 2:
            in_pool.append(traj)
        elif in_pool_count == 1:
            partial_pool.append(traj)
        else:
            out_pool.append(traj)
    
    return in_pool, partial_pool, out_pool


def select_icds_to_inject(
    partial_trajectories,
    out_pool_trajectories,
    original_pool,
    max_inject=8
):
    """
    选择需要注入到候选池中的ICD
    
    策略：
    1. 优先从partial_pool中选择（只需注入1个ICD就能构成正样本）
    2. 其次从out_pool中选择（需要注入2个ICD）
    3. 优先选择正样本轨迹中的ICD
    """
    pool_set = set(original_pool)
    
    # 收集需要注入的ICD及其来源
    inject_candidates = []
    
    # 从partial_pool中收集（只需注入1个）
    for traj in partial_trajectories:
        if traj.get('vqa_correct', 0) != 1:
            continue  # 只考虑正样本
        
        pointer = traj.get('pointer', [])[:2]
        new_icds = [idx for idx in pointer if idx not in pool_set]
        
        if len(new_icds) == 1:
            inject_candidates.append({
                'icds': new_icds,
                'score': traj.get('vqa_acc_score', 0),
                'source': 'partial',
                'trajectory': traj
            })
    
    # 从out_pool中收集（需要注入2个）
    for traj in out_pool_trajectories:
        if traj.get('vqa_correct', 0) != 1:
            continue  # 只考虑正样本
        
        pointer = traj.get('pointer', [])[:2]
        new_icds = [idx for idx in pointer if idx not in pool_set]
        
        if len(new_icds) == 2:
            inject_candidates.append({
                'icds': new_icds,
                'score': traj.get('vqa_acc_score', 0),
                'source': 'out_pool',
                'trajectory': traj
            })
    
    # 按分数排序，优先注入高分正样本的ICD
    inject_candidates.sort(key=lambda x: (-x['score'], len(x['icds'])))
    
    # 选择要注入的ICD
    icds_to_inject = set()
    trajectories_to_add = []
    
    for candidate in inject_candidates:
        needed = len(candidate['icds'])
        if len(icds_to_inject) + needed <= max_inject:
            icds_to_inject.update(candidate['icds'])
            trajectories_to_add.append(candidate['trajectory'])
    
    return list(icds_to_inject), trajectories_to_add


def select_icds_to_remove(original_pool, in_pool_trajectories, num_to_remove):
    """
    选择要从候选池中移除的ICD
    
    策略：优先移除未被任何in_pool轨迹使用的ICD
    """
    pool_set = set(original_pool)
    
    # 统计每个ICD被使用的次数
    usage_count = defaultdict(int)
    for traj in in_pool_trajectories:
        for idx in traj.get('pointer', [])[:2]:
            if idx in pool_set:
                usage_count[idx] += 1
    
    # 按使用次数排序（少的优先移除）
    candidates = [(idx, usage_count.get(idx, 0)) for idx in original_pool]
    candidates.sort(key=lambda x: x[1])
    
    return [idx for idx, _ in candidates[:num_to_remove]]


def build_fixed_candidate_pool(original_pool, icds_to_inject, icds_to_remove):
    """
    构建固定大小的候选池
    """
    new_pool = [idx for idx in original_pool if idx not in icds_to_remove]
    new_pool.extend(icds_to_inject)
    
    # 确保大小为64
    if len(new_pool) > 64:
        new_pool = new_pool[:64]
    
    return new_pool


def convert_trajectory_to_local(traj, global_to_local):
    """
    将轨迹的全局索引转换为局部索引
    """
    pointer = traj.get('pointer', [])[:2]
    
    try:
        local_pointer = [global_to_local[idx] for idx in pointer]
        new_traj = deepcopy(traj)
        new_traj['pointer_global'] = pointer
        new_traj['pointer'] = local_pointer
        return new_traj
    except KeyError:
        return None


def balance_trajectories(trajectories, target_num=12, positive_quota=(2, 4)):
    """
    平衡轨迹数量和正负样本比例
    """
    positives = [t for t in trajectories if t.get('vqa_correct', 0) == 1]
    negatives = [t for t in trajectories if t.get('vqa_correct', 0) == 0]
    
    # 按分数排序
    positives.sort(key=lambda x: -x.get('vqa_acc_score', 0))
    negatives.sort(key=lambda x: -x.get('vqa_acc_score', 0))
    
    # 选择正样本
    min_pos, max_pos = positive_quota
    num_positives = min(max_pos, len(positives))
    num_positives = max(min_pos, num_positives) if len(positives) >= min_pos else len(positives)
    selected_positives = positives[:num_positives]
    
    # 选择负样本
    remaining_slots = target_num - len(selected_positives)
    selected_negatives = negatives[:remaining_slots] if remaining_slots > 0 else []
    
    return selected_positives + selected_negatives


def fix_single_query(
    query_id,
    qdata,
    original_pool,
    target_pool_size=64,
    target_trajectory_num=12,
    max_inject_icds=8
):
    """
    修复单个query的候选池和轨迹
    """
    if not original_pool or len(original_pool) == 0:
        return None, {'error': 'no_original_pool'}
    
    # 分类轨迹
    in_pool, partial_pool, out_pool = collect_trajectories_by_pool_status(qdata, original_pool)
    
    # 统计当前状态
    in_pool_positives = sum(1 for t in in_pool if t.get('vqa_correct', 0) == 1)
    
    # 如果in_pool中已经有足够的正样本，不需要注入
    if in_pool_positives >= 2:
        # 直接使用原始候选池
        global_to_local = {idx: pos for pos, idx in enumerate(original_pool)}
        
        # 转换轨迹
        converted = []
        for traj in in_pool:
            new_traj = convert_trajectory_to_local(traj, global_to_local)
            if new_traj:
                converted.append(new_traj)
        
        # 平衡轨迹
        final_trajectories = balance_trajectories(converted, target_trajectory_num)
        
        new_qdata = {
            'query': qdata.get('query', {}),
            'candidate_pool_ids': original_pool,
            'pointer_candidates': final_trajectories,
            '_fix_info': {
                'original_pool_size': len(original_pool),
                'new_pool_size': len(original_pool),
                'icds_injected': [],
                'icds_removed': [],
                'in_pool_count': len(in_pool),
                'partial_pool_count': len(partial_pool),
                'out_pool_count': len(out_pool),
                'injection_needed': False
            }
        }
        
        final_positive_count = sum(1 for t in final_trajectories if t.get('vqa_correct', 0) == 1)
        
        return new_qdata, {
            'success': True,
            'pool_size': len(original_pool),
            'trajectory_count': len(final_trajectories),
            'positive_count': final_positive_count,
            'was_all_zero': in_pool_positives == 0,
            'is_still_all_zero': final_positive_count == 0,
            'icds_injected': 0
        }
    
    # 需要注入ICD
    icds_to_inject, trajectories_to_add = select_icds_to_inject(
        partial_pool, out_pool, original_pool, max_inject_icds
    )
    
    if icds_to_inject:
        # 选择要移除的ICD
        icds_to_remove = select_icds_to_remove(original_pool, in_pool, len(icds_to_inject))
        
        # 构建新候选池
        new_pool = build_fixed_candidate_pool(original_pool, icds_to_inject, icds_to_remove)
    else:
        new_pool = original_pool
        icds_to_remove = []
    
    # 构建索引映射
    global_to_local = {idx: pos for pos, idx in enumerate(new_pool)}
    
    # 转换in_pool轨迹
    converted = []
    for traj in in_pool:
        new_traj = convert_trajectory_to_local(traj, global_to_local)
        if new_traj:
            converted.append(new_traj)
    
    # 转换并添加注入的轨迹
    for traj in trajectories_to_add:
        new_traj = convert_trajectory_to_local(traj, global_to_local)
        if new_traj:
            new_traj['gen_method'] = 'global_positive_injection'
            converted.append(new_traj)
    
    # 平衡轨迹
    final_trajectories = balance_trajectories(converted, target_trajectory_num)
    
    new_qdata = {
        'query': qdata.get('query', {}),
        'candidate_pool_ids': new_pool,
        'pointer_candidates': final_trajectories,
        '_fix_info': {
            'original_pool_size': len(original_pool),
            'new_pool_size': len(new_pool),
            'icds_injected': icds_to_inject,
            'icds_removed': icds_to_remove,
            'in_pool_count': len(in_pool),
            'partial_pool_count': len(partial_pool),
            'out_pool_count': len(out_pool),
            'trajectories_added': len(trajectories_to_add),
            'injection_needed': True
        }
    }
    
    final_positive_count = sum(1 for t in final_trajectories if t.get('vqa_correct', 0) == 1)
    
    return new_qdata, {
        'success': True,
        'pool_size': len(new_pool),
        'trajectory_count': len(final_trajectories),
        'positive_count': final_positive_count,
        'was_all_zero': in_pool_positives == 0,
        'is_still_all_zero': final_positive_count == 0,
        'icds_injected': len(icds_to_inject)
    }


def main():
    parser = argparse.ArgumentParser(description="修复RL候选池维度一致性 + 全局正样本注入 (V2)")
    parser.add_argument("--input_path", type=str, required=True, help="输入RL数据路径")
    parser.add_argument("--sampler_cache_path", type=str, required=True, help="sampler缓存路径（包含64候选池信息）")
    parser.add_argument("--output_path", type=str, required=True, help="输出修复后数据路径")
    parser.add_argument("--target_pool_size", type=int, default=64, help="目标候选池大小（必须为64）")
    parser.add_argument("--target_trajectory_num", type=int, default=12, help="每个query的目标轨迹数")
    parser.add_argument("--max_inject_icds", type=int, default=8, help="每个query最多注入的ICD数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"加载RL数据: {args.input_path}")
    rl_data = load_json(args.input_path)
    
    print(f"加载sampler缓存: {args.sampler_cache_path}")
    sampler_cache = load_json(args.sampler_cache_path)
    print(f"  sampler缓存包含 {len(sampler_cache)} 个query的候选池")
    
    # 统计优化前的状态
    queries = {k: v for k, v in rl_data.items() if k != '_meta'}
    
    # 分析轨迹分布
    total_in_pool = 0
    total_partial = 0
    total_out_pool = 0
    all_zero_before = 0
    
    for qid, qdata in queries.items():
        original_pool = sampler_cache.get(qid, [])
        if not original_pool:
            continue
        
        in_pool, partial, out_pool = collect_trajectories_by_pool_status(qdata, original_pool)
        total_in_pool += len(in_pool)
        total_partial += len(partial)
        total_out_pool += len(out_pool)
        
        in_pool_positives = sum(1 for t in in_pool if t.get('vqa_correct', 0) == 1)
        if in_pool_positives == 0:
            all_zero_before += 1
    
    print(f"\n优化前统计:")
    print(f"  总query数: {len(queries)}")
    print(f"  轨迹分布:")
    print(f"    - 在原始64候选池中: {total_in_pool}")
    print(f"    - 部分在候选池中: {total_partial}")
    print(f"    - 完全不在候选池中: {total_out_pool}")
    print(f"  All-Zero query数（基于in_pool轨迹）: {all_zero_before} ({all_zero_before/len(queries)*100:.1f}%)")
    
    # 修复每个query
    print(f"\n开始修复候选池（目标K={args.target_pool_size}）...")
    fixed_data = {}
    if '_meta' in rl_data:
        fixed_data['_meta'] = deepcopy(rl_data['_meta'])
        fixed_data['_meta']['fixed_at'] = datetime.now().isoformat()
        fixed_data['_meta']['fix_version'] = 'v2'
        fixed_data['_meta']['fix_params'] = {
            'target_pool_size': args.target_pool_size,
            'target_trajectory_num': args.target_trajectory_num,
            'max_inject_icds': args.max_inject_icds
        }
    
    stats = {
        'success': 0,
        'failed': 0,
        'all_zero_fixed': 0,
        'still_all_zero': 0,
        'pool_size_mismatch': 0,
        'total_icds_injected': 0,
        'queries_with_injection': 0
    }
    
    for qid in tqdm(queries.keys(), desc="修复query"):
        qdata = queries[qid]
        original_pool = sampler_cache.get(qid, [])
        
        if not original_pool:
            stats['failed'] += 1
            fixed_data[qid] = qdata
            continue
        
        new_qdata, result = fix_single_query(
            query_id=qid,
            qdata=qdata,
            original_pool=original_pool,
            target_pool_size=args.target_pool_size,
            target_trajectory_num=args.target_trajectory_num,
            max_inject_icds=args.max_inject_icds
        )
        
        if new_qdata is None:
            stats['failed'] += 1
            fixed_data[qid] = qdata
            continue
        
        fixed_data[qid] = new_qdata
        stats['success'] += 1
        
        if result.get('pool_size', 0) != args.target_pool_size:
            stats['pool_size_mismatch'] += 1
        
        if result.get('was_all_zero', False):
            if result.get('is_still_all_zero', True):
                stats['still_all_zero'] += 1
            else:
                stats['all_zero_fixed'] += 1
        
        if result.get('icds_injected', 0) > 0:
            stats['queries_with_injection'] += 1
            stats['total_icds_injected'] += result['icds_injected']
    
    # 统计优化后的状态
    pool_sizes = []
    trajectory_counts = []
    positive_counts = []
    all_zero_after = 0
    
    for qid, qdata in fixed_data.items():
        if qid == '_meta':
            continue
        pool = qdata.get('candidate_pool_ids', [])
        trajectories = qdata.get('pointer_candidates', [])
        positives = sum(1 for t in trajectories if t.get('vqa_correct', 0) == 1)
        
        pool_sizes.append(len(pool))
        trajectory_counts.append(len(trajectories))
        positive_counts.append(positives)
        
        if positives == 0:
            all_zero_after += 1
    
    print(f"\n" + "=" * 70)
    print("修复结果统计")
    print("=" * 70)
    print(f"成功修复: {stats['success']}")
    print(f"修复失败: {stats['failed']}")
    print(f"需要注入ICD的query: {stats['queries_with_injection']}")
    print(f"All-Zero修复成功: {stats['all_zero_fixed']}")
    print(f"仍为All-Zero: {stats['still_all_zero']}")
    print(f"注入的ICD总数: {stats['total_icds_injected']}")
    
    print(f"\n候选池大小验证:")
    print(f"  平均: {np.mean(pool_sizes):.1f}")
    print(f"  最小: {min(pool_sizes) if pool_sizes else 0}")
    print(f"  最大: {max(pool_sizes) if pool_sizes else 0}")
    print(f"  等于{args.target_pool_size}: {sum(1 for s in pool_sizes if s == args.target_pool_size)}/{len(pool_sizes)}")
    
    print(f"\n轨迹数量统计:")
    print(f"  平均: {np.mean(trajectory_counts):.1f}")
    print(f"  最小: {min(trajectory_counts) if trajectory_counts else 0}")
    print(f"  最大: {max(trajectory_counts) if trajectory_counts else 0}")
    
    print(f"\n正样本数量统计:")
    print(f"  平均: {np.mean(positive_counts):.1f}")
    print(f"  最小: {min(positive_counts) if positive_counts else 0}")
    print(f"  最大: {max(positive_counts) if positive_counts else 0}")
    
    print(f"\nAll-Zero query变化:")
    print(f"  优化前（基于in_pool轨迹）: {all_zero_before} ({all_zero_before/len(queries)*100:.1f}%)")
    print(f"  优化后: {all_zero_after} ({all_zero_after/len(queries)*100:.1f}%)")
    print(f"  减少: {all_zero_before - all_zero_after}")
    
    # 保存
    print(f"\n保存修复后数据: {args.output_path}")
    save_json(fixed_data, args.output_path)
    
    # 验证断言
    print(f"\n验证断言...")
    assertion_passed = True
    
    for qid, qdata in fixed_data.items():
        if qid == '_meta':
            continue
        
        pool = qdata.get('candidate_pool_ids', [])
        trajectories = qdata.get('pointer_candidates', [])
        
        # 断言1：候选池大小
        if len(pool) != args.target_pool_size and len(pool) > 0:
            print(f"  ⚠️ Query {qid}: 候选池大小 {len(pool)} != {args.target_pool_size}")
            assertion_passed = False
        
        # 断言2：轨迹索引在有效范围内
        for traj in trajectories:
            pointer = traj.get('pointer', [])
            for pos in pointer:
                if pos < 0 or pos >= len(pool):
                    print(f"  ⚠️ Query {qid}: 轨迹索引 {pos} 超出范围 [0, {len(pool)-1}]")
                    assertion_passed = False
    
    if assertion_passed:
        print("  ✓ 所有断言通过")
    else:
        print("  ⚠️ 部分断言失败，请检查数据")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
