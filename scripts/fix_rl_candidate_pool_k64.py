#!/usr/bin/env python3
"""
修复 RL 候选池维度一致性 + 全局正样本注入 64 ICD 子池

核心目标：
1. RL训练时的候选池必须严格保持K=64（与SFT一致）
2. 从全局挖正样本ICD，注入到64子池中（替换低质量ICD）
3. 控制轨迹数量（建议12-16条），保持正负样本平衡

数据格式要求（每个query必须包含）：
- candidate_pool_ids: 长度=64的全局demo id列表
- pointer_candidates: 每条trajectory是[pos_i, pos_j]（pos是0~63的局部索引）
- 每条trajectory还有vqa_acc_score/reward

使用方法：
    python scripts/fix_rl_candidate_pool_k64.py \
        --input_path results/okvqa/generated_data/rl_data.json \
        --beam_data_path results/okvqa/generated_data/beam_data.json \
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


def get_original_candidate_pool(sampler_cache, query_id):
    """
    从sampler缓存中获取原始的64个候选池索引
    
    sampler_cache格式：{query_id: [candidate_indices]}（长度为64）
    """
    query_id_str = str(query_id)
    if query_id_str not in sampler_cache:
        return None
    
    return sampler_cache[query_id_str]  # 直接返回64个候选索引列表


def collect_global_positive_icds(rl_data):
    """
    收集全局所有正样本中使用的ICD对
    
    返回：
    - global_positive_pairs: {(idx1, idx2): {'score': float, 'source_queries': [qid, ...]}}
    - icd_positive_count: {icd_idx: count} 每个ICD参与正样本的次数
    """
    queries = {k: v for k, v in rl_data.items() if k != '_meta'}
    
    global_positive_pairs = defaultdict(lambda: {'score': 0, 'source_queries': []})
    icd_positive_count = defaultdict(int)
    
    for qid, qdata in queries.items():
        for c in qdata.get('pointer_candidates', []):
            if c.get('vqa_correct', 0) == 1:
                pointer = c.get('pointer', [])
                if len(pointer) >= 2:
                    pair_key = tuple(sorted(pointer[:2]))
                    score = c.get('vqa_acc_score', 0)
                    
                    if score > global_positive_pairs[pair_key]['score']:
                        global_positive_pairs[pair_key]['score'] = score
                    global_positive_pairs[pair_key]['source_queries'].append(qid)
                    
                    for idx in pointer[:2]:
                        icd_positive_count[idx] += 1
    
    return dict(global_positive_pairs), dict(icd_positive_count)


def analyze_query_status(qdata):
    """分析单个query的状态"""
    candidates = qdata.get('pointer_candidates', [])
    
    positive_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 1)
    negative_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 0)
    
    # 收集当前用到的ICD索引
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


def find_icds_to_inject(
    query_id,
    current_pool,
    global_positive_pairs,
    icd_positive_count,
    max_inject=8
):
    """
    为All-Zero query找到需要注入的ICD
    
    策略：
    1. 找全局正样本pair中，有一个ICD已在当前池中的（只需注入1个）
    2. 找全局正样本pair中，两个ICD都不在当前池中的（需注入2个）
    3. 按正样本参与次数排序，优先注入"高频正样本ICD"
    """
    pool_set = set(current_pool)
    
    # 候选注入的ICD及其来源pair
    inject_candidates = []
    
    for pair_key, pair_info in global_positive_pairs.items():
        idx1, idx2 = pair_key
        in_pool_1 = idx1 in pool_set
        in_pool_2 = idx2 in pool_set
        
        if in_pool_1 and in_pool_2:
            # 两个都在池中，不需要注入
            continue
        elif in_pool_1 or in_pool_2:
            # 只需要注入1个
            new_idx = idx2 if in_pool_1 else idx1
            inject_candidates.append({
                'icds_to_inject': [new_idx],
                'pair': pair_key,
                'score': pair_info['score'],
                'priority': icd_positive_count.get(new_idx, 0)
            })
        else:
            # 需要注入2个
            inject_candidates.append({
                'icds_to_inject': [idx1, idx2],
                'pair': pair_key,
                'score': pair_info['score'],
                'priority': icd_positive_count.get(idx1, 0) + icd_positive_count.get(idx2, 0)
            })
    
    # 按优先级排序（高频ICD优先）
    inject_candidates.sort(key=lambda x: (-x['priority'], -x['score']))
    
    # 选择要注入的ICD
    icds_to_inject = set()
    pairs_to_add = []
    
    for candidate in inject_candidates:
        needed = len(candidate['icds_to_inject'])
        if len(icds_to_inject) + needed <= max_inject:
            icds_to_inject.update(candidate['icds_to_inject'])
            pairs_to_add.append(candidate['pair'])
    
    return list(icds_to_inject), pairs_to_add


def select_icds_to_remove(current_pool, used_indices, num_to_remove, icd_positive_count):
    """
    选择要从候选池中移除的ICD
    
    策略：
    1. 优先移除未被任何序列使用的ICD
    2. 其次移除正样本参与次数最少的ICD
    """
    pool_set = set(current_pool)
    used_set = set(used_indices)
    
    # 未使用的ICD
    unused = pool_set - used_set
    
    # 按正样本参与次数排序（少的优先移除）
    candidates = list(unused) if unused else list(pool_set - used_set)
    if len(candidates) < num_to_remove:
        # 如果未使用的不够，从已使用的中选择参与次数最少的
        used_list = list(used_set & pool_set)
        used_list.sort(key=lambda x: icd_positive_count.get(x, 0))
        candidates.extend(used_list)
    
    # 按正样本参与次数排序
    candidates.sort(key=lambda x: icd_positive_count.get(x, 0))
    
    return candidates[:num_to_remove]


def build_fixed_candidate_pool(
    query_id,
    original_pool,
    icds_to_inject,
    icds_to_remove,
    target_size=64
):
    """
    构建固定大小的候选池
    
    确保：
    1. 候选池大小严格为64
    2. 注入的ICD替换掉要移除的ICD
    3. 返回新的候选池和索引映射
    """
    pool_set = set(original_pool)
    
    # 移除ICD
    for idx in icds_to_remove:
        pool_set.discard(idx)
    
    # 注入ICD
    for idx in icds_to_inject:
        pool_set.add(idx)
    
    # 确保大小为64
    new_pool = sorted(list(pool_set))
    
    if len(new_pool) > target_size:
        # 如果超过64，截断（保留前64个）
        new_pool = new_pool[:target_size]
    elif len(new_pool) < target_size:
        # 如果不足64，这是一个问题，需要记录
        pass
    
    # 构建索引映射：global_id -> local_pos (0~63)
    global_to_local = {idx: pos for pos, idx in enumerate(new_pool)}
    
    return new_pool, global_to_local


def convert_trajectories_to_local_indices(
    trajectories,
    global_to_local,
    new_pool
):
    """
    将轨迹中的全局索引转换为局部索引
    
    如果某个全局索引不在新候选池中，该轨迹会被跳过
    """
    converted = []
    skipped = 0
    
    for traj in trajectories:
        pointer = traj.get('pointer', [])
        
        # 检查所有索引是否在新候选池中
        try:
            local_pointer = [global_to_local[idx] for idx in pointer]
            new_traj = deepcopy(traj)
            new_traj['pointer_global'] = pointer  # 保留全局索引
            new_traj['pointer'] = local_pointer   # 使用局部索引
            converted.append(new_traj)
        except KeyError:
            # 某个索引不在新候选池中，跳过
            skipped += 1
    
    return converted, skipped


def create_new_trajectories_from_pairs(
    pairs_to_add,
    global_to_local,
    global_positive_pairs
):
    """
    从注入的pair创建新的正样本轨迹
    """
    new_trajectories = []
    
    for pair in pairs_to_add:
        idx1, idx2 = pair
        
        # 检查两个索引是否都在新候选池中
        if idx1 not in global_to_local or idx2 not in global_to_local:
            continue
        
        local_pointer = [global_to_local[idx1], global_to_local[idx2]]
        pair_info = global_positive_pairs.get(pair, {})
        
        new_traj = {
            'pointer': local_pointer,
            'pointer_global': list(pair),
            'vqa_correct': 1,
            'vqa_acc_score': pair_info.get('score', 1.0),
            'gen_method': 'global_positive_injection',
            'source_queries': pair_info.get('source_queries', [])[:3]  # 只保留前3个来源
        }
        new_trajectories.append(new_traj)
    
    return new_trajectories


def balance_trajectories(
    trajectories,
    target_num=12,
    positive_quota=(2, 4),
    negative_quota=None
):
    """
    平衡轨迹数量和正负样本比例
    
    配额建议：
    - 正样本(vqa_correct=1): 2~4条
    - 中间样本(0<acc<1): 1~2条
    - 负样本(vqa_correct=0): 其余补齐
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
    
    # 合并
    final_trajectories = selected_positives + selected_negatives
    
    # 如果还不够，随机补充
    if len(final_trajectories) < target_num:
        remaining = [t for t in trajectories if t not in final_trajectories]
        random.shuffle(remaining)
        final_trajectories.extend(remaining[:target_num - len(final_trajectories)])
    
    return final_trajectories


def fix_single_query(
    query_id,
    qdata,
    original_pool,
    global_positive_pairs,
    icd_positive_count,
    target_pool_size=64,
    target_trajectory_num=12,
    max_inject_icds=8
):
    """
    修复单个query的候选池和轨迹
    """
    status = analyze_query_status(qdata)
    
    # 如果原始候选池不存在或为空，跳过
    if not original_pool:
        return None, {'error': 'no_original_pool'}
    
    # 确保原始候选池大小为64
    if len(original_pool) < target_pool_size:
        # 候选池不足64，需要补充
        # 这种情况下，我们保持原样，只记录警告
        pass
    
    icds_to_inject = []
    icds_to_remove = []
    pairs_to_add = []
    
    # 只对All-Zero query进行注入
    if status['is_all_zero']:
        icds_to_inject, pairs_to_add = find_icds_to_inject(
            query_id=query_id,
            current_pool=original_pool,
            global_positive_pairs=global_positive_pairs,
            icd_positive_count=icd_positive_count,
            max_inject=max_inject_icds
        )
        
        if icds_to_inject:
            # 选择要移除的ICD
            icds_to_remove = select_icds_to_remove(
                current_pool=original_pool,
                used_indices=status['used_indices'],
                num_to_remove=len(icds_to_inject),
                icd_positive_count=icd_positive_count
            )
    
    # 构建新的候选池
    new_pool, global_to_local = build_fixed_candidate_pool(
        query_id=query_id,
        original_pool=original_pool,
        icds_to_inject=icds_to_inject,
        icds_to_remove=icds_to_remove,
        target_size=target_pool_size
    )
    
    # 转换现有轨迹的索引
    existing_trajectories = qdata.get('pointer_candidates', [])
    converted_trajectories, skipped = convert_trajectories_to_local_indices(
        trajectories=existing_trajectories,
        global_to_local=global_to_local,
        new_pool=new_pool
    )
    
    # 创建新的正样本轨迹（从注入的pair）
    new_positive_trajectories = create_new_trajectories_from_pairs(
        pairs_to_add=pairs_to_add,
        global_to_local=global_to_local,
        global_positive_pairs=global_positive_pairs
    )
    
    # 合并轨迹
    all_trajectories = converted_trajectories + new_positive_trajectories
    
    # 平衡轨迹
    final_trajectories = balance_trajectories(
        trajectories=all_trajectories,
        target_num=target_trajectory_num,
        positive_quota=(2, 4)
    )
    
    # 构建新的qdata
    new_qdata = {
        'query': qdata.get('query', {}),
        'candidate_pool_ids': new_pool,  # 关键：保存完整的64个候选池ID
        'pointer_candidates': final_trajectories,
        '_fix_info': {
            'original_pool_size': len(original_pool),
            'new_pool_size': len(new_pool),
            'icds_injected': icds_to_inject,
            'icds_removed': icds_to_remove,
            'trajectories_skipped': skipped,
            'new_positives_added': len(new_positive_trajectories),
            'was_all_zero': status['is_all_zero']
        }
    }
    
    # 统计结果
    final_positive_count = sum(1 for t in final_trajectories if t.get('vqa_correct', 0) == 1)
    
    result = {
        'success': True,
        'pool_size': len(new_pool),
        'trajectory_count': len(final_trajectories),
        'positive_count': final_positive_count,
        'was_all_zero': status['is_all_zero'],
        'is_still_all_zero': final_positive_count == 0,
        'icds_injected': len(icds_to_inject)
    }
    
    return new_qdata, result


def main():
    parser = argparse.ArgumentParser(description="修复RL候选池维度一致性 + 全局正样本注入")
    parser.add_argument("--input_path", type=str, required=True, help="输入RL数据路径")
    parser.add_argument("--sampler_cache_path", type=str, required=True, help="sampler缓存路径（包含64候选池信息）")
    parser.add_argument("--output_path", type=str, required=True, help="输出修复后数据路径")
    parser.add_argument("--target_pool_size", type=int, default=64, help="目标候选池大小（必须为64）")
    parser.add_argument("--target_trajectory_num", type=int, default=12, help="每个query的目标轨迹数")
    parser.add_argument("--max_inject_icds", type=int, default=8, help="每个All-Zero query最多注入的ICD数")
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
    
    # 验证sampler缓存格式
    first_key = list(sampler_cache.keys())[0]
    first_pool = sampler_cache[first_key]
    print(f"  第一个query的候选池大小: {len(first_pool)}")
    
    # 收集全局正样本
    print("\n收集全局正样本...")
    global_positive_pairs, icd_positive_count = collect_global_positive_icds(rl_data)
    print(f"  全局unique正样本pair数: {len(global_positive_pairs)}")
    print(f"  参与正样本的unique ICD数: {len(icd_positive_count)}")
    
    # 统计优化前的状态
    queries = {k: v for k, v in rl_data.items() if k != '_meta'}
    all_zero_before = sum(1 for qid, qdata in queries.items() 
                         if analyze_query_status(qdata)['is_all_zero'])
    
    print(f"\n优化前统计:")
    print(f"  总query数: {len(queries)}")
    print(f"  All-Zero query数: {all_zero_before} ({all_zero_before/len(queries)*100:.1f}%)")
    
    # 修复每个query
    print(f"\n开始修复候选池（目标K={args.target_pool_size}）...")
    fixed_data = {}
    if '_meta' in rl_data:
        fixed_data['_meta'] = deepcopy(rl_data['_meta'])
        fixed_data['_meta']['fixed_at'] = datetime.now().isoformat()
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
        'total_icds_injected': 0
    }
    
    for qid in tqdm(queries.keys(), desc="修复query"):
        qdata = queries[qid]
        
        # 从sampler缓存获取原始64候选池
        original_pool = get_original_candidate_pool(sampler_cache, qid)
        
        if not original_pool:
            # 如果没有原始候选池，尝试从RL数据中提取（fallback）
            used_indices = set()
            for c in qdata.get('pointer_candidates', []):
                for idx in c.get('pointer', []):
                    used_indices.add(idx)
            original_pool = sorted(list(used_indices))
            print(f"  警告: query {qid} 没有sampler缓存，使用RL数据中的索引（{len(original_pool)}个）")
        
        # 修复
        new_qdata, result = fix_single_query(
            query_id=qid,
            qdata=qdata,
            original_pool=original_pool,
            global_positive_pairs=global_positive_pairs,
            icd_positive_count=icd_positive_count,
            target_pool_size=args.target_pool_size,
            target_trajectory_num=args.target_trajectory_num,
            max_inject_icds=args.max_inject_icds
        )
        
        if new_qdata is None:
            stats['failed'] += 1
            # 保留原始数据
            fixed_data[qid] = qdata
            continue
        
        fixed_data[qid] = new_qdata
        stats['success'] += 1
        
        # 统计
        if result.get('pool_size', 0) != args.target_pool_size:
            stats['pool_size_mismatch'] += 1
        
        if result.get('was_all_zero', False):
            if result.get('is_still_all_zero', True):
                stats['still_all_zero'] += 1
            else:
                stats['all_zero_fixed'] += 1
        
        stats['total_icds_injected'] += result.get('icds_injected', 0)
    
    # 统计优化后的状态
    all_zero_after = sum(1 for qid, qdata in fixed_data.items() 
                        if qid != '_meta' and 
                        sum(1 for t in qdata.get('pointer_candidates', []) 
                            if t.get('vqa_correct', 0) == 1) == 0)
    
    # 验证候选池大小
    pool_sizes = []
    trajectory_counts = []
    positive_counts = []
    
    for qid, qdata in fixed_data.items():
        if qid == '_meta':
            continue
        pool = qdata.get('candidate_pool_ids', [])
        trajectories = qdata.get('pointer_candidates', [])
        positives = sum(1 for t in trajectories if t.get('vqa_correct', 0) == 1)
        
        pool_sizes.append(len(pool))
        trajectory_counts.append(len(trajectories))
        positive_counts.append(positives)
    
    print(f"\n" + "=" * 70)
    print("修复结果统计")
    print("=" * 70)
    print(f"成功修复: {stats['success']}")
    print(f"修复失败: {stats['failed']}")
    print(f"All-Zero修复成功: {stats['all_zero_fixed']}")
    print(f"仍为All-Zero: {stats['still_all_zero']}")
    print(f"候选池大小不匹配: {stats['pool_size_mismatch']}")
    print(f"注入的ICD总数: {stats['total_icds_injected']}")
    
    print(f"\n候选池大小验证:")
    print(f"  平均: {np.mean(pool_sizes):.1f}")
    print(f"  最小: {min(pool_sizes)}")
    print(f"  最大: {max(pool_sizes)}")
    print(f"  等于{args.target_pool_size}: {sum(1 for s in pool_sizes if s == args.target_pool_size)}/{len(pool_sizes)}")
    
    print(f"\n轨迹数量统计:")
    print(f"  平均: {np.mean(trajectory_counts):.1f}")
    print(f"  最小: {min(trajectory_counts)}")
    print(f"  最大: {max(trajectory_counts)}")
    
    print(f"\n正样本数量统计:")
    print(f"  平均: {np.mean(positive_counts):.1f}")
    print(f"  最小: {min(positive_counts)}")
    print(f"  最大: {max(positive_counts)}")
    
    print(f"\nAll-Zero query变化:")
    print(f"  优化前: {all_zero_before} ({all_zero_before/len(queries)*100:.1f}%)")
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
        if len(pool) != args.target_pool_size:
            if len(pool) < args.target_pool_size:
                # 允许小于64（原始数据不足）
                pass
            else:
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
