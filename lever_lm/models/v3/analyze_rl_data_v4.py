"""
RL 数据体检脚本 v4

按照 2025-12-13需求.md §4.1 的要求：
- 输入：RL 数据 JSON（query -> pointer_candidates）
- 输出：统计信息（打印 + 保存 CSV）

诊断指标：
- 全局统计：query 数量、候选总数、正样本比例等
- 关键诊断：zero-positive query 比例、flat reward query 比例等
- 采样来源分析：beam/sample/random 的正样本率

作者: Lever-Plus Team
日期: 2025-12-13
参考: 2025-12-13需求.md
"""

import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List
from scipy.stats import spearmanr


def analyze_rl_data(rl_data: Dict, output_csv: str = None) -> Dict:
    """
    分析 RL 数据
    
    Args:
        rl_data: RL 数据，格式为 {query_id: {"pointer_candidates": [...]}}
        output_csv: 输出 CSV 文件路径（可选）
    
    Returns:
        stats: 统计信息字典
    """
    stats = {}
    
    # 全局统计
    query_ids = list(rl_data.keys())
    num_queries = len(query_ids)
    
    all_candidates = []
    all_pointers = []
    all_vqa_correct = []
    all_vqa_acc_score = []
    all_vqa_gt_prob = []
    all_gen_methods = []
    
    # 每个 query 的统计
    query_stats = []
    
    for query_id_str, query_data in rl_data.items():
        query_id = int(query_id_str)
        pointer_candidates = query_data.get("pointer_candidates", [])
        
        if len(pointer_candidates) == 0:
            continue
        
        # 统计当前 query 的候选
        unique_pointers = set()
        pos_count = 0
        vqa_acc_scores = []
        vqa_gt_probs = []
        gen_methods = []
        
        for c in pointer_candidates:
            # 跳过评估失败的候选
            if c.get("eval_failed", False):
                continue
            
            pointer = tuple(sorted(c["pointer"]))
            unique_pointers.add(pointer)
            
            vqa_correct = c.get("vqa_correct", 0)
            vqa_acc_score = c.get("vqa_acc_score", 0.0)
            vqa_gt_prob = c.get("vqa_gt_prob", 0.0)
            gen_method = c.get("gen_method", "unknown")
            
            all_candidates.append({
                "query_id": query_id,
                "pointer": pointer,
                "vqa_correct": vqa_correct,
                "vqa_acc_score": vqa_acc_score,
                "vqa_gt_prob": vqa_gt_prob,
                "gen_method": gen_method
            })
            
            all_pointers.append(pointer)
            all_vqa_correct.append(vqa_correct)
            all_vqa_acc_score.append(vqa_acc_score)
            all_vqa_gt_prob.append(vqa_gt_prob)
            all_gen_methods.append(gen_method)
            
            if vqa_correct == 1:
                pos_count += 1
            
            vqa_acc_scores.append(vqa_acc_score)
            vqa_gt_probs.append(vqa_gt_prob)
            gen_methods.append(gen_method)
        
        # 计算当前 query 的 reward 方差（用于判断 flat reward）
        if len(vqa_gt_probs) > 0:
            reward_var = np.var(vqa_gt_probs)
        else:
            reward_var = 0.0
        
        query_stats.append({
            "query_id": query_id,
            "num_candidates": len(pointer_candidates),
            "unique_pointers": len(unique_pointers),
            "pos_count": pos_count,
            "pos_ratio": pos_count / len(pointer_candidates) if len(pointer_candidates) > 0 else 0.0,
            "reward_var": reward_var,
            "mean_vqa_acc_score": np.mean(vqa_acc_scores) if vqa_acc_scores else 0.0,
            "mean_vqa_gt_prob": np.mean(vqa_gt_probs) if vqa_gt_probs else 0.0,
            "max_vqa_acc_score": np.max(vqa_acc_scores) if vqa_acc_scores else 0.0,
            "max_vqa_gt_prob": np.max(vqa_gt_probs) if vqa_gt_probs else 0.0,
        })
    
    # 全局统计
    stats["num_queries"] = num_queries
    stats["total_candidates"] = len(all_candidates)
    stats["unique_pointers"] = len(set(all_pointers))
    stats["candidate_dedup_ratio"] = 1.0 - (len(set(all_pointers)) / len(all_pointers)) if len(all_pointers) > 0 else 0.0
    
    # 正样本统计
    stats["total_positive"] = sum(all_vqa_correct)
    stats["positive_ratio"] = stats["total_positive"] / len(all_candidates) if len(all_candidates) > 0 else 0.0
    
    # 每个 query 的正样本数分布
    pos_counts = [qs["pos_count"] for qs in query_stats]
    if pos_counts:
        stats["pos_count_min"] = int(np.min(pos_counts))
        stats["pos_count_p25"] = int(np.percentile(pos_counts, 25))
        stats["pos_count_median"] = int(np.median(pos_counts))
        stats["pos_count_p75"] = int(np.percentile(pos_counts, 75))
        stats["pos_count_max"] = int(np.max(pos_counts))
    else:
        stats["pos_count_min"] = 0
        stats["pos_count_p25"] = 0
        stats["pos_count_median"] = 0
        stats["pos_count_p75"] = 0
        stats["pos_count_max"] = 0
    
    # 关键诊断：zero-positive query 比例
    zero_positive_count = sum(1 for qs in query_stats if qs["pos_count"] == 0)
    stats["pct_zero_positive_query"] = zero_positive_count / len(query_stats) * 100 if len(query_stats) > 0 else 0.0
    stats["num_zero_positive_query"] = zero_positive_count
    
    # 关键诊断：flat reward query 比例（reward 方差 < eps）
    eps = 1e-6
    flat_reward_count = sum(1 for qs in query_stats if qs["reward_var"] < eps)
    stats["pct_flat_reward_query"] = flat_reward_count / len(query_stats) * 100 if len(query_stats) > 0 else 0.0
    stats["num_flat_reward_query"] = flat_reward_count
    
    # 采样来源分析：每个生成方法的正样本率
    gen_method_stats = {}
    for method in set(all_gen_methods):
        method_candidates = [c for c in all_candidates if c["gen_method"] == method]
        method_pos = sum(1 for c in method_candidates if c["vqa_correct"] == 1)
        gen_method_stats[method] = {
            "count": len(method_candidates),
            "positive": method_pos,
            "positive_ratio": method_pos / len(method_candidates) if len(method_candidates) > 0 else 0.0,
            "mean_vqa_acc_score": np.mean([c["vqa_acc_score"] for c in method_candidates]) if method_candidates else 0.0,
            "mean_vqa_gt_prob": np.mean([c["vqa_gt_prob"] for c in method_candidates]) if method_candidates else 0.0,
        }
    stats["gen_method_stats"] = gen_method_stats
    
    # vqa_gt_prob 与 vqa_acc_score 的相关性
    if len(all_vqa_acc_score) > 1 and len(all_vqa_gt_prob) > 1:
        correlation, p_value = spearmanr(all_vqa_acc_score, all_vqa_gt_prob)
        stats["vqa_gt_prob_vs_acc_score_correlation"] = correlation
        stats["vqa_gt_prob_vs_acc_score_p_value"] = p_value
    else:
        stats["vqa_gt_prob_vs_acc_score_correlation"] = 0.0
        stats["vqa_gt_prob_vs_acc_score_p_value"] = 1.0
    
    # Oracle vs Model Pick 分析
    # Oracle: 每个 query 中最好的候选（最大 vqa_acc_score 或 vqa_gt_prob）
    # Model Pick: 模型会选的候选（通常是 beam rank=0 或 logprob_score 最高的）
    oracle_acc_scores = []
    oracle_gt_probs = []
    model_pick_acc_scores = []
    model_pick_gt_probs = []
    
    for query_id_str, query_data in rl_data.items():
        pointer_candidates = query_data.get("pointer_candidates", [])
        if len(pointer_candidates) == 0:
            continue
        
        # 过滤掉评估失败的候选
        valid_candidates = [c for c in pointer_candidates if not c.get("eval_failed", False)]
        if len(valid_candidates) == 0:
            continue
        
        # Oracle: 最大 vqa_acc_score 或 vqa_gt_prob
        oracle_acc = max([c.get("vqa_acc_score", 0.0) for c in valid_candidates])
        oracle_gt_prob = max([c.get("vqa_gt_prob", 0.0) for c in valid_candidates])
        oracle_acc_scores.append(oracle_acc)
        oracle_gt_probs.append(oracle_gt_prob)
        
        # Model Pick: beam rank=0 或 logprob_score 最高的
        model_pick = None
        for c in valid_candidates:
            if c.get("gen_method") == "beam" and c.get("beam_rank") == 0:
                model_pick = c
                break
        
        if model_pick is None:
            # 如果没有 beam rank=0，选择 logprob_score 最高的
            model_pick = max(valid_candidates, key=lambda x: x.get("logprob_score", -float('inf')))
        
        model_pick_acc_scores.append(model_pick.get("vqa_acc_score", 0.0))
        model_pick_gt_probs.append(model_pick.get("vqa_gt_prob", 0.0))
    
    if oracle_acc_scores and model_pick_acc_scores:
        stats["oracle_mean_acc_score"] = np.mean(oracle_acc_scores)
        stats["model_pick_mean_acc_score"] = np.mean(model_pick_acc_scores)
        stats["oracle_vs_model_pick_acc_gap"] = stats["oracle_mean_acc_score"] - stats["model_pick_mean_acc_score"]
        
        stats["oracle_mean_gt_prob"] = np.mean(oracle_gt_probs)
        stats["model_pick_mean_gt_prob"] = np.mean(model_pick_gt_probs)
        stats["oracle_vs_model_pick_gt_prob_gap"] = stats["oracle_mean_gt_prob"] - stats["model_pick_mean_gt_prob"]
        
        # 有多少 query 的 oracle 明显更高（例如 +0.2）
        significant_gap_count = sum(1 for i in range(len(oracle_acc_scores)) 
                                   if oracle_acc_scores[i] - model_pick_acc_scores[i] > 0.2)
        stats["pct_query_with_significant_gap"] = significant_gap_count / len(oracle_acc_scores) * 100
    else:
        stats["oracle_mean_acc_score"] = 0.0
        stats["model_pick_mean_acc_score"] = 0.0
        stats["oracle_vs_model_pick_acc_gap"] = 0.0
        stats["oracle_mean_gt_prob"] = 0.0
        stats["model_pick_mean_gt_prob"] = 0.0
        stats["oracle_vs_model_pick_gt_prob_gap"] = 0.0
        stats["pct_query_with_significant_gap"] = 0.0
    
    # 保存 CSV
    if output_csv:
        # 保存 query 级别的统计
        query_df = pd.DataFrame(query_stats)
        query_df.to_csv(output_csv.replace(".csv", "_query_level.csv"), index=False)
        
        # 保存候选级别的统计
        candidate_df = pd.DataFrame(all_candidates)
        candidate_df.to_csv(output_csv.replace(".csv", "_candidate_level.csv"), index=False)
        
        print(f"✓ 统计结果已保存到:")
        print(f"  - {output_csv.replace('.csv', '_query_level.csv')}")
        print(f"  - {output_csv.replace('.csv', '_candidate_level.csv')}")
    
    return stats


def print_stats(stats: Dict):
    """
    打印统计信息
    """
    print("=" * 70)
    print("RL 数据体检报告")
    print("=" * 70)
    
    print("\n【全局统计】")
    print(f"  Query 数量: {stats['num_queries']}")
    print(f"  候选总数: {stats['total_candidates']}")
    print(f"  唯一 pointer 数: {stats['unique_pointers']}")
    print(f"  候选去重率: {stats['candidate_dedup_ratio']:.2%}")
    print(f"  正样本总数: {stats['total_positive']}")
    print(f"  正样本比例: {stats['positive_ratio']:.2%}")
    
    print("\n【每个 Query 的正样本数分布】")
    print(f"  Min: {stats['pos_count_min']}")
    print(f"  P25: {stats['pos_count_p25']}")
    print(f"  Median: {stats['pos_count_median']}")
    print(f"  P75: {stats['pos_count_p75']}")
    print(f"  Max: {stats['pos_count_max']}")
    
    print("\n【关键诊断】")
    print(f"  Zero-positive query 数量: {stats['num_zero_positive_query']}")
    print(f"  Zero-positive query 比例: {stats['pct_zero_positive_query']:.2f}%")
    print(f"  Flat reward query 数量: {stats['num_flat_reward_query']}")
    print(f"  Flat reward query 比例: {stats['pct_flat_reward_query']:.2f}%")
    
    print("\n【采样来源分析】")
    for method, method_stats in stats['gen_method_stats'].items():
        print(f"  {method}:")
        print(f"    候选数: {method_stats['count']}")
        print(f"    正样本数: {method_stats['positive']}")
        print(f"    正样本率: {method_stats['positive_ratio']:.2%}")
        print(f"    平均 vqa_acc_score: {method_stats['mean_vqa_acc_score']:.4f}")
        print(f"    平均 vqa_gt_prob: {method_stats['mean_vqa_gt_prob']:.4f}")
    
    print("\n【vqa_gt_prob 与 vqa_acc_score 相关性】")
    print(f"  Spearman 相关系数: {stats['vqa_gt_prob_vs_acc_score_correlation']:.4f}")
    print(f"  P-value: {stats['vqa_gt_prob_vs_acc_score_p_value']:.4f}")
    
    print("\n【Oracle vs Model Pick】")
    print(f"  Oracle 平均 acc_score: {stats['oracle_mean_acc_score']:.4f}")
    print(f"  Model Pick 平均 acc_score: {stats['model_pick_mean_acc_score']:.4f}")
    print(f"  差距: {stats['oracle_vs_model_pick_acc_gap']:.4f}")
    print(f"  Oracle 平均 gt_prob: {stats['oracle_mean_gt_prob']:.4f}")
    print(f"  Model Pick 平均 gt_prob: {stats['model_pick_mean_gt_prob']:.4f}")
    print(f"  差距: {stats['oracle_vs_model_pick_gt_prob_gap']:.4f}")
    print(f"  有明显差距的 query 比例: {stats['pct_query_with_significant_gap']:.2f}%")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="分析 RL 数据")
    parser.add_argument("--rl_data", type=str, required=True, help="RL 数据 JSON 路径")
    parser.add_argument("--output_csv", type=str, help="输出 CSV 文件路径（可选）")
    
    args = parser.parse_args()
    
    # 加载 RL 数据
    print(f"加载 RL 数据: {args.rl_data}")
    with open(args.rl_data, "r") as f:
        rl_data = json.load(f)
    
    # 分析
    stats = analyze_rl_data(rl_data, output_csv=args.output_csv)
    
    # 打印统计信息
    print_stats(stats)


if __name__ == "__main__":
    main()
