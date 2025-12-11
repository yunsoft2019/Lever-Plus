"""
RL 数据统计工具

统计 RL 数据中的正样本比例（candidate 级和 query 级）

作者: Lever-Plus Team
日期: 2025-12-12
"""

import json
import argparse
from typing import Dict, Tuple


def compute_rl_positive_stats(rl_json_path: str) -> Dict:
    """
    统计 RL 数据中的正样本比例
    
    Args:
        rl_json_path: RL 数据 JSON 文件路径
        
    Returns:
        统计结果字典
    """
    with open(rl_json_path, "r") as f:
        rl_data = json.load(f)
    
    total_candidates = 0
    total_pos_candidates = 0
    total_soft_score = 0.0
    
    total_queries = 0
    queries_with_pos = 0
    queries_with_any_score = 0  # 至少有一个 vqa_acc_score > 0 的 query
    
    # 按 vqa_correct 分布统计
    correct_distribution = {0: 0, 1: 0}
    # 按 vqa_acc_score 分布统计
    acc_score_bins = {"0": 0, "(0,0.5)": 0, "[0.5,1)": 0, "1": 0}
    
    for qid_str, qinfo in rl_data.items():
        pcs = qinfo.get("pointer_candidates", [])
        if not pcs:
            continue
        
        total_queries += 1
        has_pos = False
        has_any_score = False
        max_acc_score = 0.0
        
        for c in pcs:
            total_candidates += 1
            correct = c.get("vqa_correct", 0)
            acc_score = c.get("vqa_acc_score", 0.0)
            
            # 统计 vqa_correct 分布
            if correct in correct_distribution:
                correct_distribution[correct] += 1
            
            # 统计 vqa_acc_score 分布
            if acc_score == 0:
                acc_score_bins["0"] += 1
            elif acc_score < 0.5:
                acc_score_bins["(0,0.5)"] += 1
            elif acc_score < 1.0:
                acc_score_bins["[0.5,1)"] += 1
            else:
                acc_score_bins["1"] += 1
            
            total_soft_score += acc_score
            max_acc_score = max(max_acc_score, acc_score)
            
            if correct == 1:
                total_pos_candidates += 1
                has_pos = True
            
            if acc_score > 0:
                has_any_score = True
        
        if has_pos:
            queries_with_pos += 1
        if has_any_score:
            queries_with_any_score += 1
    
    # 计算比例
    cand_pos_ratio = total_pos_candidates / max(total_candidates, 1)
    query_pos_ratio = queries_with_pos / max(total_queries, 1)
    query_any_score_ratio = queries_with_any_score / max(total_queries, 1)
    avg_soft_score = total_soft_score / max(total_candidates, 1)
    
    stats = {
        "file": rl_json_path,
        "total_candidates": total_candidates,
        "total_pos_candidates": total_pos_candidates,
        "cand_pos_ratio": cand_pos_ratio,
        "total_queries": total_queries,
        "queries_with_pos": queries_with_pos,
        "query_pos_ratio": query_pos_ratio,
        "queries_with_any_score": queries_with_any_score,
        "query_any_score_ratio": query_any_score_ratio,
        "avg_soft_score": avg_soft_score,
        "correct_distribution": correct_distribution,
        "acc_score_bins": acc_score_bins,
    }
    
    return stats


def print_stats(stats: Dict):
    """打印统计结果"""
    print("=" * 70)
    print(f"RL 数据统计: {stats['file']}")
    print("=" * 70)
    print()
    print("【Candidate 级统计】")
    print(f"  - 候选总数: {stats['total_candidates']:,}")
    print(f"  - 正样本候选数 (vqa_correct=1): {stats['total_pos_candidates']:,}")
    print(f"  - 正样本候选比例: {stats['cand_pos_ratio'] * 100:.3f}%")
    print(f"  - 平均 vqa_acc_score: {stats['avg_soft_score']:.4f}")
    print()
    print("【Query 级统计】")
    print(f"  - Query 总数: {stats['total_queries']:,}")
    print(f"  - 至少有一个正样本的 Query 数: {stats['queries_with_pos']:,}")
    print(f"  - Query 级正样本比例: {stats['query_pos_ratio'] * 100:.3f}%")
    print(f"  - 至少有 acc_score>0 的 Query 数: {stats['queries_with_any_score']:,}")
    print(f"  - Query 级有分数比例: {stats['query_any_score_ratio'] * 100:.3f}%")
    print()
    print("【vqa_correct 分布】")
    for k, v in stats['correct_distribution'].items():
        pct = v / max(stats['total_candidates'], 1) * 100
        print(f"  - vqa_correct={k}: {v:,} ({pct:.2f}%)")
    print()
    print("【vqa_acc_score 分布】")
    for k, v in stats['acc_score_bins'].items():
        pct = v / max(stats['total_candidates'], 1) * 100
        print(f"  - {k}: {v:,} ({pct:.2f}%)")
    print("=" * 70)


def compare_stats(stats_list: list):
    """对比多个 RL 数据的统计结果"""
    if len(stats_list) < 2:
        return
    
    print("\n" + "=" * 70)
    print("【多数据对比】")
    print("=" * 70)
    
    headers = ["指标"] + [s["file"].split("/")[-1][:30] for s in stats_list]
    rows = [
        ["候选总数"] + [f"{s['total_candidates']:,}" for s in stats_list],
        ["正样本候选数"] + [f"{s['total_pos_candidates']:,}" for s in stats_list],
        ["正样本候选比例"] + [f"{s['cand_pos_ratio']*100:.3f}%" for s in stats_list],
        ["Query 总数"] + [f"{s['total_queries']:,}" for s in stats_list],
        ["有正样本的 Query"] + [f"{s['queries_with_pos']:,}" for s in stats_list],
        ["Query 级正样本比例"] + [f"{s['query_pos_ratio']*100:.3f}%" for s in stats_list],
    ]
    
    # 打印表格
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    def print_row(row):
        print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
    
    print_row(headers)
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in rows:
        print_row(row)
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="统计 RL 数据中的正样本比例")
    parser.add_argument("--rl_json", type=str, nargs="+", required=True,
                        help="RL 数据 JSON 文件路径（可以传多个进行对比）")
    args = parser.parse_args()
    
    stats_list = []
    for path in args.rl_json:
        stats = compute_rl_positive_stats(path)
        print_stats(stats)
        stats_list.append(stats)
        print()
    
    if len(stats_list) > 1:
        compare_stats(stats_list)


if __name__ == "__main__":
    main()
