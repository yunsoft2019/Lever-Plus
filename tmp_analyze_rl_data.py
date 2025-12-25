import json
from collections import defaultdict
import numpy as np

# 读取合并后的RL数据
with open('results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_final_merged.json', 'r') as f:
    data = json.load(f)

queries = {k: v for k, v in data.items() if k != '_meta'}

print("=" * 70)
print("RL数据详细统计分析")
print("=" * 70)

# 统计每个query的候选数量、正负样本数量
candidate_counts = []
positive_counts = []
negative_counts = []
positive_scores = []
negative_scores = []

# 统计候选索引的全局分布
all_indices = set()
query_indices = {}  # 每个query使用的索引

for qid, qdata in queries.items():
    candidates = qdata.get('pointer_candidates', [])
    candidate_counts.append(len(candidates))
    
    # 收集该query的所有候选索引
    indices = set()
    pos_count = 0
    neg_count = 0
    
    for c in candidates:
        # 获取pointer中的索引
        pointer = c.get('pointer', [])
        for idx in pointer:
            indices.add(idx)
            all_indices.add(idx)
        
        # 统计正负样本
        if c.get('vqa_correct', 0) == 1:
            pos_count += 1
            positive_scores.append(c.get('vqa_acc_score', 0))
        else:
            neg_count += 1
            negative_scores.append(c.get('vqa_rel_score', 0))
    
    positive_counts.append(pos_count)
    negative_counts.append(neg_count)
    query_indices[qid] = indices

print(f"\n总query数: {len(queries)}")
print(f"\n候选序列数量统计:")
print(f"  平均: {np.mean(candidate_counts):.1f}")
print(f"  最小: {min(candidate_counts)}")
print(f"  最大: {max(candidate_counts)}")
print(f"  中位数: {np.median(candidate_counts):.1f}")

print(f"\n正样本数量统计:")
print(f"  平均: {np.mean(positive_counts):.2f}")
print(f"  最小: {min(positive_counts)}")
print(f"  最大: {max(positive_counts)}")
print(f"  中位数: {np.median(positive_counts):.1f}")

print(f"\n负样本数量统计:")
print(f"  平均: {np.mean(negative_counts):.2f}")
print(f"  最小: {min(negative_counts)}")
print(f"  最大: {max(negative_counts)}")
print(f"  中位数: {np.median(negative_counts):.1f}")

# 统计zero_positive的query数量
zero_pos_queries = [qid for qid, pc in zip(queries.keys(), positive_counts) if pc == 0]
print(f"\n零正样本query数: {len(zero_pos_queries)} ({len(zero_pos_queries)/len(queries)*100:.1f}%)")

# 统计每个query使用的unique索引数量
unique_idx_counts = [len(indices) for indices in query_indices.values()]
print(f"\n每个query的unique候选索引数量:")
print(f"  平均: {np.mean(unique_idx_counts):.1f}")
print(f"  最小: {min(unique_idx_counts)}")
print(f"  最大: {max(unique_idx_counts)}")

print(f"\n全局使用的unique索引总数: {len(all_indices)}")

# 检查是否都是从64个候选中选的
print("\n" + "=" * 70)
print("候选池大小分析")
print("=" * 70)

# 分析每个query的候选池是否是64
pool_sizes = []
for qid, indices in query_indices.items():
    pool_sizes.append(len(indices))

print(f"\n候选池大小分布:")
from collections import Counter
size_dist = Counter(pool_sizes)
for size in sorted(size_dist.keys()):
    print(f"  {size}个索引: {size_dist[size]}个query")

# 检查是否有query的候选池超过64
over_64 = sum(1 for s in pool_sizes if s > 64)
under_64 = sum(1 for s in pool_sizes if s < 64)
equal_64 = sum(1 for s in pool_sizes if s == 64)
print(f"\n候选池=64: {equal_64}个query")
print(f"候选池<64: {under_64}个query")
print(f"候选池>64: {over_64}个query")

# 得分统计
print("\n" + "=" * 70)
print("得分统计")
print("=" * 70)

if positive_scores:
    print(f"\n正样本得分 (vqa_acc_score):")
    print(f"  平均: {np.mean(positive_scores):.3f}")
    print(f"  最小: {min(positive_scores):.3f}")
    print(f"  最大: {max(positive_scores):.3f}")

if negative_scores:
    print(f"\n负样本得分 (vqa_rel_score):")
    print(f"  平均: {np.mean(negative_scores):.3f}")
    print(f"  最小: {min(negative_scores):.3f}")
    print(f"  最大: {max(negative_scores):.3f}")

# 分析gen_method分布
print("\n" + "=" * 70)
print("生成方法分布")
print("=" * 70)

gen_method_counts = defaultdict(int)
gen_method_positive = defaultdict(int)
for qid, qdata in queries.items():
    for c in qdata.get('pointer_candidates', []):
        method = c.get('gen_method', 'unknown')
        gen_method_counts[method] += 1
        if c.get('vqa_correct', 0) == 1:
            gen_method_positive[method] += 1

print("\n各生成方法的样本数和正样本率:")
for method in sorted(gen_method_counts.keys()):
    total = gen_method_counts[method]
    pos = gen_method_positive[method]
    rate = pos / total * 100 if total > 0 else 0
    print(f"  {method}: {total}个样本, {pos}个正样本 ({rate:.1f}%)")
