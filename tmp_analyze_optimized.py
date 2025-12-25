import json
from collections import defaultdict
import numpy as np

# 读取优化后的数据
with open('results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_pool64_optimized.json', 'r') as f:
    data = json.load(f)

queries = {k: v for k, v in data.items() if k != '_meta'}

print("=" * 70)
print("优化后RL数据详细统计")
print("=" * 70)

# 统计
candidate_counts = []
positive_counts = []
negative_counts = []
pool_sizes = []

for qid, qdata in queries.items():
    candidates = qdata.get('pointer_candidates', [])
    candidate_counts.append(len(candidates))
    
    indices = set()
    pos_count = 0
    neg_count = 0
    
    for c in candidates:
        for idx in c.get('pointer', []):
            indices.add(idx)
        if c.get('vqa_correct', 0) == 1:
            pos_count += 1
        else:
            neg_count += 1
    
    positive_counts.append(pos_count)
    negative_counts.append(neg_count)
    pool_sizes.append(len(indices))

print(f"\n总query数: {len(queries)}")

print(f"\n候选序列数量统计:")
print(f"  平均: {np.mean(candidate_counts):.1f}")
print(f"  最小: {min(candidate_counts)}")
print(f"  最大: {max(candidate_counts)}")

print(f"\n候选池大小统计:")
print(f"  平均: {np.mean(pool_sizes):.1f}")
print(f"  最小: {min(pool_sizes)}")
print(f"  最大: {max(pool_sizes)}")
print(f"  全部=64: {sum(1 for s in pool_sizes if s == 64)}")

print(f"\n正样本数量统计:")
print(f"  平均: {np.mean(positive_counts):.1f}")
print(f"  最小: {min(positive_counts)}")
print(f"  最大: {max(positive_counts)}")

print(f"\n负样本数量统计:")
print(f"  平均: {np.mean(negative_counts):.1f}")
print(f"  最小: {min(negative_counts)}")
print(f"  最大: {max(negative_counts)}")

# 零正样本
zero_pos = sum(1 for p in positive_counts if p == 0)
print(f"\n零正样本query: {zero_pos} ({zero_pos/len(queries)*100:.1f}%)")

# 生成方法分布
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

# 正负样本比例分布
print("\n" + "=" * 70)
print("正负样本比例分布")
print("=" * 70)

ratios = [p / (p + n) if (p + n) > 0 else 0 for p, n in zip(positive_counts, negative_counts)]
print(f"\n正样本比例:")
print(f"  平均: {np.mean(ratios)*100:.1f}%")
print(f"  最小: {min(ratios)*100:.1f}%")
print(f"  最大: {max(ratios)*100:.1f}%")

# 分布
bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
for i in range(len(bins)-1):
    count = sum(1 for r in ratios if bins[i] <= r < bins[i+1])
    print(f"  {bins[i]*100:.0f}%-{bins[i+1]*100:.0f}%: {count}个query")
count = sum(1 for r in ratios if r >= 1.0)
print(f"  100%: {count}个query")

# 对比优化前后
print("\n" + "=" * 70)
print("优化前后对比")
print("=" * 70)

with open('results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_final_merged.json', 'r') as f:
    old_data = json.load(f)

old_queries = {k: v for k, v in old_data.items() if k != '_meta'}
old_candidate_counts = []
old_positive_counts = []
old_pool_sizes = []

for qid, qdata in old_queries.items():
    candidates = qdata.get('pointer_candidates', [])
    old_candidate_counts.append(len(candidates))
    
    indices = set()
    pos_count = 0
    for c in candidates:
        for idx in c.get('pointer', []):
            indices.add(idx)
        if c.get('vqa_correct', 0) == 1:
            pos_count += 1
    
    old_positive_counts.append(pos_count)
    old_pool_sizes.append(len(indices))

print(f"\n                    优化前      优化后      变化")
print(f"候选序列数(平均):   {np.mean(old_candidate_counts):.1f}        {np.mean(candidate_counts):.1f}       +{np.mean(candidate_counts)-np.mean(old_candidate_counts):.1f}")
print(f"候选池大小(平均):   {np.mean(old_pool_sizes):.1f}        {np.mean(pool_sizes):.1f}       +{np.mean(pool_sizes)-np.mean(old_pool_sizes):.1f}")
print(f"正样本数(平均):     {np.mean(old_positive_counts):.1f}         {np.mean(positive_counts):.1f}       +{np.mean(positive_counts)-np.mean(old_positive_counts):.1f}")
print(f"零正样本query:      {sum(1 for p in old_positive_counts if p == 0)}          {zero_pos}          -{sum(1 for p in old_positive_counts if p == 0)-zero_pos}")
