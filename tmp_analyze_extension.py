import json
from collections import defaultdict

# 读取合并后的RL数据
with open('results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_final_merged.json', 'r') as f:
    data = json.load(f)

# 读取原始数据来获取query类型
# 首先读取all1_all06_diverse数据
with open('results/okvqa/generated_data/all1_all06_diverse_20251219_084640/rl_data_RandSampler_v4_strictEval.json', 'r') as f:
    diverse_data = json.load(f)

# 读取all0_aggressive数据
with open('results/okvqa/generated_data/all0_aggressive_20251219_124841/rl_data_RandSampler_v4_strictEval.json', 'r') as f:
    all0_data = json.load(f)

# 获取各类query的ID
diverse_queries = set(k for k in diverse_data.keys() if k != '_meta')
all0_queries = set(k for k in all0_data.keys() if k != '_meta')

# 统计合并后数据中各类query的数量
merged_queries = set(k for k in data.keys() if k != '_meta')

print("=" * 60)
print("RL数据扩展统计")
print("=" * 60)

# 分析原始数据中的query类型
# 需要读取原始beam数据来确定类型
with open('results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval.json', 'r') as f:
    original_data = json.load(f)

# 统计原始数据中各类query
original_queries = set(k for k in original_data.keys() if k != '_meta')

# 分类统计
def classify_query(query_data):
    """根据pointer_candidates中的vqa_correct分类query"""
    candidates = query_data.get('pointer_candidates', [])
    correct_count = sum(1 for c in candidates if c.get('vqa_correct', 0) == 1)
    total = len(candidates)
    
    if total == 0:
        return 'unknown'
    
    if correct_count == 0:
        return 'all0'
    elif correct_count == total:
        return 'all1'
    elif correct_count / total <= 0.6:
        return 'all06'
    else:
        return 'diverse'

# 统计原始数据的分类
original_stats = defaultdict(list)
for qid in original_queries:
    qtype = classify_query(original_data[qid])
    original_stats[qtype].append(qid)

print("\n原始数据（800个query）分类统计：")
for qtype in ['diverse', 'all0', 'all1', 'all06']:
    count = len(original_stats[qtype])
    print(f"  {qtype}: {count} 个")

# 统计扩展后的数据
# 从diverse数据中获取扩展的query
diverse_extended = diverse_queries - original_queries
all0_extended = all0_queries - original_queries

print("\n" + "=" * 60)
print("扩展数据来源分析")
print("=" * 60)

# 分析diverse扩展数据中的query类型
diverse_ext_stats = defaultdict(list)
for qid in diverse_queries:
    if qid in original_queries:
        continue  # 跳过原始数据
    qtype = classify_query(diverse_data[qid])
    diverse_ext_stats[qtype].append(qid)

print("\n从 all1_all06_diverse 扩展的数据：")
for qtype in ['diverse', 'all0', 'all1', 'all06']:
    count = len(diverse_ext_stats[qtype])
    if count > 0:
        print(f"  {qtype}: {count} 个")

# 分析all0扩展数据
all0_ext_stats = defaultdict(list)
for qid in all0_queries:
    if qid in original_queries:
        continue
    qtype = classify_query(all0_data[qid])
    all0_ext_stats[qtype].append(qid)

print("\n从 all0_aggressive 扩展的数据：")
for qtype in ['diverse', 'all0', 'all1', 'all06']:
    count = len(all0_ext_stats[qtype])
    if count > 0:
        print(f"  {qtype}: {count} 个")

# 统计合并后的总数
print("\n" + "=" * 60)
print("合并后数据统计")
print("=" * 60)

merged_stats = defaultdict(list)
for qid in merged_queries:
    qtype = classify_query(data[qid])
    merged_stats[qtype].append(qid)

print(f"\n合并后总query数: {len(merged_queries)}")
for qtype in ['diverse', 'all0', 'all1', 'all06']:
    count = len(merged_stats[qtype])
    print(f"  {qtype}: {count} 个")

# 计算扩展数量
print("\n" + "=" * 60)
print("扩展数量汇总")
print("=" * 60)

print(f"\n原始数据: {len(original_queries)} 个query")
print(f"合并后数据: {len(merged_queries)} 个query")
print(f"总扩展数量: {len(merged_queries) - len(original_queries)} 个query")

# 按类型统计扩展
print("\n按类型统计扩展：")
for qtype in ['diverse', 'all0', 'all1', 'all06']:
    orig = len(original_stats[qtype])
    merged = len(merged_stats[qtype])
    extended = merged - orig
    print(f"  {qtype}: {orig} -> {merged} (扩展 {extended} 个)")
