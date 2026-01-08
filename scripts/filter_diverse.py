import json

with open('results/vqav2/generated_data/rl_data_merged_full_diverse.json', 'r') as f:
    data = json.load(f)

diverse_data = {}
for qid, qdata in data.items():
    candidates = qdata.get('pointer_candidates', [])
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    if len(set(scores)) >= 2:
        diverse_data[qid] = qdata

total_pos = total_cands = 0
for qid, qdata in diverse_data.items():
    for c in qdata.get('pointer_candidates', []):
        total_cands += 1
        if c.get('vqa_acc_score', 0.0) > 0:
            total_pos += 1

print(f'Diverse query: {len(diverse_data)}')
print(f'正样本比例: {total_pos/total_cands*100:.1f}%')

with open('results/vqav2/generated_data/rl_data_full_diverse_only.json', 'w') as f:
    json.dump(diverse_data, f, indent=2, ensure_ascii=False)
print('已保存')
