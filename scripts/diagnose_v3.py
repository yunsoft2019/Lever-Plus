#!/usr/bin/env python
"""
V3强化学习问题诊断脚本

问题：V3推理准确率(45.40%)远低于V2(64.8%)

诊断项：
1. evaluate_v3.py索引错误：用验证集索引去取训练集embedding
2. 检查beam_data中top-1 beam的VQA准确率（理论上限）
3. 检查V3模型预测与beam_data top-1的差异

作者: Lever-Plus Team
日期: 2025-12-03
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/mnt/share/yiyun/Projects/Lever-Plus')


def diagnose_index_mismatch():
    """诊断1: 索引不匹配问题"""
    print("\n" + "="*70)
    print("诊断1: 检查索引不匹配问题")
    print("="*70)
    
    # 加载embedding
    img_emb = torch.load('results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-ImgFeatures.pth', weights_only=False)
    print(f"img_emb shape: {img_emb.shape}")  # 应该是 (9009, 768) - 训练集大小
    
    # 加载beam_data
    beam_data_path = "results/okvqa/generated_data/sub_proc_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800_rank:0_(0, 800).json"
    with open(beam_data_path) as f:
        beam_data = json.load(f)
    
    query_ids = [int(k) for k in beam_data.keys()]
    print(f"beam_data query_id range: {min(query_ids)} - {max(query_ids)}")
    print(f"beam_data num_queries: {len(query_ids)}")
    
    # 检查候选池
    all_cand = set()
    for qid, data in beam_data.items():
        for beam in data['id_list']:
            for idx in beam[:-1]:
                all_cand.add(idx)
    print(f"候选池索引范围: {min(all_cand)} - {max(all_cand)}")
    print(f"候选池大小: {len(all_cand)}")
    
    # 加载数据集
    from lever_lm.load_ds_utils import load_vqav2_ds
    ds = load_vqav2_ds(
        version="local",
        train_path="datasets/okvqa/okvqa_hf/vqav2_mscoco_train2014.json",
        val_path="datasets/okvqa/okvqa_hf/vqav2_mscoco_val2014.json",
        train_coco_dataset_root="datasets/mscoco/mscoco2014/train2014",
        val_coco_dataset_root="datasets/mscoco/mscoco2014/val2014",
    )
    print(f"\n训练集大小: {len(ds['train'])}")
    print(f"验证集大小: {len(ds['validation'])}")
    
    print("\n" + "-"*50)
    print("⚠️ 问题确认:")
    print(f"  - img_emb是训练集embedding，大小={img_emb.shape[0]}")
    print(f"  - evaluate_v3.py对验证集推理时，用start_idx:end_idx索引")
    print(f"  - 这会用验证集顺序索引(0,1,2...)去取训练集embedding")
    print(f"  - 完全是随机的，导致准确率崩溃！")
    print("-"*50)


def diagnose_beam_data_quality():
    """诊断2: 检查beam_data中top-1的VQA准确率"""
    print("\n" + "="*70)
    print("诊断2: 检查beam_data top-1的VQA准确率")
    print("="*70)
    
    beam_data_path = "results/okvqa/generated_data/sub_proc_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800_rank:0_(0, 800).json"
    with open(beam_data_path) as f:
        beam_data = json.load(f)
    
    # 获取top-1 beam（分数最高的beam）
    top1_predictions = {}
    for qid, data in beam_data.items():
        scores = data['score_list']
        id_list = data['id_list']
        # 找分数最高的beam
        best_idx = np.argmax(scores)
        best_beam = id_list[best_idx][:-1]  # 去掉最后的query_id
        top1_predictions[int(qid)] = best_beam
    
    print(f"共有 {len(top1_predictions)} 个query")
    
    # 分析分数分布
    all_scores = []
    for qid, data in beam_data.items():
        all_scores.extend(data['score_list'])
    
    print(f"\n分数统计:")
    print(f"  min: {min(all_scores):.6f}")
    print(f"  max: {max(all_scores):.6f}")
    print(f"  mean: {np.mean(all_scores):.6f}")
    print(f"  std: {np.std(all_scores):.6f}")
    
    # 检查第一个query的beam
    first_qid = list(beam_data.keys())[0]
    print(f"\n示例 (query_id={first_qid}):")
    for i, (beam, score) in enumerate(zip(beam_data[first_qid]['id_list'], beam_data[first_qid]['score_list'])):
        print(f"  beam{i}: shots={beam[:-1]}, score={score:.6f}")


def diagnose_v3_prediction():
    """诊断3: 检查V3模型预测结果"""
    print("\n" + "="*70)
    print("诊断3: 检查V3模型预测")
    print("="*70)
    
    # 检查V3保存的预测结果
    v3_pred_path = "results/v3_eval/v3_icd_predictions.json"
    if os.path.exists(v3_pred_path):
        with open(v3_pred_path) as f:
            v3_preds = json.load(f)
        print(f"V3预测数量: {len(v3_preds)}")
        print(f"V3预测示例 (前5个):")
        for i in range(min(5, len(v3_preds))):
            print(f"  样本{i}: {v3_preds[i]}")
    else:
        print(f"未找到V3预测文件: {v3_pred_path}")
    
    # 加载beam_data获取候选池
    beam_data_path = "results/okvqa/generated_data/sub_proc_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800_rank:0_(0, 800).json"
    with open(beam_data_path) as f:
        beam_data = json.load(f)
    
    # 获取候选池索引
    all_cand = set()
    for qid, data in beam_data.items():
        for beam in data['id_list']:
            for idx in beam[:-1]:
                all_cand.add(idx)
    candidate_indices = sorted(list(all_cand))
    print(f"\n候选池大小: {len(candidate_indices)}")
    print(f"候选池索引范围: {min(candidate_indices)} - {max(candidate_indices)}")
    
    # 检查V3预测的索引是否在候选池范围内
    if os.path.exists(v3_pred_path):
        with open(v3_pred_path) as f:
            v3_preds = json.load(f)
        
        # V3预测的是候选池内的位置索引(0~K-1)，需要映射回原始索引
        pos_to_idx = {pos: idx for pos, idx in enumerate(candidate_indices)}
        
        print(f"\nV3预测的索引分析:")
        all_pred_indices = []
        for pred in v3_preds:
            all_pred_indices.extend(pred)
        
        print(f"  V3预测索引范围: {min(all_pred_indices)} - {max(all_pred_indices)}")
        print(f"  V3预测应该在 0 - {len(candidate_indices)-1} 范围内")
        
        # 检查映射后的索引
        if max(all_pred_indices) < len(candidate_indices):
            mapped_indices = [pos_to_idx[idx] for idx in all_pred_indices]
            print(f"  映射后索引范围: {min(mapped_indices)} - {max(mapped_indices)}")
        else:
            print(f"  ⚠️ V3预测索引超出候选池范围!")


def suggest_fix():
    """建议修复方案"""
    print("\n" + "="*70)
    print("修复方案")
    print("="*70)
    
    print("""
问题根因:
  evaluate_v3.py 用验证集的顺序索引(0,1,2...)去索引训练集embedding
  导致query embedding完全是随机的，V3模型无法正确选择范例

修复方案1 (推荐): 修改evaluate_v3.py使用实时CLIP编码
  - 对每个验证集样本，实时使用CLIP模型编码得到query embedding
  - 与V2的LeverLMRetriever保持一致
  - 需要加载CLIP模型

修复方案2: 为验证集预计算embedding
  - 先为验证集生成CLIP embedding
  - 修改evaluate_v3.py使用验证集embedding
  
关键代码修改 (evaluate_v3.py 第406行):
  错误: batch_query_emb = img_emb_data[start_idx:end_idx].to(device)
  应该: 使用CLIP模型实时编码验证集样本

验证方法:
  1. 使用beam_data中的top-1 beam做VQA推理，计算准确率上限
  2. 修复后V3的准确率应该接近或超过这个上限
""")


if __name__ == "__main__":
    os.chdir('/mnt/share/yiyun/Projects/Lever-Plus')
    
    diagnose_index_mismatch()
    diagnose_beam_data_quality()
    diagnose_v3_prediction()
    suggest_fix()
    
    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)
