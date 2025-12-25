"""
RL 数据多样性补全脚本

根据《RL 数据多样性补全计划书》实现：
- 对全0的query补出至少一个0.6或1.0
- 对全1.0的query补出至少一个0或0.6
- 对全0.6的query补出至少一个0或1.0

最终让每个query的候选集合至少有2档reward。

使用方法：
    python -m lever_lm.models.v3.augment_rl_data_diversity \
        --input_rl_data ./results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval.json \
        --query_embeddings ./results/okvqa/cache/query_embeddings.pt \
        --candidate_embeddings ./results/okvqa/cache/candidate_embeddings.pt \
        --output_path ./results/okvqa/generated_data/rl_data_RandSampler_v4_strictEval_diverse.json \
        --sft_ckpt <path_to_sft_checkpoint> \
        --vqa_model Qwen/Qwen2.5-VL-3B-Instruct \
        --device cuda:0

作者: Lever-Plus Team
日期: 2025-12-16
参考: RL 数据多样性补全计划书
"""

import json
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
import os
import sys
import tempfile
import contextlib
from io import StringIO
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from tqdm import tqdm
import itertools
import random
import numpy as np

# 导入项目模块
from lever_lm.models.v3.generate_rl_data import (
    compute_vqa_accuracy,
    build_vqa_prompt_and_generate,
    load_sft_model,
    load_vqa_model
)
from lever_lm.models.v3.rl_data_generation import (
    beam_search_pointer,
    sample_pointer_with_temperature
)
from lever_lm.models.v3 import PointerSelectorV3
from lever_lm.utils import init_interface
from open_mmicl.metrics.vqa_metrics import VQA, VQAEval

# 导入根目录的utils.py
import importlib.util
_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
vqa_postprocess = _root_utils.vqa_postprocess
load_ds = _root_utils.load_ds


# ==================== 数据加载 ====================

def load_rl_data(rl_data_path: str) -> Dict:
    """加载RL数据"""
    with open(rl_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_embeddings(query_emb_path: str, cand_emb_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载embeddings"""
    query_emb = torch.load(query_emb_path, map_location='cpu')
    cand_emb = torch.load(cand_emb_path, map_location='cpu')
    
    # 处理不同的存储格式
    if isinstance(query_emb, dict):
        # 如果是字典格式，需要转换为tensor
        query_ids = sorted(query_emb.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        query_emb = torch.stack([query_emb[qid] for qid in query_ids])
    
    if isinstance(cand_emb, dict):
        cand_ids = sorted(cand_emb.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        cand_emb = torch.stack([cand_emb[cid] for cid in cand_ids])
    
    return query_emb, cand_emb


# ==================== Query分类 ====================

def classify_query(candidates: List[Dict]) -> Tuple[str, Set[float]]:
    """
    对query进行分类
    
    Returns:
        (aug_type, unique_scores)
        aug_type: 'all0' / 'all1' / 'all06' / 'diverse'
    """
    scores = [c.get('vqa_acc_score', 0.0) for c in candidates]
    unique_scores = set(scores)
    
    if len(unique_scores) >= 2:
        return 'diverse', unique_scores
    
    if len(unique_scores) == 1:
        score = list(unique_scores)[0]
        if abs(score - 0.0) < 1e-6:
            return 'all0', unique_scores
        elif abs(score - 1.0) < 1e-6:
            return 'all1', unique_scores
        elif abs(score - 0.6) < 1e-6:
            return 'all06', unique_scores
        else:
            return 'diverse', unique_scores  # 其他单一值也视为diverse
    
    return 'diverse', unique_scores


# ==================== 候选生成工具 ====================

def compute_similarity_ranks(
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor
) -> Tuple[torch.Tensor, List[int]]:
    """
    计算query与所有candidate的相似度，并返回排序后的索引
    
    Returns:
        (similarities, ranked_indices)
        similarities: [K] cosine相似度
        ranked_indices: 从高到低排序的索引列表
    """
    # 归一化
    query_emb_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)  # [1, d]
    cand_emb_norm = F.normalize(cand_emb, p=2, dim=1)  # [K, d]
    
    # 计算cosine相似度
    similarities = torch.mm(query_emb_norm, cand_emb_norm.t()).squeeze(0)  # [K]
    
    # 排序（从高到低）
    _, ranked_indices = torch.sort(similarities, descending=True)
    ranked_indices = ranked_indices.tolist()
    
    return similarities, ranked_indices


def propose_topL_pairs(
    ranked_indices: List[int],
    L: int,
    num_pairs: Optional[int] = None,
    shot_num: int = 2,
    seen: Optional[Set[Tuple[int, ...]]] = None,
    pair_sampling_if_large: int = 20000  # L>120时随机采样pair数
) -> List[Tuple[int, ...]]:
    """
    从Top-L区间生成pair组合（按照方案优化）
    
    Args:
        ranked_indices: 排序后的索引列表（从高到低）
        L: Top区间大小
        num_pairs: 需要生成的pair数量（如果None，则根据L自动决定）
        shot_num: shot数量（默认2）
        seen: 已见过的pointer集合（用于去重）
        pair_sampling_if_large: L>120时随机采样pair数
    
    Returns:
        pairs: List[Tuple[int, ...]]，每个tuple是一个pointer
    """
    if seen is None:
        seen = set()
    
    top_indices = ranked_indices[:L]
    
    # 生成所有组合
    if shot_num == 2:
        all_combinations = list(itertools.combinations(top_indices, 2))
    else:
        all_combinations = list(itertools.combinations(top_indices, shot_num))
    
    # 去重
    valid_pairs = []
    for pair in all_combinations:
        sorted_pair = tuple(sorted(pair))
        if sorted_pair not in seen:
            valid_pairs.append(sorted_pair)
    
    # 按照方案：如果L <= 120，全枚举；如果L > 120，随机采样
    if num_pairs is None:
        if L <= 120:
            # 全枚举（但限制最大数量避免内存问题）
            max_enum = 10000
            if len(valid_pairs) > max_enum:
                valid_pairs = random.sample(valid_pairs, max_enum)
        else:
            # L > 120，随机采样
            num_pairs = min(pair_sampling_if_large, len(valid_pairs))
            if len(valid_pairs) > num_pairs:
                valid_pairs = random.sample(valid_pairs, num_pairs)
    else:
        # 如果指定了num_pairs，按指定数量采样
        if len(valid_pairs) > num_pairs:
            valid_pairs = random.sample(valid_pairs, num_pairs)
    
    return valid_pairs


def propose_bottom_pairs(
    ranked_indices: List[int],
    B: int,
    num_pairs: int,
    shot_num: int = 2,
    seen: Optional[Set[Tuple[int, ...]]] = None
) -> List[Tuple[int, ...]]:
    """
    从Bottom-B区间采样pair
    
    Args:
        ranked_indices: 排序后的索引列表（从高到低）
        B: Bottom区间大小
        num_pairs: 需要生成的pair数量
        shot_num: shot数量（默认2）
        seen: 已见过的pointer集合（用于去重）
    
    Returns:
        pairs: List[Tuple[int, ...]]，每个tuple是一个pointer
    """
    if seen is None:
        seen = set()
    
    K = len(ranked_indices)
    bottom_indices = ranked_indices[K-B:]
    
    # 随机采样
    valid_pairs = []
    attempts = 0
    max_attempts = num_pairs * 10
    
    while len(valid_pairs) < num_pairs and attempts < max_attempts:
        if shot_num == 2:
            pair = tuple(sorted(random.sample(bottom_indices, 2)))
        else:
            pair = tuple(sorted(random.sample(bottom_indices, shot_num)))
        
        if pair not in seen:
            valid_pairs.append(pair)
        
        attempts += 1
    
    return valid_pairs


def propose_mix_pairs(
    ranked_indices: List[int],
    L: int,
    B: int,
    num_pairs: int,
    shot_num: int = 2,
    seen: Optional[Set[Tuple[int, ...]]] = None
) -> List[Tuple[int, ...]]:
    """
    生成Top+Bottom混合pair（Top一个，Bottom一个）
    
    Args:
        ranked_indices: 排序后的索引列表（从高到低）
        L: Top区间大小
        B: Bottom区间大小
        num_pairs: 需要生成的pair数量
        shot_num: shot数量（默认2，这里固定为Top+Bottom各一个）
        seen: 已见过的pointer集合（用于去重）
    
    Returns:
        pairs: List[Tuple[int, ...]]，每个tuple是一个pointer
    """
    if seen is None:
        seen = set()
    
    K = len(ranked_indices)
    top_indices = ranked_indices[:L]
    bottom_indices = ranked_indices[K-B:]
    
    # 生成Top+Bottom组合
    valid_pairs = []
    attempts = 0
    max_attempts = num_pairs * 10
    
    while len(valid_pairs) < num_pairs and attempts < max_attempts:
        if shot_num == 2:
            top_idx = random.choice(top_indices)
            bottom_idx = random.choice(bottom_indices)
            pair = tuple(sorted([top_idx, bottom_idx]))
        else:
            # shot_num > 2时，从Top选1个，从Bottom选shot_num-1个
            top_idx = random.choice(top_indices)
            bottom_indices_sample = random.sample(bottom_indices, shot_num - 1)
            pair = tuple(sorted([top_idx] + bottom_indices_sample))
        
        if pair not in seen:
            valid_pairs.append(pair)
        
        attempts += 1
    
    return valid_pairs


def rank_proposals_by_proxy(
    proposals: List[Tuple[int, ...]],
    similarities: torch.Tensor,
    aug_type: str,
    return_scores: bool = False
) -> List[Tuple[int, ...]]:
    """
    根据proxy分数对proposals排序（按照方案：pair_sim = (sim[i] + sim[j]) / 2）
    
    Args:
        proposals: 候选pointer列表
        similarities: [K] 相似度分数
        aug_type: 'all0' / 'all1' / 'all06'
        return_scores: 是否返回带分数的列表
    
    Returns:
        如果return_scores=False: 排序后的proposals列表
        如果return_scores=True: List[Tuple[Tuple[int, ...], float]]，每个元素是(pair, proxy_score)
    """
    def compute_pair_sim(pair: Tuple[int, ...]) -> float:
        """计算pair的平均相似度（方案推荐：pair_sim = (sim[i] + sim[j]) / 2）"""
        return float(similarities[list(pair)].mean().item())
    
    # 计算每个pair的proxy分数
    pair_scores = [(pair, compute_pair_sim(pair)) for pair in proposals]
    
    # 根据aug_type排序
    if aug_type == 'all0':
        # all0需要找正例，按相似度从高到低
        pair_scores.sort(key=lambda x: x[1], reverse=True)
    elif aug_type == 'all1':
        # all1需要找负例，按相似度从低到高
        pair_scores.sort(key=lambda x: x[1], reverse=False)
    else:  # all06
        # all06两边都要，保持原序或随机
        random.shuffle(pair_scores)
    
    if return_scores:
        return pair_scores
    else:
        return [pair for pair, _ in pair_scores]


def propose_swap_candidates(
    base_pair: Tuple[int, ...],
    ranked_indices: List[int],
    L: int,
    swap_trials: int = 10,
    shot_num: int = 2,
    seen: Optional[Set[Tuple[int, ...]]] = None,
    check_seen: bool = True,  # 是否检查seen集合（Level-3策略中应该设为False）
    use_diverse_sampling: bool = True,  # 是否使用多样性采样
    similarities: Optional[torch.Tensor] = None  # 相似度分数（用于多样性采样）
) -> List[Tuple[int, ...]]:
    """
    Level-3策略：生成swap候选（单边swap和双边swap）
    
    优化版本：
    1. 支持多样性采样（避免选择过于相似的候选）
    2. 使用不同的ranked_indices区间（不只是Top-L）
    3. 增加swap策略的多样性
    
    Args:
        base_pair: 基础pair（最相关的pair）
        ranked_indices: 排序后的索引列表（从高到低）
        L: Top-L区间大小（用于选择替换候选）
        swap_trials: 每种swap类型的尝试次数
        shot_num: shot数量（默认2）
        seen: 已见过的pointer集合（用于去重，但Level-3中通常不检查）
        check_seen: 是否检查seen集合（Level-3策略中设为False，因为目的是生成新组合）
        use_diverse_sampling: 是否使用多样性采样
        similarities: 相似度分数（用于多样性采样）
    
    Returns:
        swap_candidates: List[Tuple[int, ...]]，包含单边swap和双边swap的候选
    """
    if seen is None:
        seen = set()
    
    if len(base_pair) != 2:
        return []
    
    top_indices = ranked_indices[:L]
    swap_candidates = []
    
    a, b = base_pair
    
    # 策略1：单边swap（固定a，替换b）
    # 从Top-L中选择一个不同于a和b的候选
    available_1 = [idx for idx in top_indices if idx != a and idx != b]
    if len(available_1) > 0:
        if use_diverse_sampling and similarities is not None and len(available_1) > swap_trials:
            # 多样性采样：选择与a相似度不同的候选
            # 计算available_1中每个候选与a的相似度
            sim_scores = [(idx, float(similarities[idx])) for idx in available_1]
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            # 从不同相似度区间采样
            num_intervals = min(swap_trials, len(sim_scores))
            interval_size = len(sim_scores) // num_intervals
            sampled_indices = []
            for i in range(num_intervals):
                start_idx = i * interval_size
                end_idx = start_idx + interval_size if i < num_intervals - 1 else len(sim_scores)
                if start_idx < len(sim_scores):
                    sampled_idx = random.choice(sim_scores[start_idx:end_idx])[0]
                    sampled_indices.append(sampled_idx)
            for b_new in sampled_indices:
                new_pair = tuple(sorted([a, b_new]))
                if not check_seen or new_pair not in seen:
                    swap_candidates.append(new_pair)
        else:
            # 随机采样
            for _ in range(swap_trials):
                if len(available_1) > 0:
                    b_new = random.choice(available_1)
                    new_pair = tuple(sorted([a, b_new]))
                    if not check_seen or new_pair not in seen:
                        swap_candidates.append(new_pair)
    
    # 策略2：单边swap（固定b，替换a）
    available_2 = [idx for idx in top_indices if idx != a and idx != b]
    if len(available_2) > 0:
        if use_diverse_sampling and similarities is not None and len(available_2) > swap_trials:
            # 多样性采样：选择与b相似度不同的候选
            sim_scores = [(idx, float(similarities[idx])) for idx in available_2]
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            num_intervals = min(swap_trials, len(sim_scores))
            interval_size = len(sim_scores) // num_intervals
            sampled_indices = []
            for i in range(num_intervals):
                start_idx = i * interval_size
                end_idx = start_idx + interval_size if i < num_intervals - 1 else len(sim_scores)
                if start_idx < len(sim_scores):
                    sampled_idx = random.choice(sim_scores[start_idx:end_idx])[0]
                    sampled_indices.append(sampled_idx)
            for a_new in sampled_indices:
                new_pair = tuple(sorted([a_new, b]))
                if not check_seen or new_pair not in seen:
                    swap_candidates.append(new_pair)
        else:
            # 随机采样
            for _ in range(swap_trials):
                if len(available_2) > 0:
                    a_new = random.choice(available_2)
                    new_pair = tuple(sorted([a_new, b]))
                    if not check_seen or new_pair not in seen:
                        swap_candidates.append(new_pair)
    
    # 策略3：双边swap（同时替换a和b）
    available_3 = [idx for idx in top_indices if idx != a and idx != b]
    if len(available_3) >= 2:
        for _ in range(swap_trials):
            if len(available_3) >= 2:
                a_new, b_new = random.sample(available_3, 2)
                new_pair = tuple(sorted([a_new, b_new]))
                if not check_seen or new_pair not in seen:
                    swap_candidates.append(new_pair)
    
    # 策略4：使用不同的ranked_indices区间（不只是Top-L）
    # 从L到2L区间选择候选（中等相似度）
    if L < len(ranked_indices):
        mid_indices = ranked_indices[L:min(2*L, len(ranked_indices))]
        available_mid = [idx for idx in mid_indices if idx != a and idx != b]
        if len(available_mid) >= 2:
            # 双边swap：从中等相似度区间选择
            for _ in range(swap_trials // 2):  # 使用一半的trials
                if len(available_mid) >= 2:
                    a_new, b_new = random.sample(available_mid, 2)
                    new_pair = tuple(sorted([a_new, b_new]))
                    if not check_seen or new_pair not in seen:
                        swap_candidates.append(new_pair)
    
    # 去重（内部去重，不依赖seen）
    unique_candidates = []
    seen_here = set()
    for candidate in swap_candidates:
        if candidate not in seen_here:
            unique_candidates.append(candidate)
            seen_here.add(candidate)
    
    return unique_candidates


# ==================== VQA评测 ====================

def evaluate_pointer_candidate(
    vqa_interface,
    query_item: Dict,
    candidate_pool: Optional[List[Dict]],
    pointer: Tuple[int, ...],
    rl_data: Optional[Dict] = None,
    query_id: Optional[str] = None,
    val_ques_path: Optional[str] = None,
    val_ann_path: Optional[str] = None,
    train_ques_path: Optional[str] = None,
    train_ann_path: Optional[str] = None,
    vqa_train_cache: Optional[VQA] = None,
    vqa_val_cache: Optional[VQA] = None,
    generation_kwargs: Optional[Dict] = None,
    candidate_indices: Optional[List[int]] = None  # 用于将dataset索引转换为pool索引
) -> Dict:
    """
    评测一个pointer candidate
    
    Args:
        candidate_pool: 候选池（可选，如果为None则从rl_data中获取）
        rl_data: RL数据（可选，用于获取candidate信息）
        query_id: Query ID（可选，用于从rl_data获取信息）
    
    Returns:
        candidate_dict: {
            "pointer": [i, j, ...],
            "vqa_pred_answer": str,
            "vqa_correct": int,
            "vqa_acc_score": float,
            "vqa_eval_mode": str,
            "eval_failed": bool,
            ...
        }
    """
    # 获取示例
    # 如果candidate_pool为None，尝试从rl_data中获取
    if candidate_pool is None:
        if rl_data is None or query_id is None:
            raise ValueError("candidate_pool为None时，必须提供rl_data和query_id")
        
        # 从rl_data中查找已有的candidate来推断candidate_pool的结构
        # 这里假设candidate_pool的索引与pointer中的索引对应
        # 实际使用时，可能需要从dataset加载
        examples = []
        for idx in pointer:
            # 尝试从rl_data中找到包含该索引的candidate
            found = False
            for qid, qdata in rl_data.items():
                for c in qdata.get("pointer_candidates", []):
                    if idx in c.get("pointer", []):
                        # 这里需要根据实际数据结构调整
                        # 暂时使用占位符
                        examples.append({"image": None, "question": "", "answer": ""})
                        found = True
                        break
                if found:
                    break
            if not found:
                examples.append({"image": None, "question": "", "answer": ""})
    else:
        # candidate_pool的索引是candidate_indices中的值（dataset索引），不是连续的0-N
        # candidate_pool[i]对应dataset[candidate_indices[i]]
        # 所以我们需要将pointer中的dataset索引转换为pool索引
        if len(candidate_pool) == 0:
            raise ValueError("candidate_pool为空，无法获取示例")
        
        if candidate_indices is None:
            # 如果没有提供candidate_indices，假设pointer就是pool索引（0-N）
            max_idx = max(pointer) if pointer else -1
            if max_idx >= len(candidate_pool):
                raise IndexError(f"Pointer索引 {max_idx} 超出candidate_pool范围 [0, {len(candidate_pool)-1}]")
            examples = [candidate_pool[idx] for idx in pointer]
        else:
            # 如果提供了candidate_indices，pointer中的索引是dataset索引
            # 需要转换为pool索引
            examples = []
            for idx in pointer:
                try:
                    pool_idx = candidate_indices.index(idx)
                    if pool_idx < len(candidate_pool):
                        examples.append(candidate_pool[pool_idx])
                    else:
                        raise IndexError(f"转换后的pool索引 {pool_idx} 超出candidate_pool范围 [0, {len(candidate_pool)-1}]")
                except ValueError:
                    # 索引不在candidate_indices中
                    raise IndexError(f"Pointer索引 {idx} 不在candidate_indices中，无法从candidate_pool获取")
    
    # 构建prompt并生成答案
    query_image = query_item.get('image')
    query_question = query_item.get('question', '')
    
    # 保存VQA生成的详细信息用于调试
    raw_generation = None
    prompt_text = None
    
    try:
        result = build_vqa_prompt_and_generate(
            interface=vqa_interface,
            image=query_image,
            question=query_question,
            ex1=examples[0],
            ex2=examples[1] if len(examples) > 1 else None,
            generation_kwargs=generation_kwargs
        )
        # build_vqa_prompt_and_generate 返回字典，提取 pred_answer 和其他信息
        if isinstance(result, dict):
            pred_answer = result.get("pred_answer", "")
            raw_generation = result.get("raw_generation", None)
            prompt_text = result.get("prompt_text", None)
        elif isinstance(result, str):
            pred_answer = result
        else:
            pred_answer = str(result) if result is not None else ""
        
        # 确保pred_answer是字符串，且不为None
        if pred_answer is None:
            pred_answer = ""
        elif not isinstance(pred_answer, str):
            pred_answer = str(pred_answer)
    except Exception as e:
        import traceback
        import sys
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"Warning: VQA generation failed for pointer {pointer}: {error_type}: {error_msg}")
        # 如果是strip相关的错误，打印详细的错误堆栈（只打印前3个）
        if 'strip' in error_msg.lower():
            if not hasattr(evaluate_pointer_candidate, '_strip_error_count'):
                evaluate_pointer_candidate._strip_error_count = 0
            evaluate_pointer_candidate._strip_error_count += 1
            if evaluate_pointer_candidate._strip_error_count <= 3:
                print(f"  【详细错误堆栈 #{evaluate_pointer_candidate._strip_error_count}】")
                print(f"  错误类型: {error_type}")
                print(f"  错误消息: {error_msg}")
                print(f"  完整堆栈:")
                exc_type, exc_value, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_tb, limit=10)
        pred_answer = ""
        raw_generation = None
        prompt_text = None
    
    # 获取ground truth
    gt_answers_raw = query_item.get('answers', [])
    question_id = query_item.get('question_id', str(query_item.get('query_id', '')))
    
    # 计算准确率
    used_file_metric = False
    eval_failed = False
    eval_split_used = None
    
    # 先尝试训练集文件
    if train_ques_path and train_ann_path:
        try:
            correct, acc_score, used_file_metric = compute_vqa_accuracy(
                pred_answer=pred_answer,
                ground_truth_answers=gt_answers_raw,
                question_id=question_id,
                val_ques_path=train_ques_path,
                val_ann_path=train_ann_path,
                vqa_cache=vqa_train_cache
            )
            if used_file_metric:
                eval_split_used = "train"
        except Exception as e:
            used_file_metric = False
    
    # 如果训练集文件未使用或失败，尝试验证集文件
    if not used_file_metric and val_ques_path and val_ann_path:
        try:
            correct, acc_score, used_file_metric = compute_vqa_accuracy(
                pred_answer=pred_answer,
                ground_truth_answers=gt_answers_raw,
                question_id=question_id,
                val_ques_path=val_ques_path,
                val_ann_path=val_ann_path,
                vqa_cache=vqa_val_cache
            )
            if used_file_metric:
                eval_split_used = "val"
        except Exception as e:
            used_file_metric = False
    
    # 如果都失败，标记为失败
    if not used_file_metric:
        eval_failed = True
        correct = 0
        acc_score = 0.0
    
    # 构建candidate字典
    candidate_dict = {
        "pointer": list(pointer),
        "gen_method": "aug_diverse",
        "vqa_pred_answer": pred_answer,
        "vqa_correct": int(correct),
        "vqa_acc_score": float(acc_score),
        "vqa_eval_mode": eval_split_used if used_file_metric else "fallback",
        "eval_failed": eval_failed
    }
    
    # 添加调试信息（如果存在）
    if raw_generation is not None:
        candidate_dict["raw_generation"] = raw_generation
    if prompt_text is not None:
        candidate_dict["prompt_text"] = prompt_text
    
    return candidate_dict


# ==================== 补全策略 ====================

def augment_all0_query(
    query_id: str,
    query_item: Dict,
    candidates: List[Dict],
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    sft_model: PointerSelectorV3,
    vqa_interface,
    candidate_pool: Optional[List[Dict]],
    rl_data: Optional[Dict],
    val_ques_path: Optional[str],
    val_ann_path: Optional[str],
    train_ques_path: Optional[str],
    train_ann_path: Optional[str],
    vqa_train_cache: Optional[VQA],
    vqa_val_cache: Optional[VQA],
    seen_pointers: Set[Tuple[int, ...]],
    max_candidates: int = 24,
    max_eval_budget: int = 18,
    generation_kwargs: Optional[Dict] = None,
    candidate_indices: Optional[List[int]] = None  # 用于将pool索引转换为dataset索引
) -> Tuple[List[Dict], Dict]:
    """
    补全All-Zero query（全0 → 补出0.6/1.0）
    
    策略（按照方案优化）：
    1. Level-1：Top-L组合枚举/采样（优先策略，成功率最高）
    2. Level-2：SFT模型生成候选（beam search + 温度采样，作为补强）
    
    Returns:
        (new_candidates, stats)
    """
    stats = {
        "aug_type": "all0",
        "rounds": 0,
        "eval_count": 0,
        "success": False,
        "new_candidates": []
    }
    
    # 计算相似度排序（用于后续的相似度策略）
    similarities, ranked_indices = compute_similarity_ranks(query_emb, cand_emb)
    
    # ========== Level-1：Top-L组合枚举/采样（优先策略，按照方案） ==========
    # 【改进】大幅增加搜索范围和评测数量，提高正样本生成率
    L_schedule = [50, 100, 200, 400, 600, 800, 1000]  # 扩大L值范围，覆盖更多候选（增加更多L值）
    E_per_round = 15  # 增加每轮评测数量（从10增加到15）
    pair_sampling_if_large = 100000  # L>120时随机采样pair数（大幅增加采样数，从50000增加到100000）
    
    for round_idx, L in enumerate(L_schedule):
        if stats["eval_count"] >= max_eval_budget:
            break
        
        # 对于All-Zero query，允许超过max_candidates限制，直到找到正样本或预算用完
        # 但设置一个软限制：如果超过max_candidates太多（比如+20），就停止
        soft_limit = max_candidates + 20  # 允许超过20个候选（增加限制）
        if len(candidates) + len(stats["new_candidates"]) >= soft_limit:
            break
        
        # 检查是否已成功（方案要求：找到正样本立即停止）
        current_scores = [c.get('vqa_acc_score', 0.0) for c in candidates + stats["new_candidates"]]
        if any(score > 0.0 for score in current_scores):
            stats["success"] = True
            stats["rounds"] = round_idx + 1
            break
        
        # 生成proposals（按照方案：L<=120全枚举，L>120随机采样）
        proposals = propose_topL_pairs(
            ranked_indices=ranked_indices,
            L=L,
            num_pairs=None,  # 自动决定
            shot_num=2,
            seen=seen_pointers,
            pair_sampling_if_large=pair_sampling_if_large
        )
        
        if not proposals:
            continue
        
        # 按proxy排序（在转换索引之前，因为similarities是基于pool索引的）
        # 返回带分数的列表，以便记录proxy_score
        proposals_with_scores = rank_proposals_by_proxy(proposals, similarities, 'all0', return_scores=True)
        
        # 如果提供了candidate_indices，需要将pool索引转换为dataset索引（在排序之后）
        converted_proposals_with_scores = []
        for p, proxy_score in proposals_with_scores:
            try:
                # 确保所有索引都在有效范围内
                if candidate_indices is not None:
                    if all(0 <= idx < len(candidate_indices) for idx in p):
                        converted_p = tuple(candidate_indices[idx] for idx in p)
                        converted_proposals_with_scores.append((converted_p, proxy_score))
                    else:
                        # 跳过无效的索引
                        invalid_indices = [idx for idx in p if not (0 <= idx < len(candidate_indices))]
                        print(f"Warning: 跳过无效的proposal {p}，索引 {invalid_indices} 超出范围 [0, {len(candidate_indices)})")
                else:
                    converted_proposals_with_scores.append((p, proxy_score))
            except (IndexError, TypeError) as e:
                print(f"Warning: 转换proposal {p} 时出错: {e}")
                continue
        
        # 选择top E个进行评测
        eval_proposals_with_scores = converted_proposals_with_scores[:E_per_round]
        
        # 评测
        for pointer, proxy_score in eval_proposals_with_scores:
            if stats["eval_count"] >= max_eval_budget:
                break
            
            # 对于All-Zero query，允许超过max_candidates限制
            soft_limit = max_candidates + 6
            if len(candidates) + len(stats["new_candidates"]) >= soft_limit:
                break
            
            try:
                candidate_dict = evaluate_pointer_candidate(
                    vqa_interface=vqa_interface,
                    query_item=query_item,
                    candidate_pool=candidate_pool,
                    pointer=list(pointer),
                    rl_data=rl_data,
                    query_id=query_id,
                    val_ques_path=val_ques_path,
                    val_ann_path=val_ann_path,
                    train_ques_path=train_ques_path,
                    train_ann_path=train_ann_path,
                    vqa_train_cache=vqa_train_cache,
                    vqa_val_cache=vqa_val_cache,
                    generation_kwargs=generation_kwargs,
                    candidate_indices=candidate_indices  # 用于索引转换
                )
                
                # 添加元信息（按照方案要求）
                candidate_dict["aug_type"] = "all0"
                candidate_dict["aug_round"] = round_idx + 1  # Level-1的轮次
                candidate_dict["gen_method"] = f"aug_pos_topL_L{L}"  # 方案推荐格式
                candidate_dict["proxy_score"] = float(proxy_score)  # 添加proxy_score字段
                
                stats["new_candidates"].append(candidate_dict)
                stats["eval_count"] += 1
                seen_pointers.add(tuple(sorted(pointer)))
                
                # 方案要求：找到正样本立即停止
                if candidate_dict["vqa_acc_score"] > 0.0:
                    stats["success"] = True
                    stats["rounds"] = round_idx + 1
                    return stats["new_candidates"], stats
            
            except Exception as e:
                print(f"Warning: Evaluation failed for Level-1 pointer {pointer}: {e}")
                continue
    
    # ========== Level-2：SFT模型生成候选（作为补强策略） ==========
    # 如果Level-1失败，使用SFT模型生成候选
    if not stats["success"] and stats["eval_count"] < max_eval_budget and sft_model is not None:
        # 确保embeddings和模型在同一个设备上
        device = next(sft_model.parameters()).device
        query_emb_device = query_emb.to(device)
        cand_emb_device = cand_emb.to(device)
        
        # 获取query_id在candidate_indices中的索引（用于排除）
        exclude_indices = []
        if candidate_indices is not None:
            try:
                query_idx_in_pool = candidate_indices.index(int(query_id))
                exclude_indices = [query_idx_in_pool]
            except (ValueError, TypeError):
                pass
        
        # 1.1 Beam Search生成候选
        # 【改进】增加beam size，提高正样本生成率
        try:
            beam_results = beam_search_pointer(
                model=sft_model,
                query_emb=query_emb_device,
                cand_emb=cand_emb_device,
                num_beams=20,  # 增加beam size（从10增加到20）
                shot_num=2,
                exclude_indices=exclude_indices if exclude_indices else None
            )
            
            # 转换beam候选为proposals（pool索引）
            sft_proposals = []
            for beam_result in beam_results:
                pointer = beam_result.get("pointer", [])
                if len(pointer) == 2:
                    # 转换为dataset索引
                    if candidate_indices is not None:
                        try:
                            if all(0 <= idx < len(candidate_indices) for idx in pointer):
                                converted_pointer = tuple(candidate_indices[idx] for idx in pointer)
                                if converted_pointer not in seen_pointers:
                                    sft_proposals.append((converted_pointer, "beam", beam_result.get("logprob", 0.0)))
                        except (IndexError, TypeError):
                            continue
                    else:
                        if tuple(pointer) not in seen_pointers:
                            sft_proposals.append((tuple(pointer), "beam", beam_result.get("logprob", 0.0)))
            
            # 按logprob排序
            sft_proposals.sort(key=lambda x: x[2], reverse=True)
            
            # 评测beam search候选
            # 【改进】增加评测的beam候选数量
            for pointer_tuple, gen_method, logprob in sft_proposals[:25]:  # 增加评测数量（从15增加到25）
                if stats["eval_count"] >= max_eval_budget:
                    break
                # 对于All-Zero query，允许超过max_candidates限制
                soft_limit = max_candidates + 6
                if len(candidates) + len(stats["new_candidates"]) >= soft_limit:
                    break
                
                try:
                    candidate_dict = evaluate_pointer_candidate(
                        vqa_interface=vqa_interface,
                        query_item=query_item,
                        candidate_pool=candidate_pool,
                        pointer=list(pointer_tuple),
                        rl_data=rl_data,
                        query_id=query_id,
                        val_ques_path=val_ques_path,
                        val_ann_path=val_ann_path,
                        train_ques_path=train_ques_path,
                        train_ann_path=train_ann_path,
                        vqa_train_cache=vqa_train_cache,
                        vqa_val_cache=vqa_val_cache,
                        generation_kwargs=generation_kwargs,
                        candidate_indices=candidate_indices
                    )
                    
                    candidate_dict["aug_type"] = "all0"
                    # round会在后面统一更新（Level-2的round = Level-1的max_round + 1）
                    candidate_dict["aug_round"] = 999  # 临时值，后面会更新
                    candidate_dict["gen_method"] = f"sft_beam_{gen_method}"
                    
                    stats["new_candidates"].append(candidate_dict)
                    stats["eval_count"] += 1
                    seen_pointers.add(tuple(sorted(pointer_tuple)))
                    
                    if candidate_dict["vqa_acc_score"] > 0.0:
                        stats["success"] = True
                        # 更新所有SFT候选的round编号
                        max_level1_round = max([c.get("aug_round", 0) for c in stats["new_candidates"] if c.get("gen_method", "").startswith("aug_pos_topL")], default=0)
                        for c in stats["new_candidates"]:
                            if c.get("gen_method", "").startswith("sft_"):
                                c["aug_round"] = max_level1_round + 1
                        stats["rounds"] = max_level1_round + 1
                        return stats["new_candidates"], stats
                
                except Exception as e:
                    print(f"Warning: Evaluation failed for SFT beam pointer {pointer_tuple}: {e}")
                    continue
        
        except Exception as e:
            print(f"Warning: SFT beam search failed: {e}")
        
        # 1.2 温度采样生成候选
        # 【改进】增加温度值和采样数量，提高正样本生成率
        if not stats["success"] and stats["eval_count"] < max_eval_budget:
            temps = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]  # 增加更多温度值（从4个增加到6个）
            num_samples_per_temp = 8  # 增加每个温度的采样数（从5增加到8）
            
            for tau in temps:
                if stats["eval_count"] >= max_eval_budget:
                    break
                if stats["success"]:
                    break
                
                for _ in range(num_samples_per_temp):
                    if stats["eval_count"] >= max_eval_budget:
                        break
                    if len(candidates) + len(stats["new_candidates"]) >= max_candidates:
                        break
                    
                    try:
                        pointer, logprob = sample_pointer_with_temperature(
                            model=sft_model,
                            query_emb=query_emb_device,
                            cand_emb=cand_emb_device,
                            tau=tau,
                            exclude_indices=exclude_indices if exclude_indices else None
                        )
                        
                        # 转换为dataset索引
                        if candidate_indices is not None:
                            try:
                                if all(0 <= idx < len(candidate_indices) for idx in pointer):
                                    converted_pointer = tuple(candidate_indices[idx] for idx in pointer)
                                    pointer_tuple = tuple(sorted(converted_pointer))
                                else:
                                    continue
                            except (IndexError, TypeError):
                                continue
                        else:
                            pointer_tuple = tuple(sorted(pointer))
                        
                        if pointer_tuple in seen_pointers:
                            continue
                        
                        # 评测温度采样候选
                        candidate_dict = evaluate_pointer_candidate(
                            vqa_interface=vqa_interface,
                            query_item=query_item,
                            candidate_pool=candidate_pool,
                            pointer=list(pointer_tuple),
                            rl_data=rl_data,
                            query_id=query_id,
                            val_ques_path=val_ques_path,
                            val_ann_path=val_ann_path,
                            train_ques_path=train_ques_path,
                            train_ann_path=train_ann_path,
                            vqa_train_cache=vqa_train_cache,
                            vqa_val_cache=vqa_val_cache,
                            generation_kwargs=generation_kwargs,
                            candidate_indices=candidate_indices
                        )
                        
                        candidate_dict["aug_type"] = "all0"
                        # round会在后面统一更新（Level-2的round = Level-1的max_round + 1）
                        candidate_dict["aug_round"] = 999  # 临时值，后面会更新
                        candidate_dict["gen_method"] = f"sft_temp_{tau}"
                        
                        stats["new_candidates"].append(candidate_dict)
                        stats["eval_count"] += 1
                        seen_pointers.add(pointer_tuple)
                        
                        if candidate_dict["vqa_acc_score"] > 0.0:
                            stats["success"] = True
                            stats["rounds"] = 1
                            return stats["new_candidates"], stats
                    
                    except Exception as e:
                        print(f"Warning: SFT temperature sampling failed (tau={tau}): {e}")
                        continue
        
        # 更新round编号（SFT策略是Level-2，round从Level-1的轮次之后开始）
        if stats["new_candidates"]:
            max_level1_round = max([c.get("aug_round", 0) for c in stats["new_candidates"] if c.get("gen_method", "").startswith("aug_pos_topL")], default=0)
            for c in stats["new_candidates"]:
                if c.get("gen_method", "").startswith("sft_"):
                    c["aug_round"] = max_level1_round + 1
    
    # ========== Level-3：Swap搜索（兜底策略） ==========
    # 如果Level-1和Level-2都失败，使用swap搜索
    if not stats["success"] and stats["eval_count"] < max_eval_budget:
        # Level-3策略开始（调试日志已移除）
        
        # 【改进】大幅增加swap_trials和L值，提高正样本生成率
        swap_trials = 80  # 每种swap类型的尝试次数（从50增加到80）
        eval_E = 30  # 每轮评测30个（从20增加到30，因为现在有更多候选）
        max_eval_level3 = 60  # 增加Level-3的评估预算（从50增加到60），提高正样本生成率
        L_for_swap = min(800, len(ranked_indices))  # 【改进】增加L值（从600增加到800），覆盖更多候选
        
        # 优化：尝试多种base_pair选择策略
        # 策略1：选择proxy_score最高的pair（原有策略）
        # 策略2：选择中等相似度的pair
        # 策略3：随机选择多个不同的base_pair
        
        base_pairs_pool = []  # 存储多个base_pair候选
        
        # 策略1：从Level-1的候选中收集所有pair（按proxy_score排序）
        level1_pairs = []
        for c in stats["new_candidates"]:
            if c.get("gen_method", "").startswith("aug_pos_topL"):
                proxy = c.get("proxy_score", 0.0)
                pointer = c.get("pointer", [])
                if len(pointer) == 2:
                    # pointer是dataset索引，需要转换为pool索引
                    pool_pair = None
                    if candidate_indices is not None:
                        try:
                            pool_indices = []
                            for ds_idx in pointer:
                                if ds_idx in candidate_indices:
                                    pool_idx = candidate_indices.index(ds_idx)
                                    pool_indices.append(pool_idx)
                            if len(pool_indices) == 2:
                                pool_pair = tuple(sorted(pool_indices))
                        except (ValueError, TypeError):
                            pass
                    else:
                        pool_pair = tuple(sorted(pointer))
                    
                    if pool_pair is not None:
                        level1_pairs.append((pool_pair, proxy))
        
        # 按proxy_score排序
        level1_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 策略1：选择proxy_score最高的pair
        if level1_pairs:
            base_pairs_pool.append(level1_pairs[0][0])

        # 策略2：选择中等相似度的pair（proxy_score在中间位置的pair）
        if len(level1_pairs) > 2:
            mid_idx = len(level1_pairs) // 2
            mid_pair = level1_pairs[mid_idx][0]
            if mid_pair not in base_pairs_pool:
                base_pairs_pool.append(mid_pair)

        # 策略3：随机选择多个不同的pair（从不同proxy_score区间）
        if len(level1_pairs) > 3:
            # 将pairs分成3个区间：高、中、低
            num_intervals = min(3, len(level1_pairs))
            interval_size = len(level1_pairs) // num_intervals
            for i in range(num_intervals):
                start_idx = i * interval_size
                end_idx = start_idx + interval_size if i < num_intervals - 1 else len(level1_pairs)
                if start_idx < len(level1_pairs):
                    # 从每个区间随机选择一个pair
                    interval_pairs = level1_pairs[start_idx:end_idx]
                    if interval_pairs:
                        random_pair = random.choice(interval_pairs)[0]
                        if random_pair not in base_pairs_pool:
                            base_pairs_pool.append(random_pair)

        # 如果没有找到任何pair，使用ranked_indices的前两个（已经是pool索引）
        if not base_pairs_pool:
            if len(ranked_indices) >= 2:
                base_pairs_pool.append(tuple(sorted(ranked_indices[:2])))

        # 如果没有找到任何pair，使用ranked_indices的前两个（已经是pool索引）
        if not base_pairs_pool:
            if len(ranked_indices) >= 2:
                base_pairs_pool.append(tuple(sorted(ranked_indices[:2])))

        # 使用多个base_pair生成swap候选
        if base_pairs_pool and len(ranked_indices) > 2:
            # 生成swap候选（使用pool索引）
            # Level-3策略中不检查seen_pointers，因为目的是生成新的组合
            # 【改进】使用更大的L值以扩大搜索范围
            swap_L = L_for_swap  # 使用改进后的L值（800）
            
            all_swap_candidates = []
            
            # 对每个base_pair生成swap候选
            for base_pair_idx, best_pair_pool in enumerate(base_pairs_pool):
                # 确保best_pair_pool中的元素在ranked_indices的前L个中
                # 如果不在，跳过这个base_pair
                a, b = best_pair_pool
                if a not in ranked_indices[:swap_L] or b not in ranked_indices[:swap_L]:
                    # 如果best_pair_pool不在前L个中，尝试使用ranked_indices的前两个
                    if len(ranked_indices) >= 2:
                        best_pair_pool = tuple(sorted(ranked_indices[:2]))
                    else:
                        continue

                # 使用优化后的propose_swap_candidates（支持多样性采样）
                swap_candidates = propose_swap_candidates(
                    base_pair=best_pair_pool,
                    ranked_indices=ranked_indices,
                    L=swap_L,  # 使用更大的L值（最多1000）
                    swap_trials=swap_trials,
                    shot_num=2,
                    seen=seen_pointers,
                    check_seen=False,  # Level-3策略中不检查seen，允许生成新组合
                    use_diverse_sampling=True,  # 使用多样性采样
                    similarities=similarities  # 传入相似度分数用于多样性采样
                )
                
                if swap_candidates:
                    all_swap_candidates.extend(swap_candidates)

            # 去重
            unique_swap_candidates = []
            seen_swap = set()
            for candidate in all_swap_candidates:
                if candidate not in seen_swap:
                    unique_swap_candidates.append(candidate)
                    seen_swap.add(candidate)
            
            swap_candidates = unique_swap_candidates

            # 如果swap_candidates为空，尝试使用更大的L值
            if not swap_candidates and len(ranked_indices) > 2:
                # 使用第一个base_pair尝试所有ranked_indices
                if base_pairs_pool:
                    best_pair_pool = base_pairs_pool[0]
                    swap_candidates = propose_swap_candidates(
                        base_pair=best_pair_pool,
                        ranked_indices=ranked_indices,
                        L=len(ranked_indices),  # 使用所有ranked_indices
                        swap_trials=swap_trials,
                        shot_num=2,
                        seen=seen_pointers,
                        check_seen=False,
                        use_diverse_sampling=True,
                        similarities=similarities
                    )

            if swap_candidates:
                # 按proxy排序
                swap_candidates_with_scores = rank_proposals_by_proxy(
                    swap_candidates, similarities, 'all0', return_scores=True
                )
                
                # 如果提供了candidate_indices，需要将pool索引转换为dataset索引
                converted_swap_candidates = []
                conversion_failed_count = 0
                for p, proxy_score in swap_candidates_with_scores:
                    try:
                        if candidate_indices is not None:
                            if all(0 <= idx < len(candidate_indices) for idx in p):
                                converted_p = tuple(candidate_indices[idx] for idx in p)
                                converted_swap_candidates.append((converted_p, proxy_score))
                            else:
                                conversion_failed_count += 1
                                continue
                        else:
                            converted_swap_candidates.append((p, proxy_score))
                    except (IndexError, TypeError) as e:
                        conversion_failed_count += 1
                        continue

                # 选择top E个进行评测（但需要检查seen_pointers，避免重复评测）
                # 优化：使用更智能的候选选择方法
                # 1. 优先选择不在seen_pointers中的候选
                # 2. 如果候选不足，考虑proxy_score的多样性
                eval_swap_candidates_filtered = []
                seen_proxy_scores = set()
                
                for pointer, proxy_score in converted_swap_candidates:
                    pointer_tuple = tuple(sorted(pointer))
                    if pointer_tuple not in seen_pointers:
                        # 多样性检查：避免选择proxy_score过于相似的候选
                        # 如果已经有候选，检查proxy_score的差异
                        if len(eval_swap_candidates_filtered) == 0:
                            eval_swap_candidates_filtered.append((pointer, proxy_score))
                            seen_proxy_scores.add(round(proxy_score, 2))  # 保留2位小数
                        else:
                            # 检查proxy_score是否与已有候选差异足够大（至少0.01）
                            proxy_rounded = round(proxy_score, 2)
                            if all(abs(proxy_rounded - ps) >= 0.01 for ps in seen_proxy_scores):
                                eval_swap_candidates_filtered.append((pointer, proxy_score))
                                seen_proxy_scores.add(proxy_rounded)
                            elif len(eval_swap_candidates_filtered) < eval_E:
                                # 如果差异不够大，但仍然添加（确保有足够的候选）
                                eval_swap_candidates_filtered.append((pointer, proxy_score))
                                seen_proxy_scores.add(proxy_rounded)
                    if len(eval_swap_candidates_filtered) >= eval_E:
                        break
                
                # 如果过滤后没有候选，尝试使用更多候选（放宽限制）
                # 这是因为swap生成的组合可能都在seen_pointers中
                if not eval_swap_candidates_filtered and len(converted_swap_candidates) > 0:
                    # 尝试使用更多候选（最多eval_E * 3个）
                    for pointer, proxy_score in converted_swap_candidates[:eval_E * 3]:
                        pointer_tuple = tuple(sorted(pointer))
                        if pointer_tuple not in seen_pointers:
                            eval_swap_candidates_filtered.append((pointer, proxy_score))
                        if len(eval_swap_candidates_filtered) >= eval_E:
                            break
                
                # 如果仍然没有候选，说明所有swap生成的组合都在seen_pointers中
                # 这种情况下，我们仍然尝试评测top候选（即使它们在seen_pointers中）
                # 因为swap生成的组合可能具有不同的特性，值得重新评测
                if not eval_swap_candidates_filtered and len(converted_swap_candidates) > 0:
                    # 使用top eval_E个候选，即使它们在seen_pointers中
                    eval_swap_candidates_filtered = converted_swap_candidates[:eval_E]
                    # 注意：这种情况下，我们仍然会评测，但可能会重复评测
                    # 不过，由于swap生成的组合可能具有不同的特性，这是可以接受的
                # 评测（Level-3最多评测max_eval_level3次）
                # 确保eval_swap_candidates_filtered不为空
                if eval_swap_candidates_filtered:
                    level3_eval_count = 0

                    for idx, (pointer, proxy_score) in enumerate(eval_swap_candidates_filtered):
                        if stats["eval_count"] >= max_eval_budget:
                            break
                        
                        if level3_eval_count >= max_eval_level3:
                            break
                        
                        # Level-3策略允许超过max_candidates限制（最多额外30个）
                        # 因为这是兜底策略，需要尽可能找到正样本
                        # 注意：这里应该允许Level-3策略评测，即使已经达到max_candidates限制
                        # 因为Level-3是最后的兜底策略，应该给它更多机会
                        soft_limit = max_candidates + 30  # 允许Level-3策略额外评测30个候选（从15增加到30）
                        current_total = len(candidates) + len(stats["new_candidates"])

                        if current_total >= soft_limit:

                            break
                        
                        try:
                            candidate_dict = evaluate_pointer_candidate(
                                vqa_interface=vqa_interface,
                                query_item=query_item,
                                candidate_pool=candidate_pool,
                                pointer=list(pointer),
                                rl_data=rl_data,
                                query_id=query_id,
                                val_ques_path=val_ques_path,
                                val_ann_path=val_ann_path,
                                train_ques_path=train_ques_path,
                                train_ann_path=train_ann_path,
                                vqa_train_cache=vqa_train_cache,
                                vqa_val_cache=vqa_val_cache,
                                generation_kwargs=generation_kwargs,
                                candidate_indices=candidate_indices
                            )
                            
                            # 添加元信息（按照方案要求）
                            max_round_so_far = max([c.get("aug_round", 0) for c in stats["new_candidates"]], default=0)
                            candidate_dict["aug_type"] = "all0"
                            candidate_dict["aug_round"] = max_round_so_far + 1  # Level-3的round
                            candidate_dict["gen_method"] = "aug_swap"  # 方案推荐格式
                            candidate_dict["proxy_score"] = float(proxy_score)  # 添加proxy_score字段

                            stats["new_candidates"].append(candidate_dict)
                            stats["eval_count"] += 1
                            level3_eval_count += 1
                            seen_pointers.add(tuple(sorted(pointer)))

                            # 方案要求：找到正样本立即停止
                            if candidate_dict["vqa_acc_score"] > 0.0:
                                stats["success"] = True
                                stats["rounds"] = max_round_so_far + 1
                                return stats["new_candidates"], stats
                        
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            continue
    
    # ========== 过滤低质量候选：如果补全失败，不添加任何候选 ==========
    # 如果All-Zero query补全失败（没有找到正样本），不添加任何增强候选
    # 这样可以避免负样本稀释正样本比例
    if not stats["success"]:
        # 检查所有生成的候选是否都是0分
        all_zero_scores = all(c.get("vqa_acc_score", 0.0) == 0.0 for c in stats["new_candidates"])
        if all_zero_scores:
            # 补全失败，不添加任何候选（避免负样本稀释）
            stats["new_candidates"] = []
            stats["filtered"] = True
            stats["filtered_reason"] = "no_positive_samples_found"
    
    return stats["new_candidates"], stats


def augment_all1_query(
    query_id: str,
    query_item: Dict,
    candidates: List[Dict],
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    sft_model: Optional[torch.nn.Module],
    vqa_interface,
    candidate_pool: Optional[List[Dict]],
    rl_data: Optional[Dict],
    val_ques_path: Optional[str],
    val_ann_path: Optional[str],
    train_ques_path: Optional[str],
    train_ann_path: Optional[str],
    vqa_train_cache: Optional[VQA],
    vqa_val_cache: Optional[VQA],
    seen_pointers: Set[Tuple[int, ...]],
    max_candidates: int = 24,
    max_eval_budget: int = 8,
    generation_kwargs: Optional[Dict] = None,
    candidate_indices: Optional[List[int]] = None  # 用于索引转换
) -> Tuple[List[Dict], Dict]:
    """
    补全All-One query（全1.0 → 补出0/0.6）
    
    Returns:
        (new_candidates, stats)
    """
    stats = {
        "aug_type": "all1",
        "rounds": 0,
        "eval_count": 0,
        "success": False,
        "new_candidates": []
    }
    
    # 计算相似度排序
    similarities, ranked_indices = compute_similarity_ranks(query_emb, cand_emb)
    
    # ========== Level-1：Bottom-B和Mix策略（多轮次，类似All-Zero）==========
    # 【改进】大幅增加搜索范围和评测数量，提高正样本生成率
    B_schedule = [200, 400, 600, 800, 1000]  # 扩大B值范围
    E_per_round = 12  # 增加每轮评测数量（从4增加到12）
    
    for round_idx, B in enumerate(B_schedule):
        if stats["eval_count"] >= max_eval_budget:
            break
        
        if len(candidates) + len(stats["new_candidates"]) >= max_candidates:
            break
        
        # 检查是否已成功
        current_scores = [c.get('vqa_acc_score', 1.0) for c in candidates + stats["new_candidates"]]
        if any(score < 1.0 for score in current_scores):
            stats["success"] = True
            stats["rounds"] = round_idx + 1
            break
        
        # 策略1：Bottom-B pair
        proposals = propose_bottom_pairs(
            ranked_indices=ranked_indices,
            B=B,
            num_pairs=E_per_round * 3,  # 生成更多候选
            shot_num=2,
            seen=seen_pointers
        )
        
        if not proposals:
            continue
        
        # 按proxy排序（在转换索引之前，因为similarities是基于pool索引的）
        proposals = rank_proposals_by_proxy(proposals, similarities, 'all1')
        
        # 如果提供了candidate_indices，需要将pool索引转换为dataset索引（在排序之后）
        if candidate_indices is not None:
            converted_proposals = []
            for p in proposals:
                try:
                    # 确保所有索引都在有效范围内
                    if all(0 <= idx < len(candidate_indices) for idx in p):
                        converted_proposals.append(tuple(candidate_indices[idx] for idx in p))
                    else:
                        # 跳过无效的索引
                        invalid_indices = [idx for idx in p if not (0 <= idx < len(candidate_indices))]
                        print(f"Warning: 跳过无效的proposal {p}，索引 {invalid_indices} 超出范围 [0, {len(candidate_indices)})")
                except (IndexError, TypeError) as e:
                    print(f"Warning: 转换proposal {p} 时出错: {e}")
                    continue
            proposals = converted_proposals
        
        eval_proposals = proposals[:E_per_round]
        
        for pointer in eval_proposals:
            if stats["eval_count"] >= max_eval_budget:
                break
            
            try:
                candidate_dict = evaluate_pointer_candidate(
                    vqa_interface=vqa_interface,
                    query_item=query_item,
                    candidate_pool=candidate_pool,
                    pointer=list(pointer),
                    rl_data=rl_data,
                    query_id=query_id,
                    val_ques_path=val_ques_path,
                    val_ann_path=val_ann_path,
                    train_ques_path=train_ques_path,
                    train_ann_path=train_ann_path,
                    vqa_train_cache=vqa_train_cache,
                    vqa_val_cache=vqa_val_cache,
                    generation_kwargs=generation_kwargs,
                    candidate_indices=candidate_indices  # 用于索引转换
                )
                
                candidate_dict["aug_type"] = "all1"
                candidate_dict["aug_round"] = round_idx + 1
                candidate_dict["gen_method"] = f"aug_neg_bottom_B{B}"
                
                stats["new_candidates"].append(candidate_dict)
                stats["eval_count"] += 1
                seen_pointers.add(tuple(sorted(pointer)))
                
                # 检查是否成功
                if candidate_dict["vqa_acc_score"] < 1.0:
                    stats["success"] = True
                    stats["rounds"] = round_idx + 1
                    return stats["new_candidates"], stats
            
            except Exception as e:
                print(f"Warning: Evaluation failed for Level-1 pointer {pointer}: {e}")
                continue
        
        # 策略2：Mix pair（如果Bottom-B还没成功）
        if not stats["success"] and stats["eval_count"] < max_eval_budget:
            L = min(200, len(ranked_indices) // 2)
            mix_proposals = propose_mix_pairs(
                ranked_indices=ranked_indices,
                L=L,
                B=B,
                num_pairs=E_per_round * 2,
                shot_num=2,
                seen=seen_pointers
            )
            
            if mix_proposals:
                # 按proxy排序
                mix_proposals = rank_proposals_by_proxy(mix_proposals, similarities, 'all1')
                
                # 转换索引
                if candidate_indices is not None:
                    converted_mix_proposals = []
                    for p in mix_proposals:
                        try:
                            if all(0 <= idx < len(candidate_indices) for idx in p):
                                converted_mix_proposals.append(tuple(candidate_indices[idx] for idx in p))
                        except (IndexError, TypeError):
                            continue
                    mix_proposals = converted_mix_proposals
                
                eval_mix_proposals = mix_proposals[:E_per_round]
                
                for pointer in eval_mix_proposals:
                    if stats["eval_count"] >= max_eval_budget:
                        break
                    
                    try:
                        candidate_dict = evaluate_pointer_candidate(
                            vqa_interface=vqa_interface,
                            query_item=query_item,
                            candidate_pool=candidate_pool,
                            pointer=list(pointer),
                            rl_data=rl_data,
                            query_id=query_id,
                            val_ques_path=val_ques_path,
                            val_ann_path=val_ann_path,
                            train_ques_path=train_ques_path,
                            train_ann_path=train_ann_path,
                            vqa_train_cache=vqa_train_cache,
                            vqa_val_cache=vqa_val_cache,
                            generation_kwargs=generation_kwargs,
                            candidate_indices=candidate_indices
                        )
                        
                        candidate_dict["aug_type"] = "all1"
                        candidate_dict["aug_round"] = round_idx + 1
                        candidate_dict["gen_method"] = f"aug_neg_mix_L{L}_B{B}"
                        
                        stats["new_candidates"].append(candidate_dict)
                        stats["eval_count"] += 1
                        seen_pointers.add(tuple(sorted(pointer)))
                        
                        if candidate_dict["vqa_acc_score"] < 1.0:
                            stats["success"] = True
                            stats["rounds"] = round_idx + 1
                            return stats["new_candidates"], stats
                    
                    except Exception as e:
                        print(f"Warning: Evaluation failed for Level-1 mix pointer {pointer}: {e}")
                        continue
    
    # ========== Level-2：SFT模型生成候选（作为补强策略）==========
    # 【改进】添加Level-2策略，使用SFT模型生成候选
    if not stats["success"] and stats["eval_count"] < max_eval_budget and sft_model is not None:
        device = next(sft_model.parameters()).device
        query_emb_device = query_emb.to(device)
        cand_emb_device = cand_emb.to(device)
        
        exclude_indices = []
        if candidate_indices is not None:
            try:
                query_idx_in_pool = candidate_indices.index(int(query_id))
                exclude_indices = [query_idx_in_pool]
            except (ValueError, TypeError):
                pass
        
        # Beam Search生成候选
        try:
            beam_results = beam_search_pointer(
                model=sft_model,
                query_emb=query_emb_device,
                cand_emb=cand_emb_device,
                num_beams=20,  # 增加beam size
                shot_num=2,
                exclude_indices=exclude_indices if exclude_indices else None
            )
            
            sft_proposals = []
            for beam_result in beam_results:
                pointer = beam_result.get("pointer", [])
                if len(pointer) == 2:
                    if candidate_indices is not None:
                        try:
                            if all(0 <= idx < len(candidate_indices) for idx in pointer):
                                converted_pointer = tuple(candidate_indices[idx] for idx in pointer)
                                if converted_pointer not in seen_pointers:
                                    sft_proposals.append((converted_pointer, "beam", beam_result.get("logprob", 0.0)))
                        except (IndexError, TypeError):
                            continue
                    else:
                        if tuple(pointer) not in seen_pointers:
                            sft_proposals.append((tuple(pointer), "beam", beam_result.get("logprob", 0.0)))
            
            sft_proposals.sort(key=lambda x: x[2], reverse=True)
            
            # 评测beam search候选
            for pointer_tuple, gen_method, logprob in sft_proposals[:25]:
                if stats["eval_count"] >= max_eval_budget:
                    break
                
                try:
                    candidate_dict = evaluate_pointer_candidate(
                        vqa_interface=vqa_interface,
                        query_item=query_item,
                        candidate_pool=candidate_pool,
                        pointer=list(pointer_tuple),
                        rl_data=rl_data,
                        query_id=query_id,
                        val_ques_path=val_ques_path,
                        val_ann_path=val_ann_path,
                        train_ques_path=train_ques_path,
                        train_ann_path=train_ann_path,
                        vqa_train_cache=vqa_train_cache,
                        vqa_val_cache=vqa_val_cache,
                        generation_kwargs=generation_kwargs,
                        candidate_indices=candidate_indices
                    )
                    
                    candidate_dict["aug_type"] = "all1"
                    max_level1_round = max([c.get("aug_round", 0) for c in stats["new_candidates"] if c.get("gen_method", "").startswith("aug_neg")], default=0)
                    candidate_dict["aug_round"] = max_level1_round + 1
                    candidate_dict["gen_method"] = f"sft_beam_{gen_method}"
                    
                    stats["new_candidates"].append(candidate_dict)
                    stats["eval_count"] += 1
                    seen_pointers.add(tuple(sorted(pointer_tuple)))
                    
                    if candidate_dict["vqa_acc_score"] < 1.0:
                        stats["success"] = True
                        stats["rounds"] = max_level1_round + 1
                        return stats["new_candidates"], stats
                
                except Exception as e:
                    print(f"Warning: Evaluation failed for SFT beam pointer {pointer_tuple}: {e}")
                    continue
        
        except Exception as e:
            print(f"Warning: SFT beam search failed: {e}")
    
    return stats["new_candidates"], stats


def augment_all06_query(
    query_id: str,
    query_item: Dict,
    candidates: List[Dict],
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    sft_model: Optional[torch.nn.Module],  # 【修复】添加sft_model参数
    vqa_interface,
    candidate_pool: Optional[List[Dict]],
    rl_data: Optional[Dict],
    val_ques_path: Optional[str],
    val_ann_path: Optional[str],
    train_ques_path: Optional[str],
    train_ann_path: Optional[str],
    vqa_train_cache: Optional[VQA],
    vqa_val_cache: Optional[VQA],
    seen_pointers: Set[Tuple[int, ...]],
    max_candidates: int = 24,
    max_eval_budget: int = 8,
    generation_kwargs: Optional[Dict] = None,
    candidate_indices: Optional[List[int]] = None  # 用于索引转换
) -> Tuple[List[Dict], Dict]:
    """
    补全All-0.6 query（全0.6 → 补出0或1.0）
    
    Returns:
        (new_candidates, stats)
    """
    stats = {
        "aug_type": "all06",
        "rounds": 0,
        "eval_count": 0,
        "success": False,
        "new_candidates": []
    }
    
    # 计算相似度排序
    similarities, ranked_indices = compute_similarity_ranks(query_emb, cand_emb)
    
    # ========== Level-1：先补0（bottom/mix），再补1（top）（多轮次）==========
    # 【改进】大幅增加搜索范围和评测数量，提高正样本生成率
    B_schedule = [200, 400, 600, 800, 1000]  # 扩大B值范围（用于补0）
    L_schedule = [50, 100, 200, 400, 600]  # 扩大L值范围（用于补1）
    E_per_round = 12  # 增加每轮评测数量（从4增加到12）
    
    # 先补0（生成<0.6的候选）
    for round_idx, B in enumerate(B_schedule):
        if stats["eval_count"] >= max_eval_budget:
            break
        
        if len(candidates) + len(stats["new_candidates"]) >= max_candidates:
            break
        
        # 检查是否已成功（找到<0.6的候选）
        current_scores = [c.get('vqa_acc_score', 0.6) for c in candidates + stats["new_candidates"]]
        if any(score < 0.6 for score in current_scores):
            stats["success"] = True
            stats["rounds"] = round_idx + 1
            break
        
        # 策略1：Bottom-B pair
        proposals = propose_bottom_pairs(
            ranked_indices=ranked_indices,
            B=B,
            num_pairs=E_per_round * 3,
            shot_num=2,
            seen=seen_pointers
        )
        
        if not proposals:
            continue
        
        # 按proxy排序
        proposals = rank_proposals_by_proxy(proposals, similarities, 'all06')
        
        # 转换索引
        if candidate_indices is not None:
            converted_proposals = []
            for p in proposals:
                try:
                    if all(0 <= idx < len(candidate_indices) for idx in p):
                        converted_proposals.append(tuple(candidate_indices[idx] for idx in p))
                except (IndexError, TypeError):
                    continue
            proposals = converted_proposals
        
        eval_proposals = proposals[:E_per_round]
        
        for pointer in eval_proposals:
            if stats["eval_count"] >= max_eval_budget:
                break
            
            try:
                candidate_dict = evaluate_pointer_candidate(
                    vqa_interface=vqa_interface,
                    query_item=query_item,
                    candidate_pool=candidate_pool,
                    pointer=list(pointer),
                    rl_data=rl_data,
                    query_id=query_id,
                    val_ques_path=val_ques_path,
                    val_ann_path=val_ann_path,
                    train_ques_path=train_ques_path,
                    train_ann_path=train_ann_path,
                    vqa_train_cache=vqa_train_cache,
                    vqa_val_cache=vqa_val_cache,
                    generation_kwargs=generation_kwargs,
                    candidate_indices=candidate_indices
                )
                
                candidate_dict["aug_type"] = "all06"
                candidate_dict["aug_round"] = round_idx + 1
                candidate_dict["gen_method"] = f"aug_neg_bottom_B{B}"
                
                stats["new_candidates"].append(candidate_dict)
                stats["eval_count"] += 1
                seen_pointers.add(tuple(sorted(pointer)))
                
                if candidate_dict["vqa_acc_score"] < 0.6:
                    stats["success"] = True
                    stats["rounds"] = round_idx + 1
                    return stats["new_candidates"], stats
            
            except Exception as e:
                print(f"Warning: Evaluation failed for Level-1 pointer {pointer}: {e}")
                continue
    
    # 再补1（生成>0.6的候选）
    for round_idx, L in enumerate(L_schedule):
        if stats["eval_count"] >= max_eval_budget:
            break
        
        if len(candidates) + len(stats["new_candidates"]) >= max_candidates:
            break
        
        # 检查是否已成功（找到>0.6的候选）
        current_scores = [c.get('vqa_acc_score', 0.6) for c in candidates + stats["new_candidates"]]
        if any(score > 0.6 for score in current_scores):
            if not stats["success"]:
                stats["success"] = True
            stats["rounds"] = max(stats.get("rounds", 0), len(B_schedule) + round_idx + 1)
            break
        
        proposals = propose_topL_pairs(
            ranked_indices=ranked_indices,
            L=L,
            num_pairs=None,
            shot_num=2,
            seen=seen_pointers,
            pair_sampling_if_large=50000
        )
        
        if not proposals:
            continue
        
        # 按proxy排序
        proposals = rank_proposals_by_proxy(proposals, similarities, 'all06')
        
        # 转换索引
        if candidate_indices is not None:
            converted_proposals = []
            for p in proposals:
                try:
                    if all(0 <= idx < len(candidate_indices) for idx in p):
                        converted_proposals.append(tuple(candidate_indices[idx] for idx in p))
                except (IndexError, TypeError):
                    continue
            proposals = converted_proposals
        
        eval_proposals = proposals[:E_per_round]
        
        for pointer in eval_proposals:
            if stats["eval_count"] >= max_eval_budget:
                break
            
            try:
                candidate_dict = evaluate_pointer_candidate(
                    vqa_interface=vqa_interface,
                    query_item=query_item,
                    candidate_pool=candidate_pool,
                    pointer=list(pointer),
                    rl_data=rl_data,
                    query_id=query_id,
                    val_ques_path=val_ques_path,
                    val_ann_path=val_ann_path,
                    train_ques_path=train_ques_path,
                    train_ann_path=train_ann_path,
                    vqa_train_cache=vqa_train_cache,
                    vqa_val_cache=vqa_val_cache,
                    generation_kwargs=generation_kwargs,
                    candidate_indices=candidate_indices
                )
                
                candidate_dict["aug_type"] = "all06"
                candidate_dict["aug_round"] = len(B_schedule) + round_idx + 1
                candidate_dict["gen_method"] = f"aug_pos_topL_L{L}"
                
                stats["new_candidates"].append(candidate_dict)
                stats["eval_count"] += 1
                seen_pointers.add(tuple(sorted(pointer)))
                
                if candidate_dict["vqa_acc_score"] > 0.6:
                    if not stats["success"]:
                        stats["success"] = True
                    stats["rounds"] = len(B_schedule) + round_idx + 1
                    return stats["new_candidates"], stats
            
            except Exception as e:
                print(f"Warning: Evaluation failed for Level-1 pointer {pointer}: {e}")
                continue
    
    # ========== Level-2：SFT模型生成候选（作为补强策略）==========
    # 【改进】添加Level-2策略，使用SFT模型生成候选
    if not stats["success"] and stats["eval_count"] < max_eval_budget and sft_model is not None:
        device = next(sft_model.parameters()).device
        query_emb_device = query_emb.to(device)
        cand_emb_device = cand_emb.to(device)
        
        exclude_indices = []
        if candidate_indices is not None:
            try:
                query_idx_in_pool = candidate_indices.index(int(query_id))
                exclude_indices = [query_idx_in_pool]
            except (ValueError, TypeError):
                pass
        
        # Beam Search生成候选
        try:
            beam_results = beam_search_pointer(
                model=sft_model,
                query_emb=query_emb_device,
                cand_emb=cand_emb_device,
                num_beams=20,  # 增加beam size
                shot_num=2,
                exclude_indices=exclude_indices if exclude_indices else None
            )
            
            sft_proposals = []
            for beam_result in beam_results:
                pointer = beam_result.get("pointer", [])
                if len(pointer) == 2:
                    if candidate_indices is not None:
                        try:
                            if all(0 <= idx < len(candidate_indices) for idx in pointer):
                                converted_pointer = tuple(candidate_indices[idx] for idx in pointer)
                                if converted_pointer not in seen_pointers:
                                    sft_proposals.append((converted_pointer, "beam", beam_result.get("logprob", 0.0)))
                        except (IndexError, TypeError):
                            continue
                    else:
                        if tuple(pointer) not in seen_pointers:
                            sft_proposals.append((tuple(pointer), "beam", beam_result.get("logprob", 0.0)))
            
            sft_proposals.sort(key=lambda x: x[2], reverse=True)
            
            # 评测beam search候选
            for pointer_tuple, gen_method, logprob in sft_proposals[:25]:
                if stats["eval_count"] >= max_eval_budget:
                    break
                
                try:
                    candidate_dict = evaluate_pointer_candidate(
                        vqa_interface=vqa_interface,
                        query_item=query_item,
                        candidate_pool=candidate_pool,
                        pointer=list(pointer_tuple),
                        rl_data=rl_data,
                        query_id=query_id,
                        val_ques_path=val_ques_path,
                        val_ann_path=val_ann_path,
                        train_ques_path=train_ques_path,
                        train_ann_path=train_ann_path,
                        vqa_train_cache=vqa_train_cache,
                        vqa_val_cache=vqa_val_cache,
                        generation_kwargs=generation_kwargs,
                        candidate_indices=candidate_indices
                    )
                    
                    candidate_dict["aug_type"] = "all06"
                    max_level1_round = max([c.get("aug_round", 0) for c in stats["new_candidates"]], default=0)
                    candidate_dict["aug_round"] = max_level1_round + 1
                    candidate_dict["gen_method"] = f"sft_beam_{gen_method}"
                    
                    stats["new_candidates"].append(candidate_dict)
                    stats["eval_count"] += 1
                    seen_pointers.add(tuple(sorted(pointer_tuple)))
                    
                    if candidate_dict["vqa_acc_score"] < 0.6 or candidate_dict["vqa_acc_score"] > 0.6:
                        stats["success"] = True
                        stats["rounds"] = max_level1_round + 1
                        return stats["new_candidates"], stats
                
                except Exception as e:
                    print(f"Warning: Evaluation failed for SFT beam pointer {pointer_tuple}: {e}")
                    continue
        
        except Exception as e:
            print(f"Warning: SFT beam search failed: {e}")
    
    return stats["new_candidates"], stats


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="RL数据多样性补全脚本")
    
    # 输入输出
    parser.add_argument("--input_rl_data", type=str, required=True,
                       help="输入的RL数据JSON文件路径")
    parser.add_argument("--query_embeddings", type=str, required=True,
                       help="Query embeddings文件路径")
    parser.add_argument("--candidate_embeddings", type=str, required=True,
                       help="Candidate embeddings文件路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出的补全后RL数据JSON文件路径")
    parser.add_argument("--report_path", type=str, default=None,
                       help="统计报告JSON文件路径（默认：output_path.replace('.json', '_report.json')）")
    
    # 模型配置
    parser.add_argument("--sft_ckpt", type=str, default=None,
                       help="SFT checkpoint路径（可选，用于SFT selector proposal）")
    parser.add_argument("--vqa_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="VQA模型名称")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备")
    
    # 数据路径
    parser.add_argument("--val_ques_path", type=str, default=None,
                       help="验证集问题文件路径")
    parser.add_argument("--val_ann_path", type=str, default=None,
                       help="验证集标注文件路径")
    parser.add_argument("--train_ques_path", type=str, default=None,
                       help="训练集问题文件路径")
    parser.add_argument("--train_ann_path", type=str, default=None,
                       help="训练集标注文件路径")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="数据集配置文件路径（如configs/dataset/okvqa.yaml，用于加载candidate_pool）")
    
    # 补全参数
    parser.add_argument("--max_candidates_per_query", type=int, default=24,
                       help="每个query最大候选数")
    parser.add_argument("--max_eval_budget_all0", type=int, default=100,
                       help="All-Zero query最大评测预算")
    parser.add_argument("--max_eval_budget_all1", type=int, default=8,
                       help="All-One query最大评测预算")
    parser.add_argument("--max_eval_budget_all06", type=int, default=8,
                       help="All-0.6 query最大评测预算")
    
    # 其他
    parser.add_argument("--shot_num", type=int, default=2,
                       help="Shot数量")
    parser.add_argument("--max_queries", type=int, default=-1,
                       help="最大处理query数量（-1表示全部）")
    parser.add_argument("--add_timestamp", action="store_true", default=False,
                       help="在输出文件名中添加时间戳，避免覆盖之前的数据")
    parser.add_argument("--use_full_candidate_pool", action="store_true", default=False,
                       help="对all0 query使用完整候选池（而不是限制在现有候选中），用于更激进的探索")
    parser.add_argument("--full_pool_size", type=int, default=500,
                       help="使用完整候选池时的大小（默认500，从相似度最高的候选中选取）")
    
    args = parser.parse_args()
    
    # 如果启用时间戳，在输出文件名中添加时间戳
    if args.add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 在文件名（不含扩展名）后添加时间戳
        output_path_base = os.path.splitext(args.output_path)[0]
        output_path_ext = os.path.splitext(args.output_path)[1]
        args.output_path = f"{output_path_base}_{timestamp}{output_path_ext}"
        
        # 如果report_path已设置，也添加时间戳
        if args.report_path:
            report_path_base = os.path.splitext(args.report_path)[0]
            report_path_ext = os.path.splitext(args.report_path)[1]
            args.report_path = f"{report_path_base}_{timestamp}{report_path_ext}"
    
    # 设置report_path
    if args.report_path is None:
        args.report_path = args.output_path.replace('.json', '_report.json')
    
    print("=" * 80)
    print("RL 数据多样性补全脚本")
    print("=" * 80)
    print(f"输入数据: {args.input_rl_data}")
    print(f"输出数据: {args.output_path}")
    print(f"报告文件: {args.report_path}")
    if args.add_timestamp:
        print(f"✓ 已启用时间戳，输出文件将不会覆盖之前的数据")
    print()
    
    # 加载数据
    print("加载数据...")
    rl_data = load_rl_data(args.input_rl_data)
    query_emb, cand_emb = load_embeddings(args.query_embeddings, args.candidate_embeddings)
    
    print(f"  - Query数量: {len(rl_data)}")
    print(f"  - Query embeddings: {query_emb.shape}")
    print(f"  - Candidate embeddings: {cand_emb.shape}")
    print()
    
    # 加载VQA模型
    print("加载VQA模型...")
    device_obj = torch.device(args.device)
    vqa_interface = load_vqa_model(args.vqa_model, device_obj)
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 10
    }
    
    # 加载VQA评测缓存
    vqa_train_cache = None
    vqa_val_cache = None
    if args.train_ques_path and args.train_ann_path:
        try:
            vqa_train_cache = VQA(args.train_ann_path, args.train_ques_path)
            print(f"  - 训练集VQA缓存已加载")
        except Exception as e:
            print(f"  - Warning: 训练集VQA缓存加载失败: {e}")
    
    if args.val_ques_path and args.val_ann_path:
        try:
            vqa_val_cache = VQA(args.val_ann_path, args.val_ques_path)
            print(f"  - 验证集VQA缓存已加载")
        except Exception as e:
            print(f"  - Warning: 验证集VQA缓存加载失败: {e}")
    
    # 加载SFT模型（如果需要）
    sft_model = None
    if args.sft_ckpt:
        print("加载SFT模型...")
        device = torch.device(args.device)
        sft_model = load_sft_model(args.sft_ckpt, device)
        sft_model.eval()
        print(f"  - SFT模型已加载")
    
    print()
    
    # 加载数据集（用于获取candidate_pool）
    # 从RL数据中推断candidate_indices
    print("构建candidate_pool...")
    
    # 确保环境变量已设置（如果未设置，尝试自动检测）
    if not os.environ.get("OKVQA_PATH"):
        possible_paths = [
            "/mnt/share/yiyun/datasets/okvqa",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "datasets/okvqa"),
            os.path.expanduser("~/datasets/okvqa")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["OKVQA_PATH"] = path
                print(f"  - 自动设置 OKVQA_PATH={path}")
                break
    
    if not os.environ.get("COCO_PATH"):
        possible_paths = [
            "/mnt/share/yiyun/datasets/mscoco",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "datasets/mscoco"),
            os.path.expanduser("~/datasets/mscoco")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["COCO_PATH"] = path
                print(f"  - 自动设置 COCO_PATH={path}")
                break
    
    # 【修复】方案1：限制候选池大小为64，每个query只使用自己的候选索引
    # 不再从所有query中提取候选索引，而是在处理每个query时单独提取
    print(f"  - 将使用方案1：每个query只使用自己的候选索引，限制候选池大小为64")
    
    # 加载dataset（用于构建candidate_pool）
    # 需要从config文件加载dataset配置
    dataset = None
    if args.dataset_config:
        # 如果提供了dataset配置文件，加载dataset
        from omegaconf import OmegaConf
        try:
            # 先尝试加载，如果环境变量未设置会报错
            dataset_cfg = OmegaConf.load(args.dataset_config)
            # 需要添加task字段才能使用load_ds
            # 创建一个完整的cfg对象
            cfg = OmegaConf.create({
                "task": {
                    "task_name": "vqa"  # 默认VQA任务
                },
                "dataset": dataset_cfg
            })
            dataset = load_ds(cfg, split="train")
            print(f"  - Dataset已加载，可用于构建candidate_pool")
        except Exception as e:
            print(f"  - Warning: 加载dataset失败: {e}")
            print(f"  - 可能原因：环境变量OKVQA_PATH或COCO_PATH未设置")
            print(f"  - 将跳过candidate_pool加载，补全功能可能无法正常工作")
            dataset = None
    else:
        print(f"  - Warning: 未提供dataset_config，candidate_pool将无法加载")
        print(f"  - 请使用 --dataset_config 参数指定dataset配置文件（如configs/dataset/okvqa_local.yaml）")
        print(f"  - 注意：没有candidate_pool时，补全功能可能无法正常工作")
    
    print()
    
    # 统计信息
    stats_before = {
        "total_queries": len(rl_data),
        "unique_reward_count": Counter(),
        "aug_type_count": Counter(),
        "all0_count": 0,
        "all1_count": 0,
        "all06_count": 0,
        "diverse_count": 0
    }
    
    stats_after = {
        "total_queries": 0,
        "unique_reward_count": Counter(),
        "aug_type_count": Counter(),
        "augmented_queries": 0,
        "all0_success": 0,
        "all1_success": 0,
        "all06_success": 0,
        "total_new_candidates": 0,
        "total_eval_count": 0
    }
    
    # 处理每个query
    print("开始补全...")
    augmented_data = {}
    query_ids = list(rl_data.keys())
    
    if args.max_queries > 0:
        query_ids = query_ids[:args.max_queries]
    
    for query_id in tqdm(query_ids, desc="处理queries"):
        query_data = rl_data[query_id]
        # 支持两种格式：直接是query_item，或者有"query"字段
        if "query" in query_data:
            query_item = query_data["query"]
            candidates = query_data.get("pointer_candidates", [])
        else:
            query_item = query_data
            candidates = query_item.get("pointer_candidates", [])
        
        # 统计补全前
        unique_scores = set([c.get('vqa_acc_score', 0.0) for c in candidates])
        stats_before["unique_reward_count"][len(unique_scores)] += 1
        
        # 分类
        aug_type, unique_scores = classify_query(candidates)
        stats_before["aug_type_count"][aug_type] += 1
        
        if aug_type == 'all0':
            stats_before["all0_count"] += 1
        elif aug_type == 'all1':
            stats_before["all1_count"] += 1
        elif aug_type == 'all06':
            stats_before["all06_count"] += 1
        else:
            stats_before["diverse_count"] += 1
        
        # 如果已经是diverse，直接复制（保留完整的query结构）
        if aug_type == 'diverse':
            # 确保保存完整的query结构
            if "query" in query_data:
                augmented_data[query_id] = query_data.copy()
            else:
                augmented_data[query_id] = query_item.copy()
            continue
        
        # 获取query embedding
        try:
            if isinstance(query_id, str) and query_id.isdigit():
                query_idx = int(query_id)
            else:
                # 使用hash确保索引在有效范围内
                query_idx = hash(str(query_id)) % len(query_emb)
            query_emb_single = query_emb[query_idx]
        except Exception as e:
            print(f"Warning: 无法获取query {query_id}的embedding: {e}")
            # 即使无法获取embedding，也要保存原始数据
            if "query" in query_data:
                augmented_data[query_id] = query_data.copy()
            else:
                augmented_data[query_id] = query_item.copy()
            continue
        
        # 【修复】方案1：为每个query提取其自己的候选索引
        # 对于all0 query且启用use_full_candidate_pool，使用相似度最高的64个候选
        # 否则从该query的pointer_candidates中提取
        
        if aug_type == 'all0' and args.use_full_candidate_pool:
            # 对all0 query使用完整候选池：计算与query最相似的64个候选
            query_emb_norm = F.normalize(query_emb_single.unsqueeze(0), p=2, dim=1)  # [1, d]
            cand_emb_norm = F.normalize(cand_emb, p=2, dim=1)  # [N, d]
            similarities = torch.mm(query_emb_norm, cand_emb_norm.t()).squeeze(0)  # [N]
            
            # 排除query自身
            query_idx_int = int(query_id) if isinstance(query_id, str) and query_id.isdigit() else None
            if query_idx_int is not None and query_idx_int < len(similarities):
                similarities[query_idx_int] = -1.0  # 排除自身
            
            # 选择top-64最相似的候选
            _, topk_indices = torch.topk(similarities, k=min(64, len(similarities)))
            query_candidate_indices = topk_indices.tolist()
        else:
            # 原有逻辑：从pointer_candidates中提取候选索引
            query_candidate_indices_set = set()
            for c in candidates:
                pointer = c.get("pointer", [])
                for idx in pointer:
                    query_candidate_indices_set.add(idx)
            
            query_candidate_indices = sorted(list(query_candidate_indices_set))
            
            # 限制候选池大小为64（如果超过64个，只使用前64个）
            if len(query_candidate_indices) > 64:
                print(f"  - Warning: Query {query_id}的候选索引数({len(query_candidate_indices)})超过64，将限制为64")
                query_candidate_indices = query_candidate_indices[:64]
        
        # 构建该query的candidate_pool
        candidate_pool_for_query = None
        if dataset is not None and len(query_candidate_indices) > 0:
            try:
                candidate_pool_for_query = [dataset[idx] for idx in query_candidate_indices]
            except Exception as e:
                print(f"  - Warning: 构建query {query_id}的candidate_pool失败: {e}")
                candidate_pool_for_query = None
        
        # 获取candidate embeddings（只使用该query的候选索引）
        if len(query_candidate_indices) > 0:
            # cand_emb的索引就是dataset的索引，所以直接索引
            valid_cand_emb_for_query = cand_emb[query_candidate_indices]  # [len(query_candidate_indices), d]
            # ranked_indices将直接对应candidate_pool的索引（0, 1, 2, ...）
            # 需要转换为dataset索引时，使用query_candidate_indices[ranked_idx]
            candidate_indices_for_query = query_candidate_indices
        else:
            # 如果没有候选索引，使用全部embeddings（不应该发生）
            print(f"  - Warning: Query {query_id}没有候选索引，使用全部embeddings")
            valid_cand_emb_for_query = cand_emb
            candidate_indices_for_query = None
        
        # 获取已见过的pointers
        seen_pointers = set()
        for c in candidates:
            pointer = tuple(sorted(c.get("pointer", [])))
            if pointer:
                seen_pointers.add(pointer)
        
        # 根据类型补全
        new_candidates = []
        aug_stats = {}
        
        if aug_type == 'all0':
            new_candidates, aug_stats = augment_all0_query(
                query_id=query_id,
                query_item=query_item,
                candidates=candidates,
                query_emb=query_emb_single,
                cand_emb=valid_cand_emb_for_query,
                sft_model=sft_model,
                vqa_interface=vqa_interface,
                candidate_pool=candidate_pool_for_query,  # 使用该query的candidate_pool
                rl_data=rl_data,
                val_ques_path=args.val_ques_path,
                val_ann_path=args.val_ann_path,
                train_ques_path=args.train_ques_path,
                train_ann_path=args.train_ann_path,
                vqa_train_cache=vqa_train_cache,
                vqa_val_cache=vqa_val_cache,
                seen_pointers=seen_pointers,
                max_candidates=args.max_candidates_per_query,
                max_eval_budget=args.max_eval_budget_all0,
                generation_kwargs=generation_kwargs,
                candidate_indices=candidate_indices_for_query  # 使用该query的candidate_indices
            )
            if aug_stats.get("success"):
                stats_after["all0_success"] += 1
        
        elif aug_type == 'all1':
            new_candidates, aug_stats = augment_all1_query(
                query_id=query_id,
                query_item=query_item,
                candidates=candidates,
                query_emb=query_emb_single,
                cand_emb=valid_cand_emb_for_query,
                sft_model=sft_model,
                vqa_interface=vqa_interface,
                candidate_pool=candidate_pool_for_query,  # 使用该query的candidate_pool
                rl_data=rl_data,
                val_ques_path=args.val_ques_path,
                val_ann_path=args.val_ann_path,
                train_ques_path=args.train_ques_path,
                train_ann_path=args.train_ann_path,
                vqa_train_cache=vqa_train_cache,
                vqa_val_cache=vqa_val_cache,
                seen_pointers=seen_pointers,
                max_candidates=args.max_candidates_per_query,
                max_eval_budget=args.max_eval_budget_all1,
                generation_kwargs=generation_kwargs,
                candidate_indices=candidate_indices_for_query  # 使用该query的candidate_indices
            )
            if aug_stats.get("success"):
                stats_after["all1_success"] += 1
        
        elif aug_type == 'all06':
            new_candidates, aug_stats = augment_all06_query(
                query_id=query_id,
                query_item=query_item,
                candidates=candidates,
                query_emb=query_emb_single,
                cand_emb=valid_cand_emb_for_query,
                sft_model=sft_model,  # 【修复】添加sft_model参数
                vqa_interface=vqa_interface,
                candidate_pool=candidate_pool_for_query,  # 使用该query的candidate_pool
                rl_data=rl_data,
                val_ques_path=args.val_ques_path,
                val_ann_path=args.val_ann_path,
                train_ques_path=args.train_ques_path,
                train_ann_path=args.train_ann_path,
                vqa_train_cache=vqa_train_cache,
                vqa_val_cache=vqa_val_cache,
                seen_pointers=seen_pointers,
                max_candidates=args.max_candidates_per_query,
                max_eval_budget=args.max_eval_budget_all06,
                generation_kwargs=generation_kwargs,
                candidate_indices=candidate_indices_for_query  # 使用该query的candidate_indices
            )
            if aug_stats.get("success"):
                stats_after["all06_success"] += 1
        
        # 合并新候选
        if new_candidates:
            # 确保保存完整的query结构
            if "query" in query_data:
                augmented_data[query_id] = query_data.copy()
                augmented_data[query_id]["pointer_candidates"] = candidates + new_candidates
            else:
                augmented_data[query_id] = query_item.copy()
                augmented_data[query_id]["pointer_candidates"] = candidates + new_candidates
            stats_after["augmented_queries"] += 1
            stats_after["total_new_candidates"] += len(new_candidates)
            stats_after["total_eval_count"] += aug_stats.get("eval_count", 0)
        else:
            # 即使没有新候选，也要保存原始数据
            if "query" in query_data:
                augmented_data[query_id] = query_data.copy()
            else:
                augmented_data[query_id] = query_item.copy()
        
        # 统计补全后
        final_candidates = augmented_data[query_id].get("pointer_candidates", [])
        final_unique_scores = set([c.get('vqa_acc_score', 0.0) for c in final_candidates])
        stats_after["unique_reward_count"][len(final_unique_scores)] += 1
    
    stats_after["total_queries"] = len(augmented_data)
    
    # 保存结果
    print()
    print("保存结果...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    print(f"  - 已保存到: {args.output_path}")
    
    # 生成报告
    report = {
        "before": {
            "total_queries": stats_before["total_queries"],
            "unique_reward_count": dict(stats_before["unique_reward_count"]),
            "aug_type_count": dict(stats_before["aug_type_count"]),
            "all0_count": stats_before["all0_count"],
            "all1_count": stats_before["all1_count"],
            "all06_count": stats_before["all06_count"],
            "diverse_count": stats_before["diverse_count"]
        },
        "after": {
            "total_queries": stats_after["total_queries"],
            "unique_reward_count": dict(stats_after["unique_reward_count"]),
            "augmented_queries": stats_after["augmented_queries"],
            "all0_success": stats_after["all0_success"],
            "all1_success": stats_after["all1_success"],
            "all06_success": stats_after["all06_success"],
            "total_new_candidates": stats_after["total_new_candidates"],
            "total_eval_count": stats_after["total_eval_count"]
        },
        "success_rates": {
            "all0": stats_after["all0_success"] / max(stats_before["all0_count"], 1),
            "all1": stats_after["all1_success"] / max(stats_before["all1_count"], 1),
            "all06": stats_after["all06_success"] / max(stats_before["all06_count"], 1)
        }
    }
    
    with open(args.report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  - 报告已保存到: {args.report_path}")
    
    # 打印摘要
    print()
    print("=" * 80)
    print("补全摘要")
    print("=" * 80)
    print(f"补全前:")
    print(f"  - 总query数: {stats_before['total_queries']}")
    print(f"  - All-Zero: {stats_before['all0_count']}")
    print(f"  - All-One: {stats_before['all1_count']}")
    print(f"  - All-0.6: {stats_before['all06_count']}")
    print(f"  - Diverse: {stats_before['diverse_count']}")
    print(f"  - unique=1占比: {stats_before['unique_reward_count'][1] / max(stats_before['total_queries'], 1) * 100:.1f}%")
    print()
    print(f"补全后:")
    print(f"  - 总query数: {stats_after['total_queries']}")
    print(f"  - 补全query数: {stats_after['augmented_queries']}")
    print(f"  - 新增候选数: {stats_after['total_new_candidates']}")
    print(f"  - 总评测次数: {stats_after['total_eval_count']}")
    print(f"  - unique=1占比: {stats_after['unique_reward_count'][1] / max(stats_after['total_queries'], 1) * 100:.1f}%")
    print()
    print(f"成功率:")
    print(f"  - All-Zero: {report['success_rates']['all0'] * 100:.1f}%")
    print(f"  - All-One: {report['success_rates']['all1'] * 100:.1f}%")
    print(f"  - All-0.6: {report['success_rates']['all06'] * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

