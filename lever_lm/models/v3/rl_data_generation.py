"""
RL 数据生成工具：温度采样 + Correctness 计算

按照强化学习.md §3.2 和 §3.3 实现：
- 温度采样：sample_pointer_with_temperature
- 生成候选：generate_pointer_candidates_for_query
- Correctness 计算：evaluate_pointer_candidate

作者: Lever-Plus Team
日期: 2025-12-06
参考: 强化学习.md
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import random


def beam_search_pointer(
    model,
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    num_beams: int = 5,
    shot_num: int = 2,
    exclude_indices: Optional[List[int]] = None  # 需要排除的索引
) -> List[Dict]:
    """
    对 pointer 进行 beam search，生成多个高质量候选
    
    按照强化学习.md的要求，生成3-5个beam候选作为高质量样本
    
    Args:
        model: PointerSelectorV2 或 PointerSelectorV3 模型
        query_emb: [d] query embedding
        cand_emb: [K, d] 候选 embedding
        num_beams: beam 数量（默认5）
        shot_num: 需要选择的样本数量（默认2）
    
    Returns:
        beam_results: List[Dict]，每个元素包含：
        {
            "pointer": [i, j, ...],
            "score": float (累积分数),
            "logprob": float (累积log概率)
        }
    """
    device = query_emb.device
    query_emb = query_emb.unsqueeze(0)  # [1, d]
    cand_emb = cand_emb.unsqueeze(0)    # [1, K, d]
    K = cand_emb.shape[1]
    
    # 处理exclude_indices
    exclude_set = set(exclude_indices) if exclude_indices else set()
    
    model.eval()
    with torch.no_grad():
        # 初始化 beam：每个 beam 是 (pointer_list, cumulative_logprob, cumulative_score)
        # 第一步：获取所有候选的 logits
        logits_step1 = compute_step_logits(
            model=model,
            query_emb=query_emb,
            cand_emb=cand_emb,
            selected_indices=None
        )  # [K]
        
        # Mask掉需要排除的索引（设为负无穷）
        if exclude_set:
            for idx in exclude_set:
                if 0 <= idx < K:
                    logits_step1[idx] = float('-inf')
        
        log_probs_step1 = F.log_softmax(logits_step1, dim=-1)  # [K]
        scores_step1 = F.softmax(logits_step1, dim=-1)  # [K]
        
        # 选择 top-k 作为初始 beam
        topk_logprobs, topk_indices = torch.topk(log_probs_step1, k=min(num_beams, K))
        
        # 初始化 beams
        beams = []
        for i in range(len(topk_indices)):
            idx = topk_indices[i].item()
            if idx in exclude_set:
                continue  # 跳过需要排除的索引
            logprob = topk_logprobs[i].item()
            score = scores_step1[idx].item()
            beams.append({
                "pointer": [idx],
                "logprob": logprob,
                "score": score
            })
        
        # 后续步骤：扩展每个 beam
        for step in range(1, shot_num):
            all_candidates = []
            
            for beam in beams:
                current_pointer = beam["pointer"]
                current_logprob = beam["logprob"]
                current_score = beam["score"]
                
                # 获取下一步的 logits（mask 已选的）
                logits_next = compute_step_logits(
                    model=model,
                    query_emb=query_emb,
                    cand_emb=cand_emb,
                    selected_indices=current_pointer
                )  # [K]
                
                log_probs_next = F.log_softmax(logits_next, dim=-1)  # [K]
                scores_next = F.softmax(logits_next, dim=-1)  # [K]
                
                # Mask掉需要排除的索引
                if exclude_set:
                    for idx in exclude_set:
                        if 0 <= idx < K:
                            logits_next[idx] = float('-inf')
                    log_probs_next = F.log_softmax(logits_next, dim=-1)  # 重新计算
                    scores_next = F.softmax(logits_next, dim=-1)
                
                # 对每个可能的下一个选择，计算累积分数
                for next_idx in range(K):
                    if next_idx in current_pointer:
                        continue  # 跳过已选的
                    if next_idx in exclude_set:
                        continue  # 跳过需要排除的索引
                    
                    new_logprob = current_logprob + log_probs_next[next_idx].item()
                    new_score = current_score + scores_next[next_idx].item()
                    new_pointer = current_pointer + [next_idx]
                    
                    all_candidates.append({
                        "pointer": new_pointer,
                        "logprob": new_logprob,
                        "score": new_score
                    })
            
            # 按 logprob 排序，选择 top-k
            all_candidates.sort(key=lambda x: x["logprob"], reverse=True)
            beams = all_candidates[:num_beams]
        
        # 返回最终的 beam 结果
        return beams


def compute_step_logits(
    model,
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    selected_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    计算单步的 logits（适配 PointerSelectorV2/V3 的 forward 逻辑）
    
    这是对 forward 方法的简化版本，只计算单步 logits，不进行自回归生成
    
    Args:
        model: PointerSelectorV2 或 PointerSelectorV3 模型
        query_emb: [1, d] query embedding
        cand_emb: [1, K, d] 候选 embedding
        selected_indices: 已选择的索引列表（用于 mask）
    
    Returns:
        logits: [K] 每个候选的 logits
    """
    batch_size = query_emb.shape[0]
    device = query_emb.device
    input_dim = query_emb.shape[-1]
    actual_K = cand_emb.shape[1]
    
    # 步骤1：投影到 hidden_dim
    query_reduced = model.input_proj(query_emb)  # [1, hidden_dim]
    cand_reduced = model.input_proj(cand_emb.reshape(-1, input_dim))  # [K, hidden_dim]
    cand_reduced = cand_reduced.reshape(batch_size, actual_K, model.hidden_dim)  # [1, K, hidden_dim]
    
    # 步骤2：多层 Cross-Attention 增强
    query_for_attn = query_reduced.unsqueeze(1)  # [1, 1, hidden_dim]
    
    for layer_idx in range(model.num_layers):
        attn_output, _ = model.cross_attn_layers[layer_idx](
            query=query_for_attn,
            key=cand_reduced,
            value=cand_reduced
        )
        query_for_attn = model.attn_norms[layer_idx](attn_output + query_for_attn)
    
    query_enhanced = query_for_attn.squeeze(1)  # [1, hidden_dim]
    
    # 步骤3：投影层
    query_proj = model.query_proj(query_enhanced)  # [1, hidden_dim]
    cand_proj = model.cand_proj(cand_reduced)      # [1, K, hidden_dim]
    
    # 步骤4：Dropout（推理时应该关闭，但为了兼容性保留）
    if model.training:
        query_proj = model.dropout(query_proj)
        cand_proj = model.dropout(cand_proj)
    
    # 步骤5：L2 归一化
    query_proj = F.normalize(query_proj, p=2, dim=-1)
    cand_proj = F.normalize(cand_proj, p=2, dim=-1)
    
    # 步骤6：计算注意力分数
    scores = torch.matmul(query_proj.unsqueeze(1), cand_proj.transpose(1, 2))  # [1, 1, K]
    temperature = model.temperature.to(device)
    scores = scores.squeeze(1) / temperature  # [1, K]
    
    # 步骤7：应用 mask（屏蔽已选择的候选）
    if selected_indices is not None:
        mask = torch.zeros(batch_size, actual_K, dtype=torch.bool, device=device)
        for idx in selected_indices:
            mask[0, idx] = True
        scores = scores.masked_fill(mask, -100.0)
    
    return scores[0]  # [K]


def sample_pointer_with_temperature(
    model,
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    tau: float,
    exclude_indices: Optional[List[int]] = None  # 需要排除的索引
) -> Tuple[List[int], float]:
    """
    对 pointer 进行两步温度采样
    
    按照强化学习.md §3.2 实现：
    - step1: 从 K 个 candidates 中选第一个 index
    - step2: 在剩余 K-1 个 candidates 中选第二个 index
    - 同时返回整个 pointer 序列的 logprob
    
    Args:
        model: PointerSelectorV2 或 PointerSelectorV3 模型
        query_emb: [d] query embedding
        cand_emb: [K, d] 候选 embedding
        tau: 温度参数
    
    Returns:
        selected: [idx1, idx2] 选择的索引序列
        total_logprob: 整个序列的 log 概率
    """
    device = query_emb.device
    query_emb = query_emb.unsqueeze(0)  # [1, d]
    cand_emb = cand_emb.unsqueeze(0)    # [1, K, d]
    K = cand_emb.shape[1]
    
    # 处理exclude_indices
    exclude_set = set(exclude_indices) if exclude_indices else set()
    
    selected = []
    total_logprob = 0.0
    
    # step 1: 选择第一个 index
    logits_step1 = compute_step_logits(
        model=model,
        query_emb=query_emb,
        cand_emb=cand_emb,
        selected_indices=None
    )  # [K]
    
    # Mask掉需要排除的索引
    if exclude_set:
        for idx in exclude_set:
            if 0 <= idx < K:
                logits_step1[idx] = float('-inf')
    
    # 应用温度缩放
    logits_step1 = logits_step1 / tau
    probs_step1 = F.softmax(logits_step1, dim=-1)  # [K]
    
    # 采样（如果所有概率都是0，则随机选择一个非排除的索引）
    if probs_step1.sum() < 1e-8:
        available_indices = [idx for idx in range(K) if idx not in exclude_set]
        if len(available_indices) == 0:
            raise ValueError("没有可用的索引进行采样")
        import random
        idx1 = random.choice(available_indices)
        total_logprob += torch.log(torch.tensor(1.0 / len(available_indices))).item()
    else:
        idx1 = torch.multinomial(probs_step1, num_samples=1).item()
        total_logprob += torch.log(probs_step1[idx1] + 1e-8).item()
    selected.append(idx1)
    
    # step 2: 选择第二个 index（mask 掉已选的和需要排除的）
    logits_step2 = compute_step_logits(
        model=model,
        query_emb=query_emb,
        cand_emb=cand_emb,
        selected_indices=[idx1]
    )  # [K]
    
    # Mask掉需要排除的索引
    if exclude_set:
        for idx in exclude_set:
            if 0 <= idx < K:
                logits_step2[idx] = float('-inf')
    
    # 应用温度缩放
    logits_step2 = logits_step2 / tau
    
    # Mask 已选 index 和需要排除的索引
    mask = torch.ones(K, dtype=torch.bool, device=device)
    mask[idx1] = False
    if exclude_set:
        for idx in exclude_set:
            if 0 <= idx < K:
                mask[idx] = False
    logits_step2_masked = logits_step2.masked_fill(~mask, -1e9)
    probs_step2 = F.softmax(logits_step2_masked, dim=-1)
    
    # 采样
    if probs_step2.sum() < 1e-8:
        available_indices = [idx for idx in range(K) if idx not in exclude_set and idx not in selected]
        if len(available_indices) == 0:
            raise ValueError("没有可用的索引进行采样")
        import random
        idx2 = random.choice(available_indices)
        total_logprob += torch.log(torch.tensor(1.0 / len(available_indices))).item()
    else:
        idx2 = torch.multinomial(probs_step2, num_samples=1).item()
        total_logprob += torch.log(probs_step2[idx2] + 1e-8).item()
    selected.append(idx2)
    
    return selected, total_logprob


def retrieve_pointer_by_similarity(
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    num_retrievals: int = 5,
    shot_num: int = 2,
    exclude_indices: Optional[List[int]] = None
) -> List[Dict]:
    """
    基于embedding相似度的ICD检索方法
    
    使用余弦相似度选择与query最相似的ICD，提高few-shot学习效果
    
    Args:
        query_emb: [d] query embedding
        cand_emb: [K, d] 候选 embedding
        num_retrievals: 检索的ICD组合数量
        shot_num: 每个组合选择的ICD数量（默认2）
        exclude_indices: 需要排除的索引（例如query_id本身）
    
    Returns:
        candidates: List[Dict]，每个元素包含：
        {
            "pointer": [i, j],
            "gen_method": "retrieval",
            "similarity_score": float,
            "beam_rank": None,
            "beam_score": None,
            "logprob_score": None,
            "temperature": None
        }
    """
    device = query_emb.device
    query_emb = query_emb.unsqueeze(0)  # [1, d]
    cand_emb = cand_emb.unsqueeze(0)    # [1, K, d]
    K = cand_emb.shape[1]
    
    # 处理exclude_indices
    exclude_set = set(exclude_indices) if exclude_indices else set()
    
    # 计算余弦相似度
    # query_emb: [1, d], cand_emb: [1, K, d]
    # 归一化
    query_emb_norm = F.normalize(query_emb, p=2, dim=-1)  # [1, d]
    cand_emb_norm = F.normalize(cand_emb, p=2, dim=-1)    # [1, K, d]
    
    # 计算相似度：[1, K] -> [K]
    similarities_raw = torch.matmul(query_emb_norm, cand_emb_norm.transpose(1, 2))  # [1, K] 或 [1, 1, K]
    # 确保正确squeeze：如果shape是[1, K]，squeeze(0)得到[K]
    # 如果shape是[1, 1, K]，需要squeeze两次
    similarities = similarities_raw
    while similarities.dim() > 1:
        similarities = similarities.squeeze(0)
    # 确保是1D tensor
    if similarities.dim() == 0:
        similarities = similarities.unsqueeze(0)
    
    # 确保K维度正确
    assert similarities.shape[0] == K, f"similarities shape {similarities.shape} != K={K}"
    
    # Mask掉需要排除的索引
    if exclude_set:
        for idx in exclude_set:
            if 0 <= idx < len(similarities):
                similarities[idx] = float('-inf')
    
    candidates = []
    
    # 生成多个ICD组合
    # 为了生成不同的组合，我们使用不同的策略：
    # 1. 第一个组合：选择top-2相似度最高的
    # 2. 后续组合：选择top-k相似度范围内的不同组合
    seen_pointers = set()
    
    for retrieval_idx in range(num_retrievals):
        pointer = []
        remaining_similarities = similarities.clone()  # [K]
        
        # 选择shot_num个ICD
        for step in range(shot_num):
            # 获取可用的索引
            available_indices = [idx for idx in range(len(remaining_similarities)) 
                               if idx not in exclude_set and idx not in pointer]
            
            if len(available_indices) == 0:
                break  # 没有可用索引，退出循环
            
            # 从可用索引中选择相似度最高的
            # 使用列表索引，更安全
            available_similarities_list = [remaining_similarities[idx].item() for idx in available_indices]
            
            # 为了生成不同的组合，在后续迭代中选择不同的top-k
            if retrieval_idx == 0:
                # 第一个组合：选择最高的
                best_local_idx = available_similarities_list.index(max(available_similarities_list))
            else:
                # 后续组合：选择top-k范围内的不同组合
                # 排序并选择第retrieval_idx+1高的（如果可能）
                sorted_indices = sorted(range(len(available_similarities_list)), 
                                       key=lambda i: available_similarities_list[i], 
                                       reverse=True)
                if retrieval_idx < len(sorted_indices):
                    best_local_idx = sorted_indices[retrieval_idx % len(sorted_indices)]
                else:
                    best_local_idx = sorted_indices[0]
            
            top_idx = available_indices[best_local_idx]
            pointer.append(top_idx)
            
            # Mask掉已选的索引
            remaining_similarities[top_idx] = float('-inf')
        
        if len(pointer) == shot_num:
            # 去重：如果这个pointer已经存在，跳过
            pointer_tuple = tuple(sorted(pointer))
            if pointer_tuple not in seen_pointers:
                seen_pointers.add(pointer_tuple)
                # 计算平均相似度分数
                avg_similarity = similarities[pointer].mean().item()
                candidates.append({
                    "pointer": pointer,
                    "gen_method": "retrieval",
                    "similarity_score": avg_similarity,
                    "beam_rank": None,
                    "beam_score": None,
                    "logprob_score": None,
                    "temperature": None,
                })
    
    return candidates


def generate_pointer_candidates_for_query(
    model,
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    num_beams: int = 5,
    temps: Tuple[float, ...] = (1.0, 1.3),
    num_samples_per_temp: int = 2,
    num_random: int = 1,
    num_retrieval: int = 0,  # 新增：retrieval方法的数量
    beam_search_fn: Optional[callable] = None,
    exclude_indices: Optional[List[int]] = None  # 需要排除的索引（例如query_id本身）
) -> List[Dict]:
    """
    生成 pointer 候选序列（beam + 温度采样 + 随机组合）
    
    按照强化学习.md §3.2 实现：
    - Beam search: 3-5 条高质量样本
    - 温度采样: τ=1.0 和 τ=1.3，每个温度 2-3 条
    - 随机组合: 1-2 条（可选）
    
    Args:
        model: PointerSelectorV2 或 PointerSelectorV3 模型（SFT 版本）
        query_emb: [d] query embedding
        cand_emb: [K, d] 候选 embedding
        num_beams: beam search 的 beam 数量
        temps: 温度列表，例如 (1.0, 1.3)
        num_samples_per_temp: 每个温度采样的数量
        num_random: 随机组合的数量
        beam_search_fn: beam search 函数（可选，如果提供则使用）
    
    Returns:
        candidates: List[Dict]，每个元素包含：
        {
            "pointer": [i, j],
            "gen_method": "beam" / "sample" / "random",
            "beam_rank": int or None,
            "beam_score": float or None,
            "logprob_score": float or None,
            "temperature": float or None
        }
    """
    candidates = []
    # 修复 K 维度 bug：使用 shape[-2] 以兼容 [K, d] 和 [B, K, d]
    # 按照 2025-12-13需求.md P0需求1 的要求
    K = cand_emb.shape[-2]
    
    # 处理exclude_indices：转换为set以便快速查找
    exclude_set = set(exclude_indices) if exclude_indices else set()
    
    # 1) Beam search - 使用内置的 beam_search_pointer 函数生成多个高质量候选
    if beam_search_fn is not None:
        beam_results = beam_search_fn(
            model=model,
            query_emb=query_emb,
            cand_emb=cand_emb,
            num_beams=num_beams,
            exclude_indices=exclude_indices
        )
    else:
        # 使用内置的 beam_search_pointer 函数
        beam_results = beam_search_pointer(
            model=model,
            query_emb=query_emb,
            cand_emb=cand_emb,
            num_beams=num_beams,
            shot_num=2,  # 默认选择2个样本
            exclude_indices=exclude_indices
        )
    
    # beam_results: List[{"pointer": [i, j], "score": float, "logprob": float}]
    # 过滤掉包含exclude_indices的pointer
    for rank, br in enumerate(beam_results):
        pointer = br["pointer"]
        # 检查pointer中是否包含需要排除的索引
        if exclude_set and any(idx in exclude_set for idx in pointer):
            continue  # 跳过包含exclude_indices的pointer
        candidates.append({
            "pointer": pointer,
            "gen_method": "beam",
            "beam_rank": rank,
            "beam_score": br.get("score"),
            "logprob_score": br.get("logprob"),
            "temperature": None,
        })
    
    # 2) 温度采样
    model.eval()
    with torch.no_grad():
        for tau in temps:
            for _ in range(num_samples_per_temp):
                pointer, logprob = sample_pointer_with_temperature(
                    model=model,
                    query_emb=query_emb,
                    cand_emb=cand_emb,
                    tau=tau,
                    exclude_indices=exclude_indices
                )
                # 检查pointer中是否包含需要排除的索引
                if exclude_set and any(idx in exclude_set for idx in pointer):
                    continue  # 跳过包含exclude_indices的pointer
                candidates.append({
                    "pointer": pointer,
                    "gen_method": "sample",
                    "beam_rank": None,
                    "beam_score": None,
                    "logprob_score": logprob,
                    "temperature": tau,
                })
    
    # 3) Retrieval-based选择（基于embedding相似度）
    if num_retrieval > 0:
        retrieval_candidates = retrieve_pointer_by_similarity(
            query_emb=query_emb,
            cand_emb=cand_emb,
            num_retrievals=num_retrieval,
            shot_num=2,
            exclude_indices=exclude_indices
        )
        candidates.extend(retrieval_candidates)
    
    # 4) 随机组合（可选）
    # 构建可用的索引列表（排除exclude_indices）
    available_indices = [idx for idx in range(K) if idx not in exclude_set]
    if len(available_indices) < 2:
        print(f"警告：可用索引数量不足（{len(available_indices)}），无法生成随机组合")
    else:
        for _ in range(num_random):
            i, j = random.sample(available_indices, 2)
            pointer = [i, j]
            candidates.append({
                "pointer": pointer,
                "gen_method": "random",
                "beam_rank": None,
                "beam_score": None,
                "logprob_score": None,
                "temperature": None,
            })
    
    # 5) 去重（同一个 pointer 只保留一个，优先 beam > retrieval > sample > random）
    uniq = {}
    priority = {"beam": 4, "retrieval": 3, "sample": 2, "random": 1}
    
    for c in candidates:
        key = tuple(sorted(c["pointer"]))
        if key not in uniq:
            uniq[key] = c
        else:
            # 保留优先级高的
            if priority[c["gen_method"]] > priority[uniq[key]["gen_method"]]:
                uniq[key] = c
    
    final_candidates = list(uniq.values())
    return final_candidates


def evaluate_pointer_candidate(
    vqa_model,
    image,
    question: str,
    candidate_pool: List[Dict],
    pointer: List[int],
    ground_truth_answers: List[str],
    build_vqa_prompt_fn: callable,
    compute_vqa_accuracy_fn: callable
) -> Tuple[str, int, float]:
    """
    评估 pointer 候选的 correctness
    
    按照强化学习.md §3.3 实现：
    1. 根据 pointer 从 candidate_pool 里取出两个示例
    2. 构造 in-context prompt
    3. 调 VQA 模型推理
    4. 用标准评测脚本比较答案
    
    Args:
        vqa_model: VQA 模型（如 Qwen2.5-VL-3B-Instruct）
        image: 查询图像
        question: 查询问题
        candidate_pool: 候选池列表，每个元素包含示例数据
        pointer: [i, j] 两个索引
        ground_truth_answers: 标准答案列表
        build_vqa_prompt_fn: 构建 VQA prompt 的函数
        compute_vqa_accuracy_fn: 计算 VQA 准确率的函数
    
    Returns:
        pred_answer: VQA 输出答案
        correct: 0/1（是否正确）
        acc_score: float [0,1]（准确率分数）
    """
    # 1) 根据 pointer 从 candidate_pool 里取出两个示例
    ex1 = candidate_pool[pointer[0]]
    ex2 = candidate_pool[pointer[1]]
    
    # 2) 构造 in-context prompt
    prompt = build_vqa_prompt_fn(image, question, ex1, ex2)
    
    # 3) 调 VQA 模型推理
    pred_answer = vqa_model.generate(prompt)
    
    # 4) 用标准评测脚本比较答案
    correct, acc_score = compute_vqa_accuracy_fn(pred_answer, ground_truth_answers)
    
    return pred_answer, int(correct), float(acc_score)
