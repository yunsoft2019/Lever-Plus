"""
Beam Dataset V3: 加载束搜索数据用于GRPO训练

来自强化学习.md 1.4节 束搜索数据结构：
- id_list: 5个beam，每个beam是一个shot序列（如[7232, 2229, 8211]）
- score_list: 5个beam对应的分数（如[0.046, 0.045, 0.037, ...]）

来自强化学习.md 2.2节 阶段1 数据准备：
- Query: 原始query的embedding
- Candidates: 候选池的embedding
- Labels: beam中的shot序列
- Rewards: beam的分数（归一化）
- Old_log_probs: 从SFT模型计算（冻结参数）

作者: Lever-Plus Team
日期: 2025-12-02
"""

import json
from typing import Dict, List, Optional, Tuple, Union
from functools import partial

import torch
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

from lever_lm.utils.reward_utils import compute_reward_for_candidate


class BeamDataset(Dataset):
    """
    Beam数据集：加载束搜索数据用于GRPO训练
    
    每个样本包含一个query对应的多个beam（默认5个）
    """
    
    def __init__(
        self,
        beam_data: Dict,
        index_ds: datasets.Dataset,
        beam_size: int = 5,
        shot_num: int = 2,
        query_image_field: str = "image",
        query_text_field: str = "question",
        normalize_rewards: bool = True
    ):
        """
        初始化Beam数据集
        
        Args:
            beam_data: 束搜索数据，格式为 {query_id: {"id_list": [...], "score_list": [...]}}
            index_ds: 索引数据集，用于获取query和candidate的原始数据
            beam_size: 每个query的beam数量（默认5）
            shot_num: 每个beam的shot数量（默认2）
            query_image_field: query图像字段名
            query_text_field: query文本字段名
            normalize_rewards: 是否对奖励进行归一化
        """
        super().__init__()
        
        self.index_ds = index_ds
        self.beam_size = beam_size
        self.shot_num = shot_num
        self.query_image_field = query_image_field
        self.query_text_field = query_text_field
        self.normalize_rewards = normalize_rewards
        
        # 解析beam数据
        self.samples = []
        for query_id, data in beam_data.items():
            query_id = int(query_id)
            id_list = data["id_list"]
            score_list = data["score_list"]
            
            # 确保beam数量一致
            assert len(id_list) == len(score_list), f"id_list和score_list长度不一致: {len(id_list)} vs {len(score_list)}"
            
            # 每个beam的格式: [icd1, icd2, ..., query_id]
            # 提取shot序列（不包含最后的query_id）
            beam_labels = []
            for beam in id_list:
                # beam格式: [shot1, shot2, query_id]
                shots = beam[:-1]  # 去掉最后的query_id
                assert len(shots) == shot_num, f"shot数量不匹配: {len(shots)} vs {shot_num}"
                beam_labels.append(shots)
            
            self.samples.append({
                "query_id": query_id,
                "beam_labels": beam_labels,  # [num_beams, shot_num]
                "beam_rewards": score_list   # [num_beams]
            })
        
        print(f"✓ BeamDataset 初始化完成")
        print(f"  - 样本数: {len(self.samples)}")
        print(f"  - beam_size: {beam_size}")
        print(f"  - shot_num: {shot_num}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict:
        """
        获取单个样本
        
        Returns:
            dict: {
                "query_id": int,
                "query_input": {"images": ..., "text": ...},
                "beam_labels": [num_beams, shot_num],
                "beam_rewards": [num_beams],
                "icd_indices": set of all ICD indices in this sample
            }
        """
        sample = self.samples[index]
        query_id = sample["query_id"]
        beam_labels = sample["beam_labels"]
        beam_rewards = sample["beam_rewards"]
        
        # 获取query输入
        query_input = {}
        if self.query_image_field:
            query_input["images"] = self.index_ds[query_id][self.query_image_field]
        if self.query_text_field:
            query_input["text"] = self.index_ds[query_id][self.query_text_field]
        
        # 收集所有ICD索引（用于后续获取candidate embedding）
        icd_indices = set()
        for beam in beam_labels:
            icd_indices.update(beam)
        
        # 转换为tensor
        beam_labels_tensor = torch.tensor(beam_labels, dtype=torch.long)
        beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)
        
        # 奖励归一化（组内Z-score）
        beam_rewards_normalized = beam_rewards_raw.clone()
        mean = beam_rewards_raw.mean()
        std = beam_rewards_raw.std()
        # 使用更小的阈值，因为info score可能很小（~1e-7）
        if std > 1e-12:
            beam_rewards_normalized = (beam_rewards_raw - mean) / std
        
        return {
            "query_id": query_id,
            "query_input": query_input,
            "beam_labels": beam_labels_tensor,  # [num_beams, shot_num]
            "beam_rewards": beam_rewards_normalized if self.normalize_rewards else beam_rewards_raw,  # [num_beams]
            "beam_rewards_raw": beam_rewards_raw,  # [num_beams] 原始分数，用于RCE的softmax权重
            "icd_indices": list(icd_indices)
        }


class BeamDatasetWithEmbedding(BeamDataset):
    """
    带Embedding的Beam数据集
    
    预先计算好query和candidate的embedding，用于快速训练
    """
    
    def __init__(
        self,
        beam_data: Dict,
        index_ds: datasets.Dataset,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_indices: List[int],
        beam_size: int = 5,
        shot_num: int = 2,
        normalize_rewards: bool = True
    ):
        """
        初始化带Embedding的Beam数据集
        
        Args:
            beam_data: 束搜索数据
            index_ds: 索引数据集
            query_embeddings: 预计算的query embeddings [num_queries, d]
            candidate_embeddings: 预计算的candidate embeddings [num_candidates, d]
            candidate_indices: candidate索引列表，用于映射
            beam_size: beam数量
            shot_num: shot数量
            normalize_rewards: 是否归一化奖励
        """
        # 不调用父类的__init__，直接初始化
        Dataset.__init__(self)
        
        self.index_ds = index_ds
        self.beam_size = beam_size
        self.shot_num = shot_num
        self.normalize_rewards = normalize_rewards
        
        self.query_embeddings = query_embeddings
        self.candidate_embeddings = candidate_embeddings
        
        # 构建candidate索引映射
        self.cand_idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}
        
        # 解析beam数据
        self.samples = []
        for query_id, data in beam_data.items():
            query_id = int(query_id)
            id_list = data["id_list"]
            score_list = data["score_list"]
            
            beam_labels = []
            for beam in id_list:
                shots = beam[:-1]
                beam_labels.append(shots)
            
            self.samples.append({
                "query_id": query_id,
                "beam_labels": beam_labels,
                "beam_rewards": score_list
            })
        
        print(f"✓ BeamDatasetWithEmbedding 初始化完成")
        print(f"  - 样本数: {len(self.samples)}")
        print(f"  - query_embeddings: {query_embeddings.shape}")
        print(f"  - candidate_embeddings: {candidate_embeddings.shape}")
    
    def __getitem__(self, index: int) -> Dict:
        """
        获取单个样本（包含embedding）
        """
        sample = self.samples[index]
        query_id = sample["query_id"]
        beam_labels = sample["beam_labels"]
        beam_rewards = sample["beam_rewards"]
        
        # 获取query embedding
        query_emb = self.query_embeddings[query_id]  # [d]
        
        # 获取所有candidate的embedding
        cand_emb = self.candidate_embeddings  # [K, d]
        
        # 转换beam_labels中的索引为candidate位置
        beam_labels_mapped = []
        for beam in beam_labels:
            mapped_beam = [self.cand_idx_to_pos[idx] for idx in beam]
            beam_labels_mapped.append(mapped_beam)
        
        beam_labels_tensor = torch.tensor(beam_labels_mapped, dtype=torch.long)
        beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)
        
        # 归一化
        beam_rewards_normalized = beam_rewards_raw.clone()
        mean = beam_rewards_raw.mean()
        std = beam_rewards_raw.std()
        if std > 1e-12:
            beam_rewards_normalized = (beam_rewards_raw - mean) / std
        
        return {
            "query_id": query_id,
            "query_emb": query_emb,  # [d]
            "cand_emb": cand_emb,    # [K, d]
            "beam_labels": beam_labels_tensor,  # [num_beams, shot_num]
            "beam_rewards": beam_rewards_normalized if self.normalize_rewards else beam_rewards_raw,  # [num_beams]
            "beam_rewards_raw": beam_rewards_raw  # [num_beams] 原始分数
        }


def collate_fn_v3(
    batch: List[Dict],
    processor: Optional[CLIPProcessor] = None
) -> Dict:
    """
    V3数据集的collate函数
    
    Args:
        batch: 样本列表
        processor: CLIP处理器（用于处理图像和文本）
    
    Returns:
        dict: 批次数据
    """
    # 检查是否有预计算的embedding
    has_embedding = "query_emb" in batch[0]
    
    if has_embedding:
        # 使用预计算的embedding
        return {
            "query_emb": torch.stack([item["query_emb"] for item in batch]),  # [B, d]
            "cand_emb": torch.stack([item["cand_emb"] for item in batch]),    # [B, K, d]
            "beam_labels": torch.stack([item["beam_labels"] for item in batch]),  # [B, num_beams, shot_num]
            "beam_rewards": torch.stack([item["beam_rewards"] for item in batch]),  # [B, num_beams]
            "beam_rewards_raw": torch.stack([item["beam_rewards_raw"] for item in batch]),  # [B, num_beams]
            "query_ids": [item["query_id"] for item in batch]
        }
    else:
        # 需要处理原始输入
        collate_dict = {
            "beam_labels": torch.stack([item["beam_labels"] for item in batch]),
            "beam_rewards": torch.stack([item["beam_rewards"] for item in batch]),
            "beam_rewards_raw": torch.stack([item["beam_rewards_raw"] for item in batch]),
            "query_ids": [item["query_id"] for item in batch],
            "icd_indices": [item["icd_indices"] for item in batch]
        }
        
        # 处理query输入
        if processor is not None:
            query_inputs = [item["query_input"] for item in batch]
            query_images = [q.get("images") for q in query_inputs if q.get("images") is not None]
            query_texts = [q.get("text") for q in query_inputs if q.get("text") is not None]
            
            if query_images or query_texts:
                processed = processor(
                    images=query_images if query_images else None,
                    text=query_texts if query_texts else None,
                    padding=True,
                    return_tensors="pt"
                )
                collate_dict["query_input"] = processed
        
        return collate_dict


class RLBeamDatasetWithEmbedding(Dataset):
    """
    RL Beam数据集：支持新的数据格式（包含 pointer_candidates 和 correctness）
    
    按照 v3_rl_from_current_code_full_plan.md 实现：
    - 支持新的 pointer_candidates 格式
    - 支持 vqa_correct 和 vqa_acc_score 字段
    - 使用新的 reward 公式：reward = hard + soft = vqa_correct + vqa_acc_score
    - 正样本：[1, 2]，负样本：[0, 1)
    """
    
    def __init__(
        self,
        rl_data: Dict,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_indices: List[int],
        shot_num: int = 2,
        normalize_rewards: bool = True,
        # 新的 reward 参数
        reward_mode: str = "hard_plus_soft",
        hard_weight: float = 1.0,
        soft_weight: float = 1.0,
        rel_weight: float = 0.1,  # relevance权重（hard_plus_gtprob_plus_rel模式使用）
        # 兼容旧接口的参数（默认不启用）
        reward_alpha: float = 0.0,
        reward_beta: float = 0.0,
        reward_correctness_mode: str = "01",
        use_logprob: bool = False,
        filter_gen_methods: Optional[List[str]] = None,
        skip_fallback_reward: bool = True,  # 3.3.2: 是否跳过使用 fallback 方式计算的 RL 样本（默认 True，推荐启用）
        require_positive_query: bool = False  # 是否只保留至少有一个正样本（vqa_correct=1）的 query
    ):
        """
        初始化RL Beam数据集
        
        Args:
            rl_data: RL数据，格式为 {query_id: {"pointer_candidates": [...]}}
            query_embeddings: 预计算的query embeddings [num_queries, d]
            candidate_embeddings: 预计算的candidate embeddings [num_candidates, d]
            candidate_indices: candidate索引列表，用于映射
            shot_num: shot数量
            normalize_rewards: 是否归一化奖励
            reward_mode: reward 模式（默认 "hard_plus_soft"）
                - "hard_plus_soft": reward = hard + soft（推荐）
                - "hard_only": reward = hard
                - "soft_only": reward = soft
                - "legacy": 使用旧的 alpha/beta 组合方式
            hard_weight: hard correctness 权重（默认 1.0）
            soft_weight: soft correctness 权重（默认 1.0）
            reward_alpha: quality权重（legacy 模式，默认 0.0）
            reward_beta: correctness权重（legacy 模式，默认 0.0）
            reward_correctness_mode: correctness模式（legacy 模式，"01" 或 "pm1"）
            use_logprob: 是否使用 logprob_score（legacy 模式）
            filter_gen_methods: 过滤的生成方法列表（如 ["beam", "sample"]），None表示不过滤
            skip_fallback_reward: 是否跳过使用 fallback 方式计算的 RL 样本（默认 True，推荐启用）
        """
        Dataset.__init__(self)
        
        self.shot_num = shot_num
        self.normalize_rewards = normalize_rewards
        self.reward_mode = reward_mode
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.rel_weight = rel_weight
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_correctness_mode = reward_correctness_mode
        self.use_logprob = use_logprob
        self.filter_gen_methods = filter_gen_methods
        self.skip_fallback_reward = skip_fallback_reward  # 3.3.2: 保存参数
        self.require_positive_query = require_positive_query  # 是否只保留有正样本的 query
        
        self.query_embeddings = query_embeddings
        self.candidate_embeddings = candidate_embeddings
        
        # 构建candidate索引映射
        self.cand_idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}
        
        # 解析RL数据
        self.samples = []
        for query_id_str, query_data in rl_data.items():
            # 【改动A】跳过_meta等非数字key
            try:
                query_id = int(query_id_str)
            except (ValueError, TypeError):
                # 跳过_meta等非数字key
                continue
            
            # 支持新格式：query_data可能是{"query": {...}, "pointer_candidates": [...]}
            # 或旧格式：{"pointer_candidates": [...]}
            if "query" in query_data:
                # 新格式：有query字段
                pointer_candidates = query_data.get("pointer_candidates", [])
            else:
                # 旧格式：直接是pointer_candidates
                pointer_candidates = query_data.get("pointer_candidates", [])
            
            # 过滤生成方法（如果指定）
            if self.filter_gen_methods is not None:
                pointer_candidates = [
                    c for c in pointer_candidates
                    if c.get("gen_method") in self.filter_gen_methods
                ]
            
            if len(pointer_candidates) == 0:
                continue
            
            # 提取labels和计算rewards
            beam_labels = []
            beam_rewards = []
            beam_logprobs = []
            
            for c in pointer_candidates:
                # P0: 跳过评估失败的候选（eval_failed=True）
                if c.get("eval_failed", False):
                    continue
                
                # 3.3.2: 如果要求跳过 fallback 样本
                if self.skip_fallback_reward and c.get("vqa_eval_mode") == "fallback":
                    continue
                
                # 跳过 error 类型的样本（评估过程中出错）
                if c.get("vqa_eval_mode") == "error":
                    continue
                
                # 【修复A】支持新格式：优先使用pointer_pos，否则使用pointer
                # 注意：pointer_pos 和 pointer 都是 global ID，需要映射到 position
                if "pointer_pos" in c:
                    # 优先使用pointer_pos（如果存在）
                    pointer = c["pointer_pos"]
                elif "pointer" in c:
                    # 使用pointer（如果pointer_pos不存在）
                    pointer = c["pointer"]
                else:
                    raise ValueError(f"候选数据中既没有pointer也没有pointer_pos字段 (query_id={query_id})")
                
                # 映射pointer中的索引为candidate位置（无论pointer还是pointer_pos都是global ID）
                # 严格检查：如果索引不在 candidate_indices 中，立即报错
                mapped_pointer = []
                for idx in pointer:
                    if idx not in self.cand_idx_to_pos:
                        raise KeyError(
                            f"[RLBeamDatasetWithEmbedding] Pointer index {idx} not in candidate_indices "
                            f"(query_id={query_id}, candidate_indices范围: {min(self.cand_idx_to_pos.keys()) if self.cand_idx_to_pos else 'N/A'}-{max(self.cand_idx_to_pos.keys()) if self.cand_idx_to_pos else 'N/A'})"
                        )
                    mapped_pointer.append(self.cand_idx_to_pos[idx])
                
                assert len(mapped_pointer) == shot_num, f"pointer长度不匹配: {len(mapped_pointer)} vs {shot_num}"
                beam_labels.append(mapped_pointer)
                
                # 计算组合reward（使用新的 reward_mode）
                # 根据文档要求：不要将 beam_score/logprob_score 混入 reward（除非 legacy 模式）
                # P1: 支持 vqa_gt_prob 作为 soft reward
                if self.reward_mode == "legacy":
                    reward = compute_reward_for_candidate(
                        beam_score=c.get("beam_score"),
                        logprob_score=c.get("logprob_score"),
                        vqa_correct=c.get("vqa_correct"),
                        vqa_acc_score=c.get("vqa_acc_score"),
                        vqa_gt_prob=c.get("vqa_gt_prob"),  # P1: 传入 vqa_gt_prob
                        reward_mode=self.reward_mode,
                        hard_weight=self.hard_weight,
                        soft_weight=self.soft_weight,
                        alpha=self.reward_alpha,
                        beta=self.reward_beta,
                        correctness_mode=self.reward_correctness_mode,
                        use_logprob=self.use_logprob
                    )
                else:
                    # 非 legacy 模式：传入 correctness 相关参数，包括 vqa_gt_prob 和 vqa_rel_score
                    reward = compute_reward_for_candidate(
                        vqa_correct=c.get("vqa_correct"),
                        vqa_acc_score=c.get("vqa_acc_score"),
                        vqa_gt_prob=c.get("vqa_gt_prob"),  # P1: 传入 vqa_gt_prob
                        vqa_rel_score=c.get("vqa_rel_score"),  # 【改动B】传入relevance
                        reward_mode=self.reward_mode,
                        hard_weight=self.hard_weight,
                        soft_weight=self.soft_weight,
                        rel_weight=self.rel_weight,  # relevance权重
                        alpha=self.reward_alpha,
                        beta=self.reward_beta,
                        correctness_mode=self.reward_correctness_mode,
                        use_logprob=self.use_logprob
                    )
                beam_rewards.append(reward)
                
                # 保存logprob（用于old_log_probs）
                beam_logprobs.append(c.get("logprob_score"))
            
            # 3.3.2: 如果过滤后没有任何候选，跳过这个 query
            if len(beam_labels) == 0:
                continue
            
            # 如果要求只保留有正样本的 query
            if self.require_positive_query:
                # 检查是否有任何正样本（reward >= 1.0 表示 vqa_correct=1）
                has_positive = any(r >= 1.0 for r in beam_rewards)
                if not has_positive:
                    continue
            
            self.samples.append({
                "query_id": query_id,
                "beam_labels": beam_labels,  # [num_candidates, shot_num]
                "beam_rewards": beam_rewards,  # [num_candidates]
                "beam_logprobs": beam_logprobs  # [num_candidates]（可选）
            })
        
        print(f"✓ RLBeamDatasetWithEmbedding 初始化完成")
        print(f"  - 样本数: {len(self.samples)}")
        print(f"  - query_embeddings: {query_embeddings.shape}")
        print(f"  - candidate_embeddings: {candidate_embeddings.shape}")
        if self.skip_fallback_reward:
            print(f"  - skip_fallback_reward: True（已过滤 fallback 样本，只使用官方 VQA metric）")
        else:
            print(f"  - skip_fallback_reward: False（使用所有样本，包括 fallback）")
        if self.require_positive_query:
            print(f"  - require_positive_query: True（只保留至少有一个正样本的 query）")
        else:
            print(f"  - require_positive_query: False（使用所有 query）")
        print(f"  - reward_mode: {reward_mode}")
        if reward_mode == "hard_plus_soft":
            print(f"  - hard_weight: {hard_weight}, soft_weight: {soft_weight}")
            print(f"  - reward 范围: [0, {hard_weight + soft_weight}]（正样本 [{hard_weight}, {hard_weight + soft_weight}]，负样本 [0, {hard_weight})）")
        elif reward_mode == "hard_plus_soft_v2":
            print(f"  - hard_weight: {hard_weight}, soft_weight: {soft_weight}")
            print(f"  - reward 范围: [0, {soft_weight + 2*hard_weight}]（正样本 [{2*hard_weight}, {soft_weight + 2*hard_weight}]，负样本 [0, {soft_weight})）")
            print(f"  - 增大正负样本差距")
        elif reward_mode == "separated":
            print(f"  - hard_weight: {hard_weight}, soft_weight: {soft_weight}")
            print(f"  - 正样本 reward: [{2*hard_weight}, {2*hard_weight + soft_weight}]")
            print(f"  - 负样本 reward: [0, {soft_weight}]")
            print(f"  - 【推荐】阈值分离，正负样本有明确 gap（至少 {2*hard_weight - soft_weight}）")
        elif reward_mode == "hybrid":
            print(f"  - 混合 InfoScore 和 correctness")
            print(f"  - hard_weight 作为 InfoScore 权重: {hard_weight}")
        elif reward_mode == "legacy":
            print(f"  - reward_alpha: {reward_alpha}, reward_beta: {reward_beta}")
            print(f"  - reward_correctness_mode: {reward_correctness_mode}")
        if self.filter_gen_methods:
            print(f"  - 过滤生成方法: {self.filter_gen_methods}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict:
        """
        获取单个样本（包含embedding）
        
        Returns:
            dict: {
                "query_id": int,
                "query_emb": [d],
                "cand_emb": [K, d],
                "beam_labels": [num_candidates, shot_num],
                "beam_rewards": [num_candidates],
                "beam_rewards_raw": [num_candidates],
                "beam_logprobs": [num_candidates]（可选）
            }
        """
        sample = self.samples[index]
        query_id = sample["query_id"]
        beam_labels = sample["beam_labels"]
        beam_rewards = sample["beam_rewards"]
        beam_logprobs = sample.get("beam_logprobs")
        
        # 获取query embedding
        query_emb = self.query_embeddings[query_id]  # [d]
        
        # 获取所有candidate的embedding
        cand_emb = self.candidate_embeddings  # [K, d]
        
        # 转换为tensor
        beam_labels_tensor = torch.tensor(beam_labels, dtype=torch.long)  # [num_candidates, shot_num]
        beam_rewards_raw = torch.tensor(beam_rewards, dtype=torch.float32)  # [num_candidates]
        
        # 归一化（组内Z-score）
        beam_rewards_normalized = beam_rewards_raw.clone()
        mean = beam_rewards_raw.mean()
        std = beam_rewards_raw.std()
        if std > 1e-12:
            beam_rewards_normalized = (beam_rewards_raw - mean) / std
        
        result = {
            "query_id": query_id,
            "query_emb": query_emb,  # [d]
            "cand_emb": cand_emb,    # [K, d]
            "beam_labels": beam_labels_tensor,  # [num_candidates, shot_num]
            "beam_rewards": beam_rewards_normalized if self.normalize_rewards else beam_rewards_raw,  # [num_candidates]
            "beam_rewards_raw": beam_rewards_raw  # [num_candidates] 原始reward
        }
        
        # 添加logprobs（如果可用）
        if beam_logprobs and all(lp is not None for lp in beam_logprobs):
            result["beam_logprobs"] = torch.tensor(beam_logprobs, dtype=torch.float32)
        
        return result


def collate_fn_rl_v3(batch: List[Dict]) -> Dict:
    """
    RL V3数据集的collate函数
    
    用于RLBeamDatasetWithEmbedding，假设batch_size=1（每个batch是一个query-group）
    
    Args:
        batch: 样本列表（通常只有一个样本，因为batch_size=1）
    
    Returns:
        dict: 批次数据
    """
    item = batch[0]  # batch_size=1，所以只有一个item
    
    result = {
        "query_id": item["query_id"],
        "query_emb": item["query_emb"].unsqueeze(0),  # [1, d]
        "cand_emb": item["cand_emb"].unsqueeze(0),    # [1, K, d]
        "beam_labels": item["beam_labels"].unsqueeze(0),  # [1, num_candidates, shot_num]
        "beam_rewards": item["beam_rewards"].unsqueeze(0),  # [1, num_candidates]
        "beam_rewards_raw": item["beam_rewards_raw"].unsqueeze(0),  # [1, num_candidates]
    }
    
    # 添加logprobs（如果可用）
    if "beam_logprobs" in item:
        result["beam_logprobs"] = item["beam_logprobs"].unsqueeze(0)  # [1, num_candidates]
    
    return result


def load_beam_data(json_path: str) -> Dict:
    """
    加载束搜索JSON数据
    
    支持两种格式：
    1. 旧格式：{query_id: {"id_list": [...], "score_list": [...]}}
    2. 新格式：{query_id: {"pointer_candidates": [...]}}
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        dict: 束搜索数据或RL数据
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def split_beam_data(
    beam_data: Dict,
    train_ratio: float = 0.8
) -> Tuple[Dict, Dict]:
    """
    划分训练集和验证集
    
    Args:
        beam_data: 束搜索数据
        train_ratio: 训练集比例
    
    Returns:
        (train_data, val_data)
    """
    # 跳过_meta等非数字key
    query_ids = [k for k in beam_data.keys() if k != "_meta" and (k.isdigit() or (isinstance(k, str) and k.replace('-', '').replace('_', '').isdigit()))]
    num_train = int(len(query_ids) * train_ratio)
    
    # 按query_id排序后划分，保证可复现
    query_ids_sorted = sorted(query_ids, key=int)
    train_ids = set(query_ids_sorted[:num_train])
    
    train_data = {k: v for k, v in beam_data.items() if k in train_ids}
    val_data = {k: v for k, v in beam_data.items() if k not in train_ids and k != "_meta"}
    
    # 保留_meta字段（如果存在）
    if "_meta" in beam_data:
        train_data["_meta"] = beam_data["_meta"]
        val_data["_meta"] = beam_data["_meta"]
    
    print(f"数据划分: 训练集 {len([k for k in train_data.keys() if k != '_meta'])} 个query, 验证集 {len([k for k in val_data.keys() if k != '_meta'])} 个query")
    
    return train_data, val_data


if __name__ == "__main__":
    """测试代码"""
    import os
    
    print("="*70)
    print("测试 BeamDataset")
    print("="*70)
    
    # 加载测试数据
    test_json = "results/okvqa/generated_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800.json"
    
    if os.path.exists(test_json):
        beam_data = load_beam_data(test_json)
        print(f"\n加载数据: {len(beam_data)} 个样本")
        
        # 显示第一个样本
        first_key = list(beam_data.keys())[0]
        print(f"\n第一个样本 (query_id={first_key}):")
        print(f"  id_list: {beam_data[first_key]['id_list']}")
        print(f"  score_list: {beam_data[first_key]['score_list']}")
        
        # 划分数据
        train_data, val_data = split_beam_data(beam_data, train_ratio=0.8)
        
        # 创建模拟的index_ds（用于测试）
        class MockIndexDS:
            def __getitem__(self, idx):
                return {
                    "image": f"mock_image_{idx}",
                    "question": f"mock_question_{idx}"
                }
        
        mock_ds = MockIndexDS()
        
        # 创建数据集
        dataset = BeamDataset(
            beam_data=train_data,
            index_ds=mock_ds,
            beam_size=5,
            shot_num=2,
            normalize_rewards=True
        )
        
        # 测试__getitem__
        print(f"\n测试 __getitem__:")
        sample = dataset[0]
        print(f"  query_id: {sample['query_id']}")
        print(f"  beam_labels shape: {sample['beam_labels'].shape}")
        print(f"  beam_rewards shape: {sample['beam_rewards'].shape}")
        print(f"  beam_rewards (normalized): {sample['beam_rewards'].tolist()}")
        print(f"  beam_rewards_raw: {sample['beam_rewards_raw'].tolist()}")
        print(f"  beam_rewards mean: {sample['beam_rewards'].mean():.4f}, std: {sample['beam_rewards'].std():.4f}")
        print(f"  icd_indices: {sample['icd_indices']}")
        
        # 测试DataLoader
        print(f"\n测试 DataLoader:")
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn_v3
        )
        
        batch = next(iter(dataloader))
        print(f"  batch keys: {batch.keys()}")
        print(f"  beam_labels shape: {batch['beam_labels'].shape}")
        print(f"  beam_rewards shape: {batch['beam_rewards'].shape}")
        
        # 测试BeamDatasetWithEmbedding
        print(f"\n测试 BeamDatasetWithEmbedding:")
        num_queries = max(int(k) for k in beam_data.keys()) + 1
        num_candidates = 10000  # 假设有10000个候选
        d_model = 768
        
        # 创建模拟的embedding
        query_embeddings = torch.randn(num_queries, d_model)
        candidate_embeddings = torch.randn(num_candidates, d_model)
        candidate_indices = list(range(num_candidates))
        
        dataset_with_emb = BeamDatasetWithEmbedding(
            beam_data=train_data,
            index_ds=mock_ds,
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            candidate_indices=candidate_indices,
            beam_size=5,
            shot_num=2
        )
        
        sample_emb = dataset_with_emb[0]
        print(f"  query_emb shape: {sample_emb['query_emb'].shape}")
        print(f"  cand_emb shape: {sample_emb['cand_emb'].shape}")
        print(f"  beam_labels shape: {sample_emb['beam_labels'].shape}")
        
        print("\n" + "="*70)
        print("✓ BeamDataset 测试通过！")
        print("="*70)
    else:
        print(f"测试数据文件不存在: {test_json}")
        print("请确保在 Lever-Plus 项目根目录运行此测试")
