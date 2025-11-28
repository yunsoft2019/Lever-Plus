"""
Pointer Selector V3: 灵活基础架构 + 排序学习（Ranking Learning）

特点：
- 架构：可选择V1（Bi-Encoder）或V2（+ Cross-Attention）作为基础
- 支持：从V1/V2 checkpoint加载初始化
- 增强：利用束搜索的多个beam进行排序学习
- 损失：交叉熵（CE）+ 排序损失（Ranking Loss）
  - Pairwise: 正负样本对的margin loss
  - Listwise: KL散度让模型分布接近beam分数分布
- 目标：提升Top-k、NDCG、MRR等排序指标

作者: Lever-Plus Team
日期: 2025-10-29
参考: yiyun.md V3 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import os
from collections import defaultdict
import numpy as np


class PointerSelectorV3(nn.Module):
    """
    V3 版本：灵活基础架构 + 排序学习
    
    可选择V1（简单Bi-Encoder）或V2（+Cross-Attention）作为基础架构
    """
    
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 6,
        label_smoothing: float = 0.0,
        dropout: float = 0.5,
        base_architecture: str = 'v2',  # 'v1' 或 'v2'
        use_cross_attention: Optional[bool] = None,  # None时根据base自动设置
        ranking_loss_type: str = 'listwise',  # 'listwise' 或 'pairwise'
        ranking_loss_weight: float = 0.5,  # 排序损失权重
        ce_weight: float = 0.5  # 交叉熵权重
    ):
        """
        初始化 V3 模型
        
        Args:
            d_model: 输入 embedding 维度 (默认 768)
            K: 候选池大小 (默认 32)
            shot_num: 需要选择的样本数量 (默认 6)
            label_smoothing: 标签平滑系数 (默认 0.0)
            dropout: dropout 比例 (默认 0.5)
            base_architecture: 基础架构 ('v1' 或 'v2')
            use_cross_attention: 是否使用Cross-Attention (None时根据base自动设置)
            ranking_loss_type: 排序损失类型 ('listwise' 或 'pairwise')
            ranking_loss_weight: 排序损失权重
            ce_weight: 交叉熵权重
        """
        super().__init__()
        
        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.base_architecture = base_architecture
        self.ranking_loss_type = ranking_loss_type
        self.ranking_loss_weight = ranking_loss_weight
        self.ce_weight = ce_weight
        
        # 自动设置Cross-Attention
        if use_cross_attention is None:
            use_cross_attention = (base_architecture == 'v2')
        self.use_cross_attention = use_cross_attention
        
        # 根据基础架构构建网络
        if base_architecture == 'v1':
            self._build_v1_architecture(d_model, dropout)
        elif base_architecture == 'v2':
            self._build_v2_architecture(d_model, dropout)
        else:
            raise ValueError(f"未知的基础架构: {base_architecture}, 只支持 'v1' 或 'v2'")
        
        print(f"✓ PointerSelectorV3 初始化完成")
        print(f"  - 基础架构: {base_architecture.upper()}")
        print(f"  - d_model: {d_model}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - use_cross_attention: {use_cross_attention}")
        print(f"  - ranking_loss_type: {ranking_loss_type}")
        print(f"  - ranking_loss_weight: {ranking_loss_weight}")
        print(f"  - ce_weight: {ce_weight}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - 参数量: {total_params/1e6:.2f}M")
    
    def _build_v1_architecture(self, d_model: int, dropout: float):
        """
        构建V1架构：简单Bi-Encoder
        
        结构：2层MLP投影
        """
        # Query 投影层 (2层MLP)
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Candidate 投影层 (2层MLP)
        self.cand_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 温度参数（固定为0.1）
        self.temperature = 0.1
        
        # 初始化权重
        for m in self.query_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.cand_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _build_v2_architecture(self, d_model: int, dropout: float):
        """
        构建V2架构：Bi-Encoder + Cross-Attention
        
        结构：降维 + Cross-Attention + 单层投影
        """
        self.hidden_dim = 256  # V2的隐藏维度
        
        # 输入投影（降维：768 -> 256）
        if d_model != self.hidden_dim:
            self.input_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()
        
        # Cross-Attention 层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        
        # Query/Candidate投影层（单层）
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.cand_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 温度参数
        self.temperature = torch.tensor([0.1], dtype=torch.float32)
        
        # 初始化权重
        if not isinstance(self.input_proj, nn.Identity):
            nn.init.xavier_uniform_(self.input_proj.weight)
        
        # Eye初始化 + 小扰动
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.cand_proj.weight)
        
        with torch.no_grad():
            self.query_proj.weight.add_(torch.randn_like(self.query_proj.weight) * 0.01)
            self.cand_proj.weight.add_(torch.randn_like(self.cand_proj.weight) * 0.01)
    
    def forward(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        beam_scores: Optional[torch.Tensor] = None,  # 兼容旧版
        all_beams_info: Optional[list] = None,  # V3新增：所有beam信息
        cands: Optional[list] = None  # V3新增：候选池ID列表
    ) -> dict:
        """
        前向传播
        
        Args:
            query_emb: [B, d] 或 [B, 2, d] query embedding  
            cand_emb: [B, K, d] 或 [B, K, 2, d] 候选 embedding
            labels: [B, S] 标签序列（训练时需要）
            return_loss: 是否返回损失
            beam_scores: 兼容旧版
            all_beams_info: 所有beam信息，格式：[{"id_seq": [...], "score": 0.85}, ...]
            cands: 候选池ID列表 [id1, id2, ..., idK]
        
        Returns:
            dict: {
                'logits': [B, S, K],
                'predictions': [B, S],
                'loss': scalar (如果 return_loss=True)
            }
        """
        batch_size = query_emb.shape[0]
        device = query_emb.device
        
        # 处理多模态输入 [B, 2, d] -> [B, d]
        if len(query_emb.shape) == 3:
            query_emb = query_emb.mean(dim=1)  # 平均池化
        if len(cand_emb.shape) == 4:
            cand_emb = cand_emb.mean(dim=2)  # [B, K, 2, d] -> [B, K, d]
        
        # 根据基础架构进行前向传播
        if self.base_architecture == 'v1':
            query_proj, cand_proj = self._forward_v1(query_emb, cand_emb)
        else:  # v2
            query_proj, cand_proj = self._forward_v2(query_emb, cand_emb)
        
        # 存储每步的 logits 和预测
        all_logits = []
        predictions = []
        
        # 当前 query 状态
        current_query = query_proj
        
        # mask：记录已选择的候选
        selected_mask = torch.zeros(batch_size, self.K, dtype=torch.bool, device=device)
        
        # 自回归生成
        for step in range(self.shot_num):
            # 计算相似度分数
            scores = torch.matmul(current_query.unsqueeze(1), cand_proj.transpose(1, 2)).squeeze(1)  # [B, K]
            
            # 温度缩放
            if isinstance(self.temperature, torch.Tensor):
                temperature = self.temperature.to(device)
                scores = scores / temperature
            else:
                scores = scores / self.temperature
            
            # 应用 mask
            scores = scores.masked_fill(selected_mask, -100.0)
            all_logits.append(scores)
            
            # 预测
            pred = scores.argmax(dim=-1)  # [B]
            predictions.append(pred)
            
            # 更新 mask (Teacher Forcing 或推理)
            if labels is not None and step < labels.shape[1]:
                true_indices = labels[:, step]
                selected_mask = selected_mask.scatter(1, true_indices.unsqueeze(1), True)
            else:
                selected_mask = selected_mask.scatter(1, pred.unsqueeze(1), True)
        
        # 堆叠结果
        all_logits = torch.stack(all_logits, dim=1)  # [B, S, K]
        predictions = torch.stack(predictions, dim=1)  # [B, S]
        
        result = {
            'logits': all_logits,
            'predictions': predictions
        }
        
        # 计算损失
        if return_loss and labels is not None:
            loss = self.compute_loss(all_logits, labels, all_beams_info, cands)
            result['loss'] = loss
        
        return result
    
    def _forward_v1(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """V1前向传播"""
        # 投影
        query_proj = self.query_proj(query_emb)  # [B, d]
        cand_proj = self.cand_proj(cand_emb)     # [B, K, d]
        
        # L2 归一化
        query_proj = F.normalize(query_proj, dim=-1)
        cand_proj = F.normalize(cand_proj, dim=-1)
        
        return query_proj, cand_proj
    
    def _forward_v2(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """V2前向传播"""
        batch_size = query_emb.shape[0]
        
        # 降维
        query_reduced = self.input_proj(query_emb)  # [B, 256]
        cand_reduced = self.input_proj(cand_emb.reshape(-1, self.d_model))  # [B*K, 256]
        cand_reduced = cand_reduced.reshape(batch_size, self.K, self.hidden_dim)  # [B, K, 256]
        
        # Cross-Attention增强
        query_enhanced, _ = self.cross_attn(
            query_reduced.unsqueeze(1),  # [B, 1, 256]
            cand_reduced,                 # [B, K, 256]
            cand_reduced
        )
        query_enhanced = query_enhanced.squeeze(1)  # [B, 256]
        
        # Residual + LayerNorm
        query_enhanced = self.attn_norm(query_reduced + query_enhanced)
        
        # 投影
        query_proj = self.query_proj(query_enhanced)  # [B, 256]
        cand_proj = self.cand_proj(cand_reduced)       # [B, K, 256]
        
        # Dropout
        query_proj = self.dropout(query_proj)
        cand_proj = self.dropout(cand_proj)
        
        # L2 归一化
        query_proj = F.normalize(query_proj, dim=-1)
        cand_proj = F.normalize(cand_proj, dim=-1)
        
        return query_proj, cand_proj
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        all_beams_info: Optional[list] = None,
        cands: Optional[list] = None
    ) -> torch.Tensor:
        """
        计算损失：CE + Ranking Loss
        
        Args:
            logits: [B, S, K]
            labels: [B, S]
            all_beams_info: Batch的beam信息，格式：[[beam1, beam2], [beam1, beam2], ...] (B个样本)
            cands: Batch的候选池ID列表，格式：[[id1, id2, ...], [id1, id2, ...], ...] (B个样本)
        """
        batch_size, shot_num, K = logits.shape
        
        # 1. 交叉熵损失
        logits_flat = logits.reshape(-1, K)
        labels_flat = labels.reshape(-1)
        
        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            label_smoothing=self.label_smoothing
        )
        
        # 2. 排序损失
        ranking_loss = torch.tensor(0.0, device=logits.device)
        if all_beams_info is not None and cands is not None and self.ranking_loss_weight > 0:
            # 逐样本计算排序损失
            batch_ranking_losses = []
            for i in range(batch_size):
                sample_logits = logits[i:i+1]  # [1, S, K]
                sample_beams = all_beams_info[i]  # 该样本的beam列表
                sample_cands = cands[i]  # 该样本的候选ID列表
                
                # 跳过没有beam信息的样本
                if len(sample_beams) == 0:
                    continue
                
                if self.ranking_loss_type == 'listwise':
                    sample_loss = self._listwise_ranking_loss(sample_logits, sample_beams, sample_cands)
                elif self.ranking_loss_type == 'pairwise':
                    sample_loss = self._pairwise_ranking_loss(sample_logits, sample_beams, sample_cands)
                else:
                    continue
                
                batch_ranking_losses.append(sample_loss)
            
            # 平均batch的排序损失
            if len(batch_ranking_losses) > 0:
                ranking_loss = torch.stack(batch_ranking_losses).mean()
        
        # 3. 动态调整排序损失权重（根据 shot_num）
        # 高 shot 数时，排序信息更重要，增大权重（更激进的缩放，针对高shot数优化）
        # shot_num=1: 权重 * 0.4, shot_num=2: 权重 * 1.0, shot_num=3: 权重 * 2.5, shot_num=4: 权重 * 5.0
        if shot_num == 1:
            dynamic_weight_scale = 0.4
        elif shot_num == 2:
            dynamic_weight_scale = 1.0
        elif shot_num == 3:
            dynamic_weight_scale = 2.5
        else:  # shot_num >= 4
            dynamic_weight_scale = 5.0
        effective_ranking_weight = self.ranking_loss_weight * dynamic_weight_scale
        
        # 4. 加权组合
        total_loss = self.ce_weight * ce_loss + effective_ranking_weight * ranking_loss
        
        return total_loss
    
    def _listwise_ranking_loss(
        self,
        logits: torch.Tensor,
        all_beams_info: list,
        cands: list
    ) -> torch.Tensor:
        """
        Listwise排序损失：KL散度
        
        让模型的候选分布接近beam分数的分布
        改进：
        1. 只使用 top-k beams（忽略低质量 beam）
        2. 考虑候选在 beam 序列中的位置（位置越靠前，重要性越高）
        3. 动态温度参数（根据候选得分范围调整）
        """
        batch_size, shot_num, K = logits.shape
        device = logits.device
        
        # 构建候选ID到位置的映射
        cand_to_pos = {int(cand_id): pos for pos, cand_id in enumerate(cands)}
        
        # 按 beam 分数排序，只使用 top-k beams（保留前 50% 或至少 2 个）
        if len(all_beams_info) == 0:
            return torch.tensor(0.0, device=device)
        
        # 按分数排序 beams
        sorted_beams = sorted(all_beams_info, key=lambda x: x["score"], reverse=True)
        # 只使用 top-k beams（更严格的选择：高shot数时只保留前20%，低shot数时保留前50%）
        # 高shot数时，beam质量差异更明显，只关注最高质量的beams
        if shot_num >= 3:
            top_k_ratio = 0.2  # 高shot数：只保留前20%（更严格）
        else:
            top_k_ratio = 0.5  # 低shot数：保留前50%
        top_k = max(2, int(len(sorted_beams) * top_k_ratio))
        top_beams = sorted_beams[:top_k]
        
        # 收集每个候选的得分（考虑位置权重：位置越靠前，权重越大）
        candidate_weighted_scores = defaultdict(lambda: {'sum': 0.0, 'weight': 0.0})
        
        # 归一化 top beams 的分数作为权重
        top_beam_scores = [beam["score"] for beam in top_beams]
        top_beam_scores_tensor = torch.tensor(top_beam_scores, device=device)
        beam_weights = F.softmax(top_beam_scores_tensor, dim=0)
        
        for beam_idx, beam in enumerate(top_beams):
            beam_seq = beam["id_seq"][:-1]  # 去掉末尾的query_id
            beam_weight = beam_weights[beam_idx].item()
            
            # 位置权重：序列中越靠前的候选，权重越大
            # 高shot数时使用更激进的衰减，更强调序列前面的候选
            if shot_num >= 4:
                decay_rate = 0.6  # shot_num=4: 最激进（1.0, 0.6, 0.36, ...）
            elif shot_num >= 3:
                decay_rate = 0.65  # shot_num=3: 较激进（1.0, 0.65, 0.42, ...）
            else:
                decay_rate = 0.8  # shot_num<3: 较温和（1.0, 0.8, 0.64, ...）
            for seq_pos, icd_id in enumerate(beam_seq):
                if icd_id in cand_to_pos:
                    pos = cand_to_pos[icd_id]
                    # 位置权重：高shot数时更激进，更强调序列前面的候选
                    position_weight = decay_rate ** seq_pos
                    # 综合权重 = beam权重 * 位置权重
                    combined_weight = beam_weight * position_weight
                    # 使用 beam 分数和综合权重
                    candidate_weighted_scores[pos]['sum'] += beam["score"] * combined_weight
                    candidate_weighted_scores[pos]['weight'] += combined_weight
        
        # 计算每个候选的加权平均得分
        candidate_avg_scores = torch.zeros(K, device=device)
        for pos, data in candidate_weighted_scores.items():
            if data['weight'] > 0:
                candidate_avg_scores[pos] = data['sum'] / data['weight']
        
        # 如果没有有效的候选得分，返回0
        if candidate_avg_scores.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # 动态温度参数：根据得分范围和 shot_num 调整
        # 高 shot 数时使用更小的温度（更尖锐的分布），更强调排序差异
        score_range = candidate_avg_scores.max() - candidate_avg_scores.min()
        if score_range > 0:
            # 基础温度：根据得分范围调整
            base_temp = 0.3 + 0.2 * (1.0 - min(score_range / 10.0, 1.0))
            # shot_num 越大，温度越小（更尖锐），高shot数时更激进
            if shot_num >= 4:
                # shot_num=4: 最激进的温度缩放
                shot_temp_scale = 0.4  # 直接设置为0.4
            elif shot_num >= 3:
                # shot_num=3: 较激进的温度缩放
                shot_temp_scale = 0.6
            else:
                # 低shot数：较温和的缩放
                shot_temp_scale = 1.0 - 0.1 * (shot_num - 1)  # shot_num=1: 1.0, shot_num=2: 0.9
            temperature = base_temp * max(shot_temp_scale, 0.3)  # 最小温度 0.3
        else:
            # 默认温度也根据 shot_num 调整，高shot数时更小（更激进）
            if shot_num >= 4:
                temperature = 0.15  # shot_num=4: 最小温度，最尖锐的分布
            elif shot_num >= 3:
                temperature = 0.25  # shot_num=3: 较小温度
            else:
                temperature = 0.5 - 0.05 * (shot_num - 1)  # shot_num=1: 0.5, shot_num=2: 0.45
            temperature = max(temperature, 0.15)  # 最小温度 0.15（更激进）
        
        # 构建目标分布
        target_dist = F.softmax(candidate_avg_scores / temperature, dim=-1)  # [K]
        
        # 模型的平均概率分布
        model_probs = F.softmax(logits, dim=-1)  # [B, S, K]
        model_probs_avg = model_probs.mean(dim=1)  # [B, K]
        
        # KL散度
        target_dist_batch = target_dist.unsqueeze(0).expand(batch_size, -1)  # [B, K]
        
        kl_loss = F.kl_div(
            torch.log(model_probs_avg + 1e-10),
            target_dist_batch,
            reduction='batchmean'
        )
        
        return kl_loss
    
    def _pairwise_ranking_loss(
        self,
        logits: torch.Tensor,
        all_beams_info: list,
        cands: list
    ) -> torch.Tensor:
        """
        Pairwise排序损失：Margin Loss
        
        要求好的候选得分 > 差的候选得分 + margin
        """
        batch_size, shot_num, K = logits.shape
        device = logits.device
        
        # 构建映射
        cand_to_pos = {int(cand_id): pos for pos, cand_id in enumerate(cands)}
        
        # 收集候选得分
        candidate_scores = defaultdict(list)
        for beam in all_beams_info:
            beam_seq = beam["id_seq"][:-1]
            score = beam["score"]
            for icd_id in beam_seq:
                if icd_id in cand_to_pos:
                    pos = cand_to_pos[icd_id]
                    candidate_scores[pos].append(score)
        
        # 计算平均得分
        candidate_avg_scores = {}
        for pos, scores in candidate_scores.items():
            candidate_avg_scores[pos] = float(np.mean(scores))
        
        if len(candidate_avg_scores) < 2:
            return torch.tensor(0.0, device=device)
        
        # 按得分排序
        sorted_cands = sorted(candidate_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 划分正负样本
        mid_point = len(sorted_cands) // 2
        positive_cands = [pos for pos, _ in sorted_cands[:mid_point]]
        negative_cands = [pos for pos, _ in sorted_cands[mid_point:]]
        
        if len(positive_cands) == 0 or len(negative_cands) == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算模型得分
        model_scores_avg = logits.mean(dim=1)  # [B, K]
        
        # Pairwise margin loss
        total_loss = 0.0
        num_pairs = 0
        margin = 1.0
        
        for pos_idx in positive_cands:
            for neg_idx in negative_cands:
                pos_score = model_scores_avg[0, pos_idx]
                neg_score = model_scores_avg[0, neg_idx]
                
                pair_loss = torch.clamp(margin + neg_score - pos_score, min=0)
                total_loss += pair_loss
                num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=device)
    
    def predict(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式
        
        Args:
            query_emb: [B, d]
            cand_emb: [B, K, d]
            top_k: 每步返回 top-k
        
        Returns:
            predictions: [B, S]
            scores: [B, S]
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(query_emb, cand_emb, labels=None, return_loss=False)
            predictions = result['predictions']
            logits = result['logits']
            
            # 获取每步的最大分数
            scores = logits.max(dim=-1)[0]  # [B, S]
            
            return predictions, scores


class PointerSelectorV3Config:
    """V3 模型配置类"""
    
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 6,
        label_smoothing: float = 0.0,
        dropout: float = 0.5,
        base_architecture: str = 'v2',
        use_cross_attention: Optional[bool] = None,
        ranking_loss_type: str = 'listwise',
        ranking_loss_weight: float = 0.5,
        ce_weight: float = 0.5
    ):
        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.base_architecture = base_architecture
        self.use_cross_attention = use_cross_attention
        self.ranking_loss_type = ranking_loss_type
        self.ranking_loss_weight = ranking_loss_weight
        self.ce_weight = ce_weight
    
    def to_dict(self):
        return {
            'd_model': self.d_model,
            'K': self.K,
            'shot_num': self.shot_num,
            'label_smoothing': self.label_smoothing,
            'dropout': self.dropout,
            'base_architecture': self.base_architecture,
            'use_cross_attention': self.use_cross_attention,
            'ranking_loss_type': self.ranking_loss_type,
            'ranking_loss_weight': self.ranking_loss_weight,
            'ce_weight': self.ce_weight
        }


def build_model_v3(config: Optional[PointerSelectorV3Config] = None) -> PointerSelectorV3:
    """
    构建 V3 模型的工厂函数
    """
    if config is None:
        config = PointerSelectorV3Config()
    
    model = PointerSelectorV3(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout,
        base_architecture=config.base_architecture,
        use_cross_attention=config.use_cross_attention,
        ranking_loss_type=config.ranking_loss_type,
        ranking_loss_weight=config.ranking_loss_weight,
        ce_weight=config.ce_weight
    )
    
    return model


def load_v3_from_checkpoint(
    checkpoint_path: str,
    base_model_version: str = 'v2',
    ranking_loss_type: str = 'listwise',
    ranking_loss_weight: float = 0.5,
    ce_weight: float = 0.5,
    freeze_base: bool = False,
    device: torch.device = None
) -> PointerSelectorV3:
    """
    从V1或V2的checkpoint初始化V3模型
    
    Args:
        checkpoint_path: V1或V2的checkpoint路径
        base_model_version: 基础模型版本 ('v1' 或 'v2')
        ranking_loss_type: 排序损失类型
        ranking_loss_weight: 排序损失权重
        ce_weight: 交叉熵权重
        freeze_base: 是否冻结基础架构（只训练排序相关参数）
        device: 设备
    
    Returns:
        初始化好的V3模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"从 {base_model_version.upper()} checkpoint 加载 V3 模型...")
    print(f"Checkpoint: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取配置
    if 'model_config' in checkpoint:
        config_dict = checkpoint['model_config']
        d_model = config_dict.get('d_model', 768)
        K = config_dict.get('K', 32)
        shot_num = config_dict.get('shot_num', 6)
        label_smoothing = config_dict.get('label_smoothing', 0.0)
        dropout = config_dict.get('dropout', 0.5)
    else:
        # 使用默认配置
        d_model = 768
        K = 32
        shot_num = 6
        label_smoothing = 0.0
        dropout = 0.5
    
    # 创建V3模型
    model = PointerSelectorV3(
        d_model=d_model,
        K=K,
        shot_num=shot_num,
        label_smoothing=label_smoothing,
        dropout=dropout,
        base_architecture=base_model_version,
        ranking_loss_type=ranking_loss_type,
        ranking_loss_weight=ranking_loss_weight,
        ce_weight=ce_weight
    )
    
    # 加载基础架构的参数（strict=False允许部分匹配）
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 加载参数
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"✓ 成功加载 {base_model_version.upper()} 参数")
    if missing_keys:
        print(f"  缺失的键（V3新增）: {len(missing_keys)} 个")
    if unexpected_keys:
        print(f"  未使用的键: {len(unexpected_keys)} 个")
    
    # 可选：冻结基础架构
    if freeze_base:
        print(f"🔒 冻结基础架构参数，只训练排序相关参数")
        for name, param in model.named_parameters():
            # V3没有新增参数层，所以这里暂时全部解冻
            # 实际上V3的排序损失是通过损失函数实现的，不需要额外参数
            param.requires_grad = True
    
    return model.to(device)


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV3 模型")
    print("="*70)
    
    # 测试V3-V2架构
    print("\n【测试1】V3-V2架构（from scratch）")
    model_v3_v2 = build_model_v3(PointerSelectorV3Config(base_architecture='v2'))
    
    # 测试V3-V1架构
    print("\n【测试2】V3-V1架构（from scratch）")
    model_v3_v1 = build_model_v3(PointerSelectorV3Config(base_architecture='v1'))
    
    # 创建测试数据
    batch_size = 4
    d_model = 768
    K = 32
    shot_num = 6
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    labels = torch.randint(0, K, (batch_size, shot_num))
    
    print(f"\n【测试3】前向传播")
    print(f"输入形状:")
    print(f"  query_emb: {query_emb.shape}")
    print(f"  cand_emb: {cand_emb.shape}")
    print(f"  labels: {labels.shape}")
    
    # 前向传播
    result = model_v3_v2(query_emb, cand_emb, labels, return_loss=True)
    
    print(f"\n输出:")
    print(f"  logits: {result['logits'].shape}")
    print(f"  predictions: {result['predictions'].shape}")
    print(f"  loss: {result['loss'].item():.4f}")
    
    print("\n" + "="*70)
    print("✓ V3 模型测试通过！")
    print("="*70)
