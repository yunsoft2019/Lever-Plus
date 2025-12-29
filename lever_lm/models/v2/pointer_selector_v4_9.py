"""
Pointer Selector V4-9: Two-Stage Coarse-to-Fine TopM 精排

特点：
- 第一阶段：cheap score（点积）快速筛选 TopM 候选
- 第二阶段：heavy refine（Cross-Attention / MLP）精排 TopM
- 主要利好效率，也可能提升鲁棒性（减少噪声候选干扰）
- 保持输出 [B, S, K] logits，完全兼容 RCE/GRPO

核心改动：
- 新增 refine_attn: 对 TopM 候选进行精排的 Cross-Attention
- 新增 refine_mlp: 可选的 MLP 精排头
- 每步先用点积选 TopM，再用 refine 模块精排

作者: Lever-Plus Team
日期: 2025-12-28
参考: Lever-Plus_PointerSelector_Upgrade_Plans_Keep_RCE_GRPO.md V4-9 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PointerSelectorV4_9(nn.Module):
    """
    V4-9 版本：Two-Stage Coarse-to-Fine TopM 精排
    
    第一阶段用 cheap score 快速筛选，第二阶段用 heavy refine 精排
    """
    
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 2,
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        num_heads: int = 1,
        attn_dropout: float = 0.1,
        num_layers: int = 1,
        top_m: int = 8,  # 精排的候选数量
        refine_type: str = "attn",  # "attn" 或 "mlp"
        use_gru: bool = True,  # 是否使用 GRU 状态更新
        use_step_emb: bool = True,  # 是否使用 step embedding
    ):
        """
        初始化 V4-9 模型
        
        Args:
            d_model: 输入 embedding 维度 (默认 768, CLIP ViT-L/14 输出)
            K: 候选池大小 (默认 32)
            shot_num: 需要选择的样本数量 (默认 2)
            label_smoothing: 标签平滑系数 (默认 0.1)
            dropout: dropout 比例 (默认 0.1)
            hidden_dim: 输出维度 (默认 256)
            num_heads: Cross-Attention 的头数 (默认 1)
            attn_dropout: Attention 层的 dropout (默认 0.1)
            num_layers: Query-Candidate Cross-Attention 的层数 (默认 1)
            top_m: 精排的候选数量 (默认 8)
            refine_type: 精排类型，"attn" 或 "mlp" (默认 "attn")
            use_gru: 是否使用 GRU 状态更新 (默认 True)
            use_step_emb: 是否使用 step embedding (默认 True)
        """
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.num_layers = num_layers
        self.top_m = top_m
        self.refine_type = refine_type
        self.use_gru = use_gru
        self.use_step_emb = use_step_emb
        
        # 投影层
        if d_model != hidden_dim:
            self.input_proj = nn.Linear(d_model, hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()
        
        # 多层 Cross-Attention（Query-Candidate）
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer Normalization for Cross-Attention
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 投影层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 【V4-9 核心】精排模块
        if refine_type == "attn":
            # Cross-Attention 精排
            self.refine_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            self.refine_norm = nn.LayerNorm(hidden_dim)
            self.refine_score = nn.Linear(hidden_dim, 1, bias=True)
        else:
            # MLP 精排
            self.refine_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        # GRU 状态更新（可选）
        if use_gru:
            self.decoder_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Step embedding（可选）
        if use_step_emb:
            self.step_emb = nn.Embedding(shot_num, hidden_dim)
        
        # 温度参数
        self.temperature = torch.tensor([0.1], dtype=torch.float32)
        
        # 初始化权重
        self._init_weights()
        
        print(f"✓ PointerSelectorV4_9 初始化完成")
        print(f"  - d_model (输入): {d_model} -> hidden_dim (输出): {hidden_dim}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - top_m: {top_m}")
        print(f"  - refine_type: {refine_type}")
        print(f"  - use_gru: {use_gru}")
        print(f"  - use_step_emb: {use_step_emb}")
        print(f"  - 架构: CLIP {d_model} → proj → {hidden_dim} + Two-Stage (TopM={top_m}, {refine_type})")
    
    def _init_weights(self):
        """初始化模型权重"""
        if not isinstance(self.input_proj, nn.Identity):
            nn.init.xavier_uniform_(self.input_proj.weight)
        
        # query/cand 投影层使用接近单位矩阵的初始化
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.cand_proj.weight)
        
        with torch.no_grad():
            self.query_proj.weight.add_(torch.randn_like(self.query_proj.weight) * 0.01)
            self.cand_proj.weight.add_(torch.randn_like(self.cand_proj.weight) * 0.01)
        
        # Step embedding 使用正态分布初始化
        if self.use_step_emb:
            nn.init.normal_(self.step_emb.weight, mean=0.0, std=0.02)
    
    def _refine_scores(
        self,
        h: torch.Tensor,
        cand_sub: torch.Tensor,
    ) -> torch.Tensor:
        """
        精排 TopM 候选
        
        Args:
            h: [B, H] 当前状态
            cand_sub: [B, M, H] TopM 候选
        
        Returns:
            refined_scores: [B, M] 精排后的分数
        """
        batch_size = h.shape[0]
        M = cand_sub.shape[1]
        
        if self.refine_type == "attn":
            # Cross-Attention 精排
            h_expanded = h.unsqueeze(1)  # [B, 1, H]
            attn_out, _ = self.refine_attn(
                query=h_expanded,
                key=cand_sub,
                value=cand_sub
            )  # [B, 1, H]
            
            # 计算每个候选的精排分数
            # 方法：用 attention 后的 query 与每个候选计算相似度
            refined_query = self.refine_norm(attn_out + h_expanded)  # [B, 1, H]
            refined_query = F.normalize(refined_query, p=2, dim=-1)
            cand_sub_norm = F.normalize(cand_sub, p=2, dim=-1)
            
            # 点积打分
            refined_scores = torch.bmm(refined_query, cand_sub_norm.transpose(1, 2)).squeeze(1)  # [B, M]
            
            # 或者用 refine_score 线性层
            # combined = refined_query.expand(-1, M, -1) * cand_sub  # [B, M, H]
            # refined_scores = self.refine_score(combined).squeeze(-1)  # [B, M]
        else:
            # MLP 精排
            h_expanded = h.unsqueeze(1).expand(-1, M, -1)  # [B, M, H]
            combined = torch.cat([h_expanded, cand_sub], dim=-1)  # [B, M, 2H]
            refined_scores = self.refine_mlp(combined).squeeze(-1)  # [B, M]
        
        return refined_scores
    
    def forward(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> dict:
        """
        前向传播
        
        Args:
            query_emb: [B, d] query embedding
            cand_emb: [B, K, d] 候选 embedding
            labels: [B, S] 标签序列（训练时需要）
            return_loss: 是否返回损失
        
        Returns:
            dict: {
                'logits': [B, S, K] 每步的 logits,
                'predictions': [B, S] 预测序列,
                'loss': scalar (如果 return_loss=True)
            }
        """
        batch_size = query_emb.shape[0]
        device = query_emb.device
        
        input_dim = query_emb.shape[-1]
        actual_K = cand_emb.shape[1]
        
        # 动态调整 top_m（不能超过 actual_K）
        effective_top_m = min(self.top_m, actual_K)
        
        # 步骤1：投影到 hidden_dim
        query_reduced = self.input_proj(query_emb)
        cand_reduced = self.input_proj(cand_emb.reshape(-1, input_dim))
        cand_reduced = cand_reduced.reshape(batch_size, actual_K, self.hidden_dim)
        
        # 步骤2：多层 Cross-Attention 增强 query
        query_for_attn = query_reduced.unsqueeze(1)
        
        for layer_idx in range(self.num_layers):
            attn_output, _ = self.cross_attn_layers[layer_idx](
                query=query_for_attn,
                key=cand_reduced,
                value=cand_reduced
            )
            query_for_attn = self.attn_norms[layer_idx](attn_output + query_for_attn)
        
        query_enhanced = query_for_attn.squeeze(1)  # [B, H]
        
        # 步骤3：Dropout + 投影
        query_proj = self.dropout(query_enhanced)
        cand_proj_out = self.dropout(cand_reduced)
        
        query_proj = self.query_proj(query_proj)  # [B, H]
        cand_proj_out = self.cand_proj(cand_proj_out)  # [B, K, H]
        
        # 步骤4：L2 归一化
        h = F.normalize(query_proj, p=2, dim=-1)  # [B, H] - 当前状态
        cand_proj_norm = F.normalize(cand_proj_out, p=2, dim=-1)  # [B, K, H]
        
        # 温度参数
        temperature = self.temperature.to(device)
        
        # 存储每步的 logits 和预测
        all_logits = []
        predictions = []
        
        # mask：记录已选择的候选
        selected_mask = torch.zeros(batch_size, actual_K, dtype=torch.bool, device=device)
        
        # 多步选择
        for step in range(self.shot_num):
            # 添加 step embedding（可选）
            if self.use_step_emb:
                step_idx = min(step, self.step_emb.num_embeddings - 1)
                step_emb = self.step_emb(torch.tensor(step_idx, device=device))
                h_step = F.normalize(h + step_emb, p=2, dim=-1)
            else:
                h_step = h
            
            # 【V4-9 核心】第一阶段：cheap score（点积）
            cheap_scores = torch.bmm(h_step.unsqueeze(1), cand_proj_norm.transpose(1, 2)).squeeze(1)  # [B, K]
            cheap_scores = cheap_scores / temperature
            cheap_scores = cheap_scores.masked_fill(selected_mask, -100.0)
            
            # 选择 TopM 候选
            top_values, top_indices = cheap_scores.topk(effective_top_m, dim=-1)  # [B, M]
            
            # 获取 TopM 候选的 embedding
            cand_sub = cand_proj_norm.gather(
                1, 
                top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )  # [B, M, H]
            
            # 【V4-9 核心】第二阶段：heavy refine
            refined_scores = self._refine_scores(h_step, cand_sub)  # [B, M]
            refined_scores = refined_scores / temperature
            
            # 将精排分数 scatter 回全量 K
            scores = torch.full((batch_size, actual_K), -100.0, device=device)
            scores = scores.scatter(1, top_indices, refined_scores)
            
            # 确保已选择的候选被 mask
            scores = scores.masked_fill(selected_mask, -100.0)
            
            all_logits.append(scores)
            
            # 预测
            pred = scores.argmax(dim=-1)
            predictions.append(pred)
            
            # 确定本步使用的索引（Teacher Forcing）
            if labels is not None and step < labels.shape[1]:
                idx = labels[:, step]
            else:
                idx = pred
            
            # 更新 mask
            selected_mask = selected_mask.scatter(1, idx.unsqueeze(1), True)
            
            # 获取被选中的候选 embedding
            chosen = cand_proj_norm.gather(
                1,
                idx.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)
            ).squeeze(1)  # [B, H]
            
            # 更新状态
            if self.use_gru:
                h = self.decoder_gru(chosen, h)
                h = F.normalize(h, p=2, dim=-1)
            else:
                # 简单的加权融合
                alpha = 0.6
                h = F.normalize(alpha * h + (1 - alpha) * chosen, p=2, dim=-1)
        
        # 堆叠结果
        all_logits = torch.stack(all_logits, dim=1)  # [B, S, K]
        predictions = torch.stack(predictions, dim=1)  # [B, S]
        
        result = {
            'logits': all_logits,
            'predictions': predictions
        }
        
        if return_loss and labels is not None:
            loss = self.compute_loss(all_logits, labels)
            result['loss'] = loss
        
        return result
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算损失函数"""
        batch_size, shot_num, K = logits.shape
        
        logits_flat = logits.reshape(-1, K)
        labels_flat = labels.reshape(-1)
        
        logits_flat = torch.clamp(logits_flat, min=-100.0)
        
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            label_smoothing=self.label_smoothing
        )
        
        return loss
    
    def predict(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        top_k: int = 1,
        shot_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理模式"""
        self.eval()
        
        original_shot_num = self.shot_num
        if shot_num is not None:
            self.shot_num = shot_num
        
        with torch.no_grad():
            result = self.forward(query_emb, cand_emb, labels=None, return_loss=False)
            predictions = result['predictions']
            logits = result['logits']
            scores = logits.max(dim=-1)[0]
        
        self.shot_num = original_shot_num
        
        return predictions, scores


class PointerSelectorV4_9Config:
    """V4-9 模型配置类"""
    
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 2,
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        num_heads: int = 1,
        attn_dropout: float = 0.1,
        num_layers: int = 1,
        top_m: int = 8,
        refine_type: str = "attn",
        use_gru: bool = True,
        use_step_emb: bool = True,
    ):
        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.num_layers = num_layers
        self.top_m = top_m
        self.refine_type = refine_type
        self.use_gru = use_gru
        self.use_step_emb = use_step_emb
    
    def to_dict(self):
        return {
            'd_model': self.d_model,
            'K': self.K,
            'shot_num': self.shot_num,
            'label_smoothing': self.label_smoothing,
            'dropout': self.dropout,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'attn_dropout': self.attn_dropout,
            'num_layers': self.num_layers,
            'top_m': self.top_m,
            'refine_type': self.refine_type,
            'use_gru': self.use_gru,
            'use_step_emb': self.use_step_emb,
        }


def build_model_v4_9(config: Optional[PointerSelectorV4_9Config] = None) -> PointerSelectorV4_9:
    """构建 V4-9 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_9Config()
    
    model = PointerSelectorV4_9(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        attn_dropout=config.attn_dropout,
        num_layers=config.num_layers,
        top_m=config.top_m,
        refine_type=config.refine_type,
        use_gru=config.use_gru,
        use_step_emb=config.use_step_emb,
    )
    
    return model
