"""
Pointer Selector V4-10: STOP 自适应 shot

特点：
- 添加可学习的 STOP token，让模型自己决定何时停止选择
- 解决"shot 越多越伤"的问题
- 当模型认为已选够 demo 时，可以提前停止
- 保持输出 [B, S, K+1] logits，完全兼容 RCE/GRPO

核心改动：
- 新增 stop_token: 可学习的 STOP embedding
- forward 中支持 STOP 逻辑：一旦选择 STOP，后续 step 强制选 STOP
- 推理时可以根据 STOP 提前终止

作者: Lever-Plus Team
日期: 2025-12-29
参考: Lever-Plus_PointerSelector_Upgrade_Plans_Keep_RCE_GRPO.md V4-10 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PointerSelectorV4_10(nn.Module):
    """
    V4-10 版本：STOP 自适应 shot
    
    添加可学习的 STOP token，让模型自己决定何时停止选择
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
        use_gru: bool = True,
        use_step_emb: bool = True,
    ):
        """
        初始化 V4-10 模型
        
        Args:
            d_model: 输入 embedding 维度 (默认 768, CLIP ViT-L/14 输出)
            K: 候选池大小 (默认 32)，实际会扩展为 K+1（包含 STOP）
            shot_num: 最大选择的样本数量 (默认 2)
            label_smoothing: 标签平滑系数 (默认 0.1)
            dropout: dropout 比例 (默认 0.1)
            hidden_dim: 输出维度 (默认 256)
            num_heads: Cross-Attention 的头数 (默认 1)
            attn_dropout: Attention 层的 dropout (默认 0.1)
            num_layers: Query-Candidate Cross-Attention 的层数 (默认 1)
            use_gru: 是否使用 GRU 状态更新 (默认 True)
            use_step_emb: 是否使用 step embedding (默认 True)
        """
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.K = K  # 原始候选数，不包含 STOP
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.num_layers = num_layers
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
        
        # 【V4-10 核心】可学习的 STOP token
        # 在 hidden_dim 空间中的 STOP embedding
        self.stop_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
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
        
        print(f"✓ PointerSelectorV4_10 初始化完成")
        print(f"  - d_model (输入): {d_model} -> hidden_dim (输出): {hidden_dim}")
        print(f"  - K (候选池大小): {K} + 1 (STOP) = {K+1}")
        print(f"  - shot_num (最大): {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - use_gru: {use_gru}")
        print(f"  - use_step_emb: {use_step_emb}")
        print(f"  - 架构: CLIP {d_model} → proj → {hidden_dim} + STOP token")
    
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
        
        # STOP token 初始化为较小的随机值
        nn.init.normal_(self.stop_token, mean=0.0, std=0.02)
    
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
            cand_emb: [B, K, d] 候选 embedding（不包含 STOP）
            labels: [B, S] 标签序列（训练时需要，可以包含 STOP index = K）
            return_loss: 是否返回损失
        
        Returns:
            dict: {
                'logits': [B, S, K+1] 每步的 logits（包含 STOP）,
                'predictions': [B, S] 预测序列,
                'loss': scalar (如果 return_loss=True),
                'stopped_at': [B] 每个样本停止的 step（-1 表示未停止）
            }
        """
        batch_size = query_emb.shape[0]
        device = query_emb.device
        
        input_dim = query_emb.shape[-1]
        actual_K = cand_emb.shape[1]  # 实际候选数（不含 STOP）
        K_plus_1 = actual_K + 1  # 包含 STOP
        stop_idx = actual_K  # STOP 的索引
        
        # 步骤1：投影到 hidden_dim
        query_reduced = self.input_proj(query_emb)
        cand_reduced = self.input_proj(cand_emb.reshape(-1, input_dim))
        cand_reduced = cand_reduced.reshape(batch_size, actual_K, self.hidden_dim)
        
        # 步骤2：多层 Cross-Attention 增强 query（只用原始候选，不含 STOP）
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
        
        # 【V4-10 核心】添加 STOP token 到候选池
        stop_proj = F.normalize(self.stop_token, p=2, dim=-1)  # [1, H]
        stop_proj_expanded = stop_proj.expand(batch_size, 1, -1)  # [B, 1, H]
        cand_with_stop = torch.cat([cand_proj_norm, stop_proj_expanded], dim=1)  # [B, K+1, H]
        
        # 温度参数
        temperature = self.temperature.to(device)
        
        # 存储每步的 logits 和预测
        all_logits = []
        predictions = []
        
        # mask：记录已选择的候选（包含 STOP 位置）
        selected_mask = torch.zeros(batch_size, K_plus_1, dtype=torch.bool, device=device)
        
        # 【V4-10 核心】记录是否已经 ended
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
        stopped_at = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        
        # 调试：记录每步的选择
        debug_step_choices = []
        debug_stop_scores = []
        
        # 多步选择
        for step in range(self.shot_num):
            # 添加 step embedding（可选）
            if self.use_step_emb:
                step_idx = min(step, self.step_emb.num_embeddings - 1)
                step_emb = self.step_emb(torch.tensor(step_idx, device=device))
                h_step = F.normalize(h + step_emb, p=2, dim=-1)
            else:
                h_step = h
            
            # 计算分数（包含 STOP）
            scores = torch.bmm(h_step.unsqueeze(1), cand_with_stop.transpose(1, 2)).squeeze(1)  # [B, K+1]
            scores = scores / temperature
            scores = scores.masked_fill(selected_mask, -100.0)
            
            # 【V4-10 核心】如果已经 ended，只允许选 STOP
            if ended.any():
                # 创建一个只有 STOP 位置为 0，其他为 -100 的 tensor
                stop_only_scores = torch.full_like(scores, -100.0)
                stop_only_scores[:, stop_idx] = 0.0
                
                # 对于已经 ended 的样本，使用 stop_only_scores
                scores = torch.where(
                    ended.unsqueeze(1).expand_as(scores),
                    stop_only_scores,
                    scores
                )
            
            all_logits.append(scores)
            
            # 调试：记录 STOP token 的分数
            stop_score = scores[:, stop_idx].mean().item()
            max_cand_score = scores[:, :stop_idx].max(dim=-1)[0].mean().item()
            debug_stop_scores.append({
                'step': step,
                'stop_score': stop_score,
                'max_cand_score': max_cand_score,
                'stop_higher': stop_score > max_cand_score
            })
            
            # 预测
            pred = scores.argmax(dim=-1)
            predictions.append(pred)
            
            # 确定本步使用的索引（Teacher Forcing）
            if labels is not None and step < labels.shape[1]:
                idx = labels[:, step]
            else:
                idx = pred
            
            # 【V4-10 核心】更新 ended 状态
            newly_ended = (idx == stop_idx) & (~ended)
            stopped_at = torch.where(newly_ended, torch.tensor(step, device=device), stopped_at)
            ended = ended | (idx == stop_idx)
            
            # 更新 mask（STOP 可以被多次选择，所以不 mask）
            # 但普通候选只能选一次
            mask_update = idx.unsqueeze(1)
            # 只对非 STOP 的选择更新 mask
            non_stop_mask = (idx != stop_idx).unsqueeze(1)
            selected_mask = selected_mask.scatter(
                1, 
                mask_update, 
                non_stop_mask.expand_as(mask_update)
            )
            
            # 获取被选中的候选 embedding
            # 对于 STOP，使用 stop_token
            chosen = cand_with_stop.gather(
                1,
                idx.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)
            ).squeeze(1)  # [B, H]
            
            # 更新状态（只对未 ended 的样本更新）
            if self.use_gru:
                h_new = self.decoder_gru(chosen, h)
                h_new = F.normalize(h_new, p=2, dim=-1)
                # 对于已经 ended 的样本，保持状态不变
                h = torch.where(ended.unsqueeze(1).expand_as(h), h, h_new)
            else:
                alpha = 0.6
                h_new = F.normalize(alpha * h + (1 - alpha) * chosen, p=2, dim=-1)
                h = torch.where(ended.unsqueeze(1).expand_as(h), h, h_new)
        
        # 堆叠结果
        all_logits = torch.stack(all_logits, dim=1)  # [B, S, K+1]
        predictions = torch.stack(predictions, dim=1)  # [B, S]
        
        result = {
            'logits': all_logits,
            'predictions': predictions,
            'stopped_at': stopped_at,
            'stop_idx': stop_idx,
            'debug_stop_scores': debug_stop_scores  # 调试信息
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
        batch_size, shot_num, K_plus_1 = logits.shape
        
        logits_flat = logits.reshape(-1, K_plus_1)
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
        shot_num: Optional[int] = None,
        return_early_stop: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        推理模式
        
        Args:
            query_emb: [B, d] query embedding
            cand_emb: [B, K, d] 候选 embedding
            top_k: 返回 top-k 预测
            shot_num: 覆盖默认的 shot_num
            return_early_stop: 是否返回提前停止信息
        
        Returns:
            predictions: [B, S] 预测序列
            scores: [B, S] 每步的最高分数
            stopped_at: [B] 每个样本停止的 step（-1 表示未停止）
        """
        self.eval()
        
        original_shot_num = self.shot_num
        if shot_num is not None:
            self.shot_num = shot_num
        
        with torch.no_grad():
            result = self.forward(query_emb, cand_emb, labels=None, return_loss=False)
            predictions = result['predictions']
            logits = result['logits']
            scores = logits.max(dim=-1)[0]
            stopped_at = result['stopped_at']
        
        self.shot_num = original_shot_num
        
        if return_early_stop:
            return predictions, scores, stopped_at
        else:
            return predictions, scores


class PointerSelectorV4_10Config:
    """V4-10 模型配置类"""
    
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
            'use_gru': self.use_gru,
            'use_step_emb': self.use_step_emb,
        }


def build_model_v4_10(config: Optional[PointerSelectorV4_10Config] = None) -> PointerSelectorV4_10:
    """构建 V4-10 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_10Config()
    
    model = PointerSelectorV4_10(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        attn_dropout=config.attn_dropout,
        num_layers=config.num_layers,
        use_gru=config.use_gru,
        use_step_emb=config.use_step_emb,
    )
    
    return model
