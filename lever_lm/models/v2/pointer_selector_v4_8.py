"""
Pointer Selector V4-8: Slot/Set Decoder（并行 slots 协同）

特点：
- 同时维护 S 个"slot"，slots 之间 self-attn 协同分工
- 每个 slot 生成一行 logits，实现并行预测
- 仍然按 step 输出 logits，兼容现有训练流程
- 推理时使用 greedy mask 保证不重复

核心改动：
- 新增 slot_emb: [S, H] 每个 slot 一个 learnable embedding
- 新增 slot_self_attn: slots 之间的 self-attention
- 新增 slot_cand_attn: slots 对 candidates 的 cross-attention（可选）
- 并行计算所有 slot 的 logits，然后按 step 应用 mask

作者: Lever-Plus Team
日期: 2025-12-28
参考: Lever-Plus_PointerSelector_Upgrade_Plans_Keep_RCE_GRPO.md V4-8 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PointerSelectorV4_8(nn.Module):
    """
    V4-8 版本：Slot/Set Decoder（并行 slots 协同）
    
    与其自回归一步步挑，不如同时维护 S 个"slot"，slots 之间 self-attn 协同分工
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
        num_slot_layers: int = 2,  # slot self-attention 层数
        use_slot_cand_attn: bool = True,  # 是否使用 slot-candidate cross-attention
    ):
        """
        初始化 V4-8 模型
        
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
            num_slot_layers: Slot Self-Attention 的层数 (默认 2)
            use_slot_cand_attn: 是否使用 slot-candidate cross-attention (默认 True)
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
        self.num_slot_layers = num_slot_layers
        self.use_slot_cand_attn = use_slot_cand_attn
        
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
        
        # 【V4-8 核心】Slot Embeddings
        # 支持最多 8 个 slots（推理时可能需要更多 shot）
        self.max_shot_num = 8
        self.slot_emb = nn.Embedding(self.max_shot_num, hidden_dim)
        
        # 【V4-8 核心】Slot Self-Attention（多层）
        self.slot_self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            for _ in range(num_slot_layers)
        ])
        
        self.slot_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_slot_layers)
        ])
        
        # 【V4-8 可选】Slot-Candidate Cross-Attention
        if use_slot_cand_attn:
            self.slot_cand_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            self.slot_cand_norm = nn.LayerNorm(hidden_dim)
        
        # 温度参数
        self.temperature = torch.tensor([0.1], dtype=torch.float32)
        
        # 初始化权重
        self._init_weights()
        
        print(f"✓ PointerSelectorV4_8 初始化完成")
        print(f"  - d_model (输入): {d_model} -> hidden_dim (输出): {hidden_dim}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers (query-cand): {num_layers}")
        print(f"  - num_slot_layers: {num_slot_layers}")
        print(f"  - use_slot_cand_attn: {use_slot_cand_attn}")
        print(f"  - 架构: CLIP {d_model} → proj → {hidden_dim} + Slot Decoder ({num_slot_layers} layers)")
    
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
        
        # Slot embedding 使用正态分布初始化
        nn.init.normal_(self.slot_emb.weight, mean=0.0, std=0.02)
    
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
        query_proj = F.normalize(query_proj, p=2, dim=-1)
        cand_proj_norm = F.normalize(cand_proj_out, p=2, dim=-1)
        
        # 【V4-8 核心】初始化 slots
        # 推理时可能需要更多的 shot，使用 labels 的长度或 self.shot_num
        actual_shot_num = labels.shape[1] if labels is not None else self.shot_num
        
        # 获取 slot embeddings（限制在 max_shot_num 内）
        slot_indices = torch.arange(min(actual_shot_num, self.max_shot_num), device=device)
        slot_embs = self.slot_emb(slot_indices)  # [S, H]
        
        # 如果 actual_shot_num > max_shot_num，复用最后一个 slot embedding
        if actual_shot_num > self.max_shot_num:
            extra_slots = actual_shot_num - self.max_shot_num
            last_slot = slot_embs[-1:].expand(extra_slots, -1)
            slot_embs = torch.cat([slot_embs, last_slot], dim=0)
        
        # slots = query + slot_emb
        slots = query_proj.unsqueeze(1) + slot_embs.unsqueeze(0)  # [B, S, H]
        
        # 【V4-8 核心】Slot Self-Attention（多层）
        for layer_idx in range(self.num_slot_layers):
            attn_out, _ = self.slot_self_attn_layers[layer_idx](
                query=slots,
                key=slots,
                value=slots
            )
            slots = self.slot_norms[layer_idx](slots + attn_out)
        
        # 【V4-8 可选】Slot-Candidate Cross-Attention
        if self.use_slot_cand_attn:
            slot_cand_out, _ = self.slot_cand_attn(
                query=slots,
                key=cand_proj_norm,
                value=cand_proj_norm
            )
            slots = self.slot_cand_norm(slots + slot_cand_out)
        
        # L2 归一化 slots
        slots = F.normalize(slots, p=2, dim=-1)  # [B, S, H]
        
        # 温度参数
        temperature = self.temperature.to(device)
        
        # 【V4-8 核心】并行计算所有 slot 的 logits
        # logits[b, s, k] = slots[b, s, :] · cand_proj_norm[b, k, :] / temperature
        all_logits_raw = torch.einsum("bsh,bkh->bsk", slots, cand_proj_norm) / temperature  # [B, S, K]
        
        # 存储每步的 logits 和预测（应用 mask 后）
        all_logits = []
        predictions = []
        
        # mask：记录已选择的候选
        selected_mask = torch.zeros(batch_size, actual_K, dtype=torch.bool, device=device)
        
        # 按 step 应用 mask（保证不重复）
        for step in range(actual_shot_num):
            # 获取当前 slot 的 logits
            scores = all_logits_raw[:, step, :].clone()  # [B, K]
            
            # 应用 mask
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
        """推理模式
        
        Args:
            query_emb: [B, d] query embedding
            cand_emb: [B, K, d] 候选 embedding
            top_k: 返回 top-k 个预测（未使用）
            shot_num: 需要选择的样本数量，如果为 None 则使用 self.shot_num
        
        Returns:
            predictions: [B, shot_num] 预测的候选索引
            scores: [B, shot_num] 预测的分数
        """
        self.eval()
        
        # 临时修改 shot_num
        original_shot_num = self.shot_num
        if shot_num is not None:
            self.shot_num = shot_num
        
        with torch.no_grad():
            result = self.forward(query_emb, cand_emb, labels=None, return_loss=False)
            predictions = result['predictions']
            logits = result['logits']
            scores = logits.max(dim=-1)[0]
        
        # 恢复原始 shot_num
        self.shot_num = original_shot_num
        
        return predictions, scores


class PointerSelectorV4_8Config:
    """V4-8 模型配置类"""
    
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
        num_slot_layers: int = 2,
        use_slot_cand_attn: bool = True,
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
        self.num_slot_layers = num_slot_layers
        self.use_slot_cand_attn = use_slot_cand_attn
    
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
            'num_slot_layers': self.num_slot_layers,
            'use_slot_cand_attn': self.use_slot_cand_attn,
        }


def build_model_v4_8(config: Optional[PointerSelectorV4_8Config] = None) -> PointerSelectorV4_8:
    """构建 V4-8 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_8Config()
    
    model = PointerSelectorV4_8(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        attn_dropout=config.attn_dropout,
        num_layers=config.num_layers,
        num_slot_layers=config.num_slot_layers,
        use_slot_cand_attn=config.use_slot_cand_attn,
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV4_8 模型")
    print("="*70)
    
    batch_size = 4
    d_model = 768
    K = 32
    shot_num = 4  # 测试多 shot 场景
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    labels = torch.randint(0, K, (batch_size, shot_num))
    
    # 测试不同的 num_slot_layers
    for num_slot_layers in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"测试 num_slot_layers = {num_slot_layers}")
        print("="*70)
        
        config = PointerSelectorV4_8Config(
            shot_num=shot_num,
            num_slot_layers=num_slot_layers
        )
        model = build_model_v4_8(config)
        
        print(f"\n输入形状:")
        print(f"  query_emb: {query_emb.shape}")
        print(f"  cand_emb: {cand_emb.shape}")
        print(f"  labels: {labels.shape}")
        
        print(f"\n前向传播...")
        result = model(query_emb, cand_emb, labels, return_loss=True)
        
        print(f"\n输出:")
        print(f"  logits: {result['logits'].shape}")
        print(f"  predictions: {result['predictions'].shape}")
        print(f"  loss: {result['loss'].item():.4f}")
        
        print(f"\n推理模式...")
        predictions, scores = model.predict(query_emb, cand_emb)
        print(f"  predictions: {predictions.shape}")
        print(f"  scores: {scores.shape}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n参数量: {total_params:,}")
    
    print("\n" + "="*70)
    print("✓ V4-8 模型测试通过！")
    print("="*70)
