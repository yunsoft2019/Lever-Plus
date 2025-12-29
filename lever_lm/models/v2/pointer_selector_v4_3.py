"""
Pointer Selector V4-3: Cross-Attn + GRU Pointer Decoder + Learnable MMR 多样性残差

特点：
- 在 V4-2 基础上添加 Learnable MMR (Maximum Marginal Relevance) 多样性残差
- 专门解决 shot≥3 时的冗余问题
- 每步选择时不只看"和 query 的相关性"，还惩罚"和已选集合的相似度"
- λ 做成可学习（per-step），让模型自动学习每步需要多少多样性

核心改动：
- 新增 div_lambda: 可学习的多样性权重（per-step）
- 每步计算 redundancy = max(cosine(cand, selected_embs))
- scores = base_scores - relu(div_lambda[step]) * redundancy

作者: Lever-Plus Team
日期: 2025-12-26
参考: Lever-Plus_PointerSelector_Upgrade_Plans_Keep_RCE_GRPO.md V4-3 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PointerSelectorV4_3(nn.Module):
    """
    V4-3 版本：Cross-Attn + GRU Pointer Decoder + Learnable MMR
    
    在 V4-2 基础上添加 MMR 多样性残差，专治 shot≥3 冗余问题
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
        use_step_emb: bool = True,
        use_gru: bool = True,  # 是否使用 GRU（可以关闭来测试纯 MMR 效果）
        mmr_reduction: str = 'max'  # 'max' 或 'mean'，max 更像"去重"
    ):
        """
        初始化 V4-3 模型
        
        Args:
            d_model: 输入 embedding 维度 (默认 768, CLIP ViT-L/14 输出)
            K: 候选池大小 (默认 32)
            shot_num: 需要选择的样本数量 (默认 2)
            label_smoothing: 标签平滑系数 (默认 0.1)
            dropout: dropout 比例 (默认 0.1)
            hidden_dim: 输出维度 (默认 256)
            num_heads: Cross-Attention 的头数 (默认 1)
            attn_dropout: Attention 层的 dropout (默认 0.1)
            num_layers: Cross-Attention 的层数 (默认 1)
            use_step_emb: 是否使用 step embedding (默认 True)
            use_gru: 是否使用 GRU decoder (默认 True)
            mmr_reduction: MMR 冗余计算方式 (默认 'max')
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
        self.use_step_emb = use_step_emb
        self.use_gru = use_gru
        self.mmr_reduction = mmr_reduction
        
        # 投影层
        if d_model != hidden_dim:
            self.input_proj = nn.Linear(d_model, hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()
        
        # 多层 Cross-Attention
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer Normalization
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 投影层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 【V4-2】GRU Decoder（可选）
        if use_gru:
            self.decoder_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 【V4-2】Step Embedding（可选）
        if use_step_emb:
            self.step_emb = nn.Embedding(shot_num, hidden_dim)
        
        # 【V4-3 核心】Learnable MMR 多样性权重（per-step）
        # 初始化为小正值（0.1），让 MMR 项有初始贡献，可以学习
        # 使用 softplus 代替 relu，保证梯度始终存在
        self.div_lambda = nn.Parameter(torch.full((shot_num,), 0.1))
        
        # 温度参数
        self.temperature = torch.tensor([0.1], dtype=torch.float32)
        
        # 初始化权重
        self._init_weights()
        
        print(f"✓ PointerSelectorV4_3 初始化完成")
        print(f"  - d_model (输入): {d_model} -> hidden_dim (输出): {hidden_dim}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - use_step_emb: {use_step_emb}")
        print(f"  - use_gru: {use_gru}")
        print(f"  - mmr_reduction: {mmr_reduction}")
        print(f"  - 架构: CLIP {d_model} → proj → {hidden_dim} + {num_layers}层 Cross-Attention + GRU + MMR")
    
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
        
        # div_lambda 初始化为 0（等价于原模型）
        # 已在定义时初始化
    
    def compute_redundancy(
        self,
        cand_proj_norm: torch.Tensor,
        selected_embs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算每个候选与已选集合的冗余度
        
        Args:
            cand_proj_norm: [B, K, H] 候选 embedding（已归一化）
            selected_embs: List of [B, H] 已选 embedding
        
        Returns:
            redundancy: [B, K] 冗余度分数
        """
        if len(selected_embs) == 0:
            return torch.zeros(
                cand_proj_norm.shape[0], 
                cand_proj_norm.shape[1], 
                device=cand_proj_norm.device
            )
        
        # 堆叠已选 embedding: [B, num_selected, H]
        sel = torch.stack(selected_embs, dim=1)
        
        # 计算 cosine 相似度（因为都已归一化，点积就是 cosine）
        # cand_proj_norm: [B, K, H], sel: [B, num_selected, H]
        # sim: [B, K, num_selected]
        sim = torch.einsum("bkh,bth->bkt", cand_proj_norm, sel)
        
        # 根据 reduction 方式计算冗余度
        if self.mmr_reduction == 'max':
            # max 更像"去重"：只要和任一已选很像就惩罚
            redundancy = sim.max(dim=-1).values  # [B, K]
        else:
            # mean：平均相似度
            redundancy = sim.mean(dim=-1)  # [B, K]
        
        return redundancy
    
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
        
        # 步骤2：多层 Cross-Attention 增强
        query_for_attn = query_reduced.unsqueeze(1)
        
        for layer_idx in range(self.num_layers):
            attn_output, _ = self.cross_attn_layers[layer_idx](
                query=query_for_attn,
                key=cand_reduced,
                value=cand_reduced
            )
            query_for_attn = self.attn_norms[layer_idx](attn_output + query_for_attn)
        
        query_enhanced = query_for_attn.squeeze(1)
        
        # 步骤3：Dropout
        query_proj = self.dropout(query_enhanced)
        cand_proj_out = self.dropout(cand_reduced)
        
        # 步骤4：投影层
        query_proj = self.query_proj(query_proj)
        cand_proj_out = self.cand_proj(cand_proj_out)
        
        # 步骤5：L2 归一化
        h = F.normalize(query_proj, p=2, dim=-1)
        cand_proj_norm = F.normalize(cand_proj_out, p=2, dim=-1)
        
        # 存储每步的 logits 和预测
        all_logits = []
        predictions = []
        
        # 【V4-3】存储已选 embedding（用于计算冗余度）
        selected_embs: List[torch.Tensor] = []
        
        # mask：记录已选择的候选
        selected_mask = torch.zeros(batch_size, actual_K, dtype=torch.bool, device=device)
        
        # 自回归生成 shot_num 步
        for step in range(self.shot_num):
            # 【V4-2】可选：添加 step embedding
            if self.use_step_emb:
                step_tensor = torch.tensor([step], device=device)
                h_step = h + self.step_emb(step_tensor)
                h_step = F.normalize(h_step, p=2, dim=-1)
            else:
                h_step = h
            
            # 计算基础注意力分数（相关性）
            base_scores = torch.matmul(h_step.unsqueeze(1), cand_proj_norm.transpose(1, 2))
            temperature = self.temperature.to(device)
            base_scores = base_scores.squeeze(1) / temperature  # [B, K]
            
            # 【V4-3 核心】计算 MMR 多样性残差
            if step > 0:
                # 计算冗余度
                redundancy = self.compute_redundancy(cand_proj_norm, selected_embs)  # [B, K]
                
                # 使用 softplus 保证非负且梯度始终存在（比 relu 更好）
                # softplus(x) = log(1 + exp(x))，当 x=0.1 时约为 0.74
                div_weight = F.softplus(self.div_lambda[step])
                
                # 从基础分数中减去冗余惩罚
                scores = base_scores - div_weight * redundancy
            else:
                scores = base_scores
            
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
            
            # 取出被选 demo 的 embedding
            chosen = cand_proj_norm.gather(
                1, idx.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)
            ).squeeze(1)  # [B, H]
            
            # 【V4-3】记录已选 embedding
            selected_embs.append(chosen)
            
            # 【V4-2】GRU 更新 hidden state（可选）
            if self.use_gru:
                h = self.decoder_gru(chosen, h)
                h = F.normalize(h, p=2, dim=-1)
        
        # 堆叠结果
        all_logits = torch.stack(all_logits, dim=1)
        predictions = torch.stack(predictions, dim=1)
        
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
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理模式"""
        self.eval()
        with torch.no_grad():
            result = self.forward(query_emb, cand_emb, labels=None, return_loss=False)
            predictions = result['predictions']
            logits = result['logits']
            scores = logits.max(dim=-1)[0]
            return predictions, scores
    
    def get_div_lambda_values(self) -> List[float]:
        """获取当前的 div_lambda 值（用于监控，返回 softplus 后的实际值）"""
        with torch.no_grad():
            return [F.softplus(self.div_lambda[i]).item() for i in range(self.shot_num)]


class PointerSelectorV4_3Config:
    """V4-3 模型配置类"""
    
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
        use_step_emb: bool = True,
        use_gru: bool = True,
        mmr_reduction: str = 'max'
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
        self.use_step_emb = use_step_emb
        self.use_gru = use_gru
        self.mmr_reduction = mmr_reduction
    
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
            'use_step_emb': self.use_step_emb,
            'use_gru': self.use_gru,
            'mmr_reduction': self.mmr_reduction
        }


def build_model_v4_3(config: Optional[PointerSelectorV4_3Config] = None) -> PointerSelectorV4_3:
    """构建 V4-3 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_3Config()
    
    model = PointerSelectorV4_3(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        attn_dropout=config.attn_dropout,
        num_layers=config.num_layers,
        use_step_emb=config.use_step_emb,
        use_gru=config.use_gru,
        mmr_reduction=config.mmr_reduction
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV4_3 模型")
    print("="*70)
    
    model = build_model_v4_3()
    
    batch_size = 4
    d_model = 768
    K = 32
    shot_num = 2
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    labels = torch.randint(0, K, (batch_size, shot_num))
    
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
    
    print(f"\ndiv_lambda 值: {model.get_div_lambda_values()}")
    
    print("\n" + "="*70)
    print("✓ V4-3 模型测试通过！")
    print("="*70)
