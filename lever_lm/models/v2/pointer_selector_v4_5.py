"""
Pointer Selector V4-5: Cross-Attn + GRU Pointer Decoder + Additive/Bilinear Attention 打分头

特点：
- 把 dot-product 换成 Additive (Bahdanau) 或 Bilinear Attention 打分头
- 解决 query embedding 和 candidate embedding 不完全同空间的问题
- 提升打分的可表达性

核心改动：
- 新增 attention_type: 'dot', 'additive', 'bilinear' 三种打分方式
- Additive: score = v^T * tanh(W_q * q + W_c * c)
- Bilinear: score = q^T * W * c

作者: Lever-Plus Team
日期: 2025-12-26
参考: Lever-Plus_PointerSelector_Upgrade_Plans_Keep_RCE_GRPO.md V4-5 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PointerSelectorV4_5(nn.Module):
    """
    V4-5 版本：Cross-Attn + GRU Pointer Decoder + Additive/Bilinear Attention
    
    把 dot-product 换成更强的打分头，解决 embedding 不完全同空间的问题
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
        use_gru: bool = True,
        attention_type: str = 'additive'  # 'dot', 'additive', 'bilinear'
    ):
        """
        初始化 V4-5 模型
        
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
            attention_type: 打分头类型 (默认 'additive')
                - 'dot': 原始点积打分
                - 'additive': Bahdanau Attention (推荐)
                - 'bilinear': Bilinear Attention
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
        self.attention_type = attention_type
        
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
        
        # 【V4-5 核心】Attention 打分头
        if attention_type == 'additive':
            # Bahdanau (Additive) Attention
            # score = v^T * tanh(W_q * q + W_c * c)
            self.attn_Wq = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.attn_Wc = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        elif attention_type == 'bilinear':
            # Bilinear Attention
            # score = q^T * W * c
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)
        # 'dot' 不需要额外参数
        
        # 温度参数
        self.temperature = torch.tensor([0.1], dtype=torch.float32)
        
        # 初始化权重
        self._init_weights()
        
        print(f"✓ PointerSelectorV4_5 初始化完成")
        print(f"  - d_model (输入): {d_model} -> hidden_dim (输出): {hidden_dim}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - use_step_emb: {use_step_emb}")
        print(f"  - use_gru: {use_gru}")
        print(f"  - attention_type: {attention_type}")
        print(f"  - 架构: CLIP {d_model} → proj → {hidden_dim} + {num_layers}层 Cross-Attention + GRU + {attention_type.upper()} Attention")
    
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
        
        # Attention 打分头初始化
        if self.attention_type == 'additive':
            nn.init.xavier_uniform_(self.attn_Wq.weight)
            nn.init.zeros_(self.attn_Wq.bias)
            nn.init.xavier_uniform_(self.attn_Wc.weight)
            # attn_v 使用较小的初始化，使输出范围接近 [-1, 1]
            # xavier_uniform 的范围是 [-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
            # 对于 (1, 256)，范围约为 [-0.15, 0.15]
            # 我们希望输出范围接近 [-1, 1]，所以需要缩放
            nn.init.xavier_uniform_(self.attn_v.weight)
            # 缩放 attn_v 使其输出范围更小（接近 dot-product 的范围）
            with torch.no_grad():
                self.attn_v.weight.mul_(0.1)  # 缩小 10 倍
        elif self.attention_type == 'bilinear':
            nn.init.xavier_uniform_(self.bilinear.weight)
    
    def compute_attention_scores(
        self,
        h: torch.Tensor,
        cand_proj_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力分数
        
        Args:
            h: [B, H] query hidden state
            cand_proj_norm: [B, K, H] 候选 embedding（已归一化）
        
        Returns:
            scores: [B, K] 注意力分数
        """
        B, K, H = cand_proj_norm.shape
        device = h.device
        temperature = self.temperature.to(device)
        
        if self.attention_type == 'dot':
            # 原始点积打分
            # h 和 cand_proj_norm 都是 L2 归一化的，所以点积范围是 [-1, 1]
            scores = torch.matmul(h.unsqueeze(1), cand_proj_norm.transpose(1, 2))
            scores = scores.squeeze(1) / temperature  # [B, K]
            
        elif self.attention_type == 'additive':
            # Bahdanau (Additive) Attention
            # score = v^T * tanh(W_q * q + W_c * c)
            # tanh 输出范围是 [-1, 1]，v^T 的输出是标量
            q_transformed = self.attn_Wq(h).unsqueeze(1)  # [B, 1, H]
            c_transformed = self.attn_Wc(cand_proj_norm)  # [B, K, H]
            combined = torch.tanh(q_transformed + c_transformed)  # [B, K, H]
            scores = self.attn_v(combined).squeeze(-1)  # [B, K]
            # 注意：Additive Attention 的输出范围取决于 v 的初始化
            # 为了与 dot-product 保持一致，我们也除以 temperature
            # 但由于 tanh 的输出范围是 [-1, 1]，v 的输出范围大约是 [-sqrt(H), sqrt(H)]
            # 所以我们需要先归一化到 [-1, 1] 范围，再除以 temperature
            # 使用 tanh 来限制范围，然后除以 temperature
            scores = scores / temperature
            
        elif self.attention_type == 'bilinear':
            # Bilinear Attention
            # score = q^T * W * c
            h_expanded = h.unsqueeze(1).expand(-1, K, -1)  # [B, K, H]
            scores = self.bilinear(h_expanded, cand_proj_norm).squeeze(-1)  # [B, K]
            scores = scores / temperature
        
        return scores
    
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
        
        # 步骤5：L2 归一化（对于 dot 和 bilinear 有用，additive 可选）
        h = F.normalize(query_proj, p=2, dim=-1)
        cand_proj_norm = F.normalize(cand_proj_out, p=2, dim=-1)
        
        # 存储每步的 logits 和预测
        all_logits = []
        predictions = []
        
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
            
            # 【V4-5 核心】使用新的打分头计算注意力分数
            scores = self.compute_attention_scores(h_step, cand_proj_norm)  # [B, K]
            
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


class PointerSelectorV4_5Config:
    """V4-5 模型配置类"""
    
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
        attention_type: str = 'additive'
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
        self.attention_type = attention_type
    
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
            'attention_type': self.attention_type
        }


def build_model_v4_5(config: Optional[PointerSelectorV4_5Config] = None) -> PointerSelectorV4_5:
    """构建 V4-5 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_5Config()
    
    model = PointerSelectorV4_5(
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
        attention_type=config.attention_type
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV4_5 模型")
    print("="*70)
    
    batch_size = 4
    d_model = 768
    K = 32
    shot_num = 2
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    labels = torch.randint(0, K, (batch_size, shot_num))
    
    for attn_type in ['dot', 'additive', 'bilinear']:
        print(f"\n{'='*70}")
        print(f"测试 attention_type = '{attn_type}'")
        print("="*70)
        
        config = PointerSelectorV4_5Config(attention_type=attn_type)
        model = build_model_v4_5(config)
        
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
    print("✓ V4-5 模型测试通过！")
    print("="*70)
