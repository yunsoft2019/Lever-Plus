"""
Pointer Selector V4-6: Cross-Attn + GRU Pointer Decoder + Coverage / Topic 原型覆盖

特点：
- 引入 M 个可学习的"原型向量"（topic prototypes）
- 每个 candidate 计算其 topic 分布
- query 预测需要哪些 topics
- 选择时倾向于覆盖未覆盖的 topics，强化互补覆盖、减少重复

核心改动：
- 新增 topic_prototypes: [M, H] 可学习原型向量
- 新增 query_topic_head: 预测 query 需要哪些 topics
- 新增 cover_lambda: 可学习的覆盖增益权重
- 每步计算 coverage gain，倾向选择能覆盖未覆盖 topic 的候选

作者: Lever-Plus Team
日期: 2025-12-27
参考: Lever-Plus_PointerSelector_Upgrade_Plans_Keep_RCE_GRPO.md V4-6 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PointerSelectorV4_6(nn.Module):
    """
    V4-6 版本：Cross-Attn + GRU Pointer Decoder + Coverage / Topic 原型覆盖
    
    让模型学会"互补覆盖"：后续 demo 更倾向于覆盖前面没覆盖到的"原型/簇"
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
        num_topics: int = 16,  # 原型/topic 数量
        cover_lambda_init: float = 0.0  # 覆盖增益权重初始值
    ):
        """
        初始化 V4-6 模型
        
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
            num_topics: 原型/topic 数量 (默认 16)
            cover_lambda_init: 覆盖增益权重初始值 (默认 0.0，先学相关性再学覆盖)
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
        self.num_topics = num_topics
        
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
        # 使用 max_shot_num=8 以支持推理时更大的 shot_num
        self.max_shot_num = 8
        if use_step_emb:
            self.step_emb = nn.Embedding(self.max_shot_num, hidden_dim)
        
        # 【V4-6 核心】Coverage / Topic 原型覆盖
        # M 个可学习的原型向量
        self.topic_prototypes = nn.Parameter(torch.randn(num_topics, hidden_dim))
        
        # query 需要哪些 topics 的预测头
        self.query_topic_head = nn.Linear(hidden_dim, num_topics, bias=True)
        
        # 可学习的覆盖增益权重
        # 使用 softplus 激活，所以初始化为负数使得 softplus(init) ≈ 0
        # softplus(-2) ≈ 0.127，softplus(-3) ≈ 0.049
        self.cover_lambda = nn.Parameter(torch.tensor(cover_lambda_init if cover_lambda_init != 0.0 else -2.0))
        
        # 温度参数
        self.temperature = torch.tensor([0.1], dtype=torch.float32)
        
        # 初始化权重
        self._init_weights()
        
        print(f"✓ PointerSelectorV4_6 初始化完成")
        print(f"  - d_model (输入): {d_model} -> hidden_dim (输出): {hidden_dim}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - use_step_emb: {use_step_emb}")
        print(f"  - use_gru: {use_gru}")
        print(f"  - num_topics: {num_topics}")
        print(f"  - cover_lambda_init: {cover_lambda_init}")
        print(f"  - 架构: CLIP {d_model} → proj → {hidden_dim} + {num_layers}层 Cross-Attention + GRU + Coverage({num_topics} topics)")
    
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
        
        # 【V4-6】Topic prototypes 使用正态分布初始化
        nn.init.normal_(self.topic_prototypes, mean=0.0, std=0.02)
        
        # query_topic_head 使用 Xavier 初始化
        nn.init.xavier_uniform_(self.query_topic_head.weight)
        nn.init.zeros_(self.query_topic_head.bias)
    
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
        
        # 【V4-6 核心】预计算 topic 相关信息
        # 归一化 topic prototypes
        proto = F.normalize(self.topic_prototypes, p=2, dim=-1)  # [M, H]
        
        # 每个 candidate 的 topic 分布
        # cand_proj_norm: [B, K, H], proto: [M, H]
        topic_logits = torch.matmul(cand_proj_norm, proto.t())  # [B, K, M]
        topic_probs = F.softmax(topic_logits, dim=-1)  # [B, K, M]
        
        # query 需要的 topics（使用初始 h 计算）
        h0 = h  # 保存初始状态用于计算 topic need
        need = F.softmax(self.query_topic_head(h0), dim=-1)  # [B, M]
        
        # 存储每步的 logits 和预测
        all_logits = []
        predictions = []
        
        # mask：记录已选择的候选
        selected_mask = torch.zeros(batch_size, actual_K, dtype=torch.bool, device=device)
        
        # 已覆盖的 topics
        covered = torch.zeros(batch_size, self.num_topics, device=device)  # [B, M]
        
        # 温度参数
        temperature = self.temperature.to(device)
        
        # 自回归生成 shot_num 步
        # 推理时可能需要更多的 shot，使用 labels 的长度或 self.shot_num
        actual_shot_num = labels.shape[1] if labels is not None else self.shot_num
        for step in range(actual_shot_num):
            # 【V4-2】可选：添加 step embedding
            if self.use_step_emb:
                # 使用 min 确保不会越界
                step_idx = min(step, self.max_shot_num - 1)
                step_tensor = torch.tensor([step_idx], device=device)
                h_step = h + self.step_emb(step_tensor)
                h_step = F.normalize(h_step, p=2, dim=-1)
            else:
                h_step = h
            
            # 基础打分（点积）
            base_scores = torch.matmul(h_step.unsqueeze(1), cand_proj_norm.transpose(1, 2))
            base_scores = base_scores.squeeze(1) / temperature  # [B, K]
            
            # 【V4-6 核心】计算 coverage gain
            if step > 0:
                # 未覆盖的 topics
                uncovered = (1.0 - covered).clamp(min=0.0, max=1.0)  # [B, M]
                
                # coverage gain：倾向选择能覆盖未覆盖 topic 的候选
                # need * uncovered: [B, M] - query 需要且未覆盖的 topics
                # topic_probs: [B, K, M] - 每个候选的 topic 分布
                # gain: [B, K] - 每个候选能带来的覆盖增益
                gain = torch.einsum("bm,bkm->bk", need * uncovered, topic_probs)  # [B, K]
                
                # 使用 softplus 保证 cover_lambda 非负，同时允许梯度流过
                # softplus(x) = log(1 + exp(x))，当 x=0 时 softplus(0) ≈ 0.693
                # 这样即使初始化为 0，也能有梯度更新
                scores = base_scores + F.softplus(self.cover_lambda) * gain
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
            
            # 【V4-6】更新 covered（已覆盖的 topics）
            # 取出被选 demo 的 topic 分布
            chosen_topic = topic_probs.gather(
                1, idx.view(batch_size, 1, 1).expand(-1, 1, self.num_topics)
            ).squeeze(1)  # [B, M]
            
            # 累加覆盖（clamp 到 [0, 1]）
            covered = (covered + chosen_topic).clamp(max=1.0)
            
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


class PointerSelectorV4_6Config:
    """V4-6 模型配置类"""
    
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
        num_topics: int = 16,
        cover_lambda_init: float = 0.0
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
        self.num_topics = num_topics
        self.cover_lambda_init = cover_lambda_init
    
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
            'num_topics': self.num_topics,
            'cover_lambda_init': self.cover_lambda_init
        }


def build_model_v4_6(config: Optional[PointerSelectorV4_6Config] = None) -> PointerSelectorV4_6:
    """构建 V4-6 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_6Config()
    
    model = PointerSelectorV4_6(
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
        num_topics=config.num_topics,
        cover_lambda_init=config.cover_lambda_init
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV4_6 模型")
    print("="*70)
    
    batch_size = 4
    d_model = 768
    K = 32
    shot_num = 4  # 测试多 shot 场景
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    labels = torch.randint(0, K, (batch_size, shot_num))
    
    # 测试不同的 num_topics
    for num_topics in [8, 16, 32]:
        print(f"\n{'='*70}")
        print(f"测试 num_topics = {num_topics}")
        print("="*70)
        
        config = PointerSelectorV4_6Config(
            shot_num=shot_num,
            num_topics=num_topics
        )
        model = build_model_v4_6(config)
        
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
        
        # 检查 cover_lambda 的值
        print(f"\n  cover_lambda: {model.cover_lambda.item():.4f}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n参数量: {total_params:,}")
    
    print("\n" + "="*70)
    print("✓ V4-6 模型测试通过！")
    print("="*70)
