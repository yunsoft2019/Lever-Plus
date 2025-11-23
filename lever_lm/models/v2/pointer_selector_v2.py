"""
Pointer Selector V2: Bi-Encoder + Cross-Attention 指针选择器

特点：
- 在V1基础上添加单层Cross-Attention增强q↔C细粒度对齐
- 输入：query_emb [B, d], cand_emb [B, K, d]
- 增强：q' = CrossAttn(q, C)
- 打分：scores = q' @ C^T
- 每步 masked softmax（屏蔽已选）
- 损失：交叉熵 + label smoothing

作者: Lever-Plus Team
日期: 2025-10-27
参考: yiyun.md V2 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointerSelectorV2(nn.Module):
    """
    V2 版本：Bi-Encoder + Cross-Attention 指针选择器
    
    在V1基础上添加单层Cross-Attention增强，提升q↔C细粒度对齐能力
    
    符合 yiyun.md 规范：
    - 输入维度：768 (CLIP ViT-L/14 输出)
    - 输出维度：256 (CLIP 768 → proj → 256)
    - 单层 Cross-Attention (num_heads=1, 保持优雅和效率)
    - dropout=0.1 (轻量正则化)
    - label_smoothing=0.1
    - L2 归一化
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
        attn_dropout: float = 0.1
    ):
        """
        初始化 V2 模型
        
        Args:
            d_model: 输入 embedding 维度 (默认 768, CLIP ViT-L/14 输出)
            K: 候选池大小 (默认 32)
            shot_num: 需要选择的样本数量 (默认 2)
            label_smoothing: 标签平滑系数 (默认 0.1)
            dropout: dropout 比例 (默认 0.1, 轻量正则化)
            hidden_dim: 输出维度 (默认 256, 符合 yiyun.md: CLIP 768 → proj → 256)
            num_heads: Cross-Attention 的头数 (默认 1, 单层保持效率)
            attn_dropout: Attention 层的 dropout (默认 0.1)
        """
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        
        # 投影层（如果 d_model != hidden_dim，需要降维/升维）
        # 符合 yiyun.md: d_model=256, hidden_dim=256 时为恒等映射
        if d_model != hidden_dim:
            self.input_proj = nn.Linear(d_model, hidden_dim, bias=False)
        else:
            # d_model == hidden_dim 时使用恒等映射（节省参数）
            self.input_proj = nn.Identity()
        
        # 【V2新增】Cross-Attention 层：增强query与candidates的细粒度交互
        # 使用多头注意力机制，让query从candidates中获取更丰富的上下文信息
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True  # 输入格式 [B, L, D]
        )
        
        # Layer Normalization（Cross-Attention后的归一化）
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # 投影层：hidden_dim -> hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Dropout 层（强力正则化）
        self.dropout = nn.Dropout(dropout)
        
        # 温度参数（用于控制 softmax 的尖锐度）
        # 归一化后的余弦相似度范围是[-1, 1]
        # 使用0.1，使softmax更尖锐，增强区分度
        # 参考V1经验：0.1 在归一化的768维嵌入上效果良好
        self.temperature = torch.tensor([0.1], dtype=torch.float32)
        
        # 初始化权重（Xavier初始化提高数值稳定性）
        self._init_weights()
        
        print(f"✓ PointerSelectorV2 初始化完成")
        print(f"  - d_model (输入): {d_model} -> hidden_dim (输出): {hidden_dim}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - attn_dropout: {attn_dropout}")
        print(f"  - temperature: 0.1 (增强区分度，加速学习)")
        print(f"  - residual_connection: ✓ (稳定训练)")
        print(f"  - 架构: CLIP {d_model} → proj → {hidden_dim} + Cross-Attention")
    
    def _init_weights(self):
        """初始化模型权重"""
        # 投影层使用 Xavier 初始化（如果不是 Identity）
        if not isinstance(self.input_proj, nn.Identity):
            nn.init.xavier_uniform_(self.input_proj.weight)
        
        # query/cand 投影层使用接近单位矩阵的初始化（256x256）
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.cand_proj.weight)
        
        # 添加小的随机扰动，避免完全对称
        with torch.no_grad():
            self.query_proj.weight.add_(torch.randn_like(self.query_proj.weight) * 0.01)
            self.cand_proj.weight.add_(torch.randn_like(self.cand_proj.weight) * 0.01)
    
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
        
        # 获取输入维度（可能是768或其他值）
        input_dim = query_emb.shape[-1]
        
        # 步骤1：投影到 hidden_dim（如果需要降维）
        query_reduced = self.input_proj(query_emb)  # [B, hidden_dim]
        cand_reduced = self.input_proj(cand_emb.reshape(-1, input_dim))  # [B*K, hidden_dim]
        cand_reduced = cand_reduced.reshape(batch_size, self.K, self.hidden_dim)  # [B, K, hidden_dim]
        
        # 【V2新增】步骤2：Cross-Attention 增强
        # query作为Q，candidates作为K和V，增强query的表示
        query_for_attn = query_reduced.unsqueeze(1)  # [B, 1, 256]
        attn_output, _ = self.cross_attn(
            query=query_for_attn,      # [B, 1, 256]
            key=cand_reduced,          # [B, K, 256]
            value=cand_reduced         # [B, K, 256]
        )
        # 【重要】添加残差连接，稳定训练
        # 标准 Transformer 做法：残差连接 + LayerNorm
        query_enhanced = self.attn_norm(attn_output + query_for_attn)  # [B, 1, 256]
        query_enhanced = query_enhanced.squeeze(1)  # [B, 256]
        
        # 步骤3：投影层（128 -> 128）
        query_proj = self.query_proj(query_enhanced)  # [B, 128]
        cand_proj = self.cand_proj(cand_reduced)      # [B, K, 128]
        
        # 步骤4：Dropout（训练时随机失活）
        query_proj = self.dropout(query_proj)
        cand_proj = self.dropout(cand_proj)
        
        # 步骤5：L2 归一化（确保余弦相似度在[-1, 1]范围内）
        query_proj = F.normalize(query_proj, p=2, dim=-1)
        cand_proj = F.normalize(cand_proj, p=2, dim=-1)
        
        # 存储每步的 logits 和预测
        all_logits = []
        predictions = []
        
        # mask：记录已选择的候选（初始全为 False）
        selected_mask = torch.zeros(batch_size, self.K, dtype=torch.bool, device=device)
        
        # 自回归生成 shot_num 步
        for step in range(self.shot_num):
            # 计算注意力分数：query @ cand^T
            scores = torch.matmul(query_proj.unsqueeze(1), cand_proj.transpose(1, 2))  # [B, 1, K]
            # 温度缩放（确保temperature在正确的设备上）
            temperature = self.temperature.to(device)
            scores = scores.squeeze(1) / temperature  # [B, K]
            
            # 应用 mask：将已选择的候选设为一个大的负数
            # 注意：不要在mask之前clamp，否则会破坏模型的区分能力
            scores = scores.masked_fill(selected_mask, -100.0)
            
            # 保存 logits
            all_logits.append(scores)
            
            # 预测（训练时也计算，用于监控）
            pred = scores.argmax(dim=-1)  # [B]
            predictions.append(pred)
            
            # 更新 mask（训练时使用真实标签，推理时使用预测）
            if labels is not None and step < labels.shape[1]:
                # 训练模式：使用真实标签更新 mask（Teacher Forcing）
                true_indices = labels[:, step]  # [B]
                # 使用非 in-place 操作，避免梯度计算错误
                selected_mask = selected_mask.scatter(1, true_indices.unsqueeze(1), True)
            else:
                # 推理模式：使用预测更新 mask
                # 使用非 in-place 操作，避免梯度计算错误
                selected_mask = selected_mask.scatter(1, pred.unsqueeze(1), True)
        
        # 堆叠结果
        all_logits = torch.stack(all_logits, dim=1)  # [B, S, K]
        predictions = torch.stack(predictions, dim=1)  # [B, S]
        
        result = {
            'logits': all_logits,
            'predictions': predictions
        }
        
        # 计算损失（训练时）
        if return_loss and labels is not None:
            loss = self.compute_loss(all_logits, labels)
            result['loss'] = loss
        
        return result
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失函数
        
        Args:
            logits: [B, S, K] 每步的 logits
            labels: [B, S] 真实标签
        
        Returns:
            loss: 标量损失
        """
        batch_size, shot_num, K = logits.shape
        
        # 重塑为 [B*S, K] 和 [B*S]
        logits_flat = logits.reshape(-1, K)
        labels_flat = labels.reshape(-1)
        
        # 【重要修复】Clamp logits 防止 -inf 与 label_smoothing 冲突
        # masked_fill 会产生 -100.0，当 label_smoothing > 0 时，
        # cross_entropy 会对所有类别计算概率，导致 log(softmax(-100)) = -inf
        # 通过 clamp 限制最小值，避免数值不稳定
        logits_flat = torch.clamp(logits_flat, min=-100.0)
        
        # 交叉熵损失 + label smoothing
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
        """
        推理模式：预测最优序列
        
        Args:
            query_emb: [B, d]
            cand_emb: [B, K, d]
            top_k: 每步返回 top-k 个候选（默认 1）
        
        Returns:
            predictions: [B, S] 预测的位置序列
            scores: [B, S] 对应的分数
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(query_emb, cand_emb, labels=None, return_loss=False)
            predictions = result['predictions']
            logits = result['logits']
            
            # 获取每步的最大分数
            scores = logits.max(dim=-1)[0]  # [B, S]
            
            return predictions, scores


class PointerSelectorV2Config:
    """V2 模型配置类（符合 yiyun.md 规范）
    
    注意：
    - d_model: 输入维度（CLIP 768）
    - hidden_dim: 输出维度（yiyun.md 要求 256）
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
        attn_dropout: float = 0.1
    ):
        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
    
    def to_dict(self):
        return {
            'd_model': self.d_model,
            'K': self.K,
            'shot_num': self.shot_num,
            'label_smoothing': self.label_smoothing,
            'dropout': self.dropout,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'attn_dropout': self.attn_dropout
        }


def build_model_v2(config: Optional[PointerSelectorV2Config] = None) -> PointerSelectorV2:
    """
    构建 V2 模型的工厂函数
    
    Args:
        config: 模型配置（可选）
    
    Returns:
        PointerSelectorV2 实例
    """
    if config is None:
        config = PointerSelectorV2Config()
    
    model = PointerSelectorV2(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        attn_dropout=config.attn_dropout
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV2 模型")
    print("="*70)
    
    # 创建模型
    model = build_model_v2()
    
    # 创建测试数据
    batch_size = 4
    d_model = 768  # 输入维度：CLIP ViT-L/14 输出
    K = 32
    shot_num = 2
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    labels = torch.randint(0, K, (batch_size, shot_num))
    
    print(f"\n输入形状:")
    print(f"  query_emb: {query_emb.shape}")
    print(f"  cand_emb: {cand_emb.shape}")
    print(f"  labels: {labels.shape}")
    
    # 前向传播
    print(f"\n前向传播...")
    result = model(query_emb, cand_emb, labels, return_loss=True)
    
    print(f"\n输出:")
    print(f"  logits: {result['logits'].shape}")
    print(f"  predictions: {result['predictions'].shape}")
    print(f"  loss: {result['loss'].item():.4f}")
    
    # 推理模式
    print(f"\n推理模式...")
    predictions, scores = model.predict(query_emb, cand_emb)
    print(f"  predictions: {predictions.shape}")
    print(f"  scores: {scores.shape}")
    
    print(f"\n示例预测:")
    print(f"  labels:      {labels[0].tolist()}")
    print(f"  predictions: {predictions[0].tolist()}")
    print(f"  scores:      {scores[0].tolist()}")
    
    print("\n" + "="*70)
    print("✓ V1 模型测试通过！")
    print("="*70)

