"""
Pointer Selector V1: Bi-Encoder 指针选择器（基础版）

特点：
- 输入：query_emb [B, d], cand_emb [B, K, d]
- 打分：scores = query_emb @ cand_emb^T
- 每步 masked softmax（屏蔽已选）
- 损失：交叉熵 + label smoothing

作者: Lever-Plus Team
日期: 2025-10-26
参考: yiyun.md V1 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointerSelectorV1(nn.Module):
    """
    V1 版本：Bi-Encoder 指针选择器
    
    最简单的指针网络实现，使用点积注意力机制
    """
    
    def __init__(
        self,
        d_model: int = 256,
        K: int = 32,
        shot_num: int = 2,
        label_smoothing: float = 0.1,
        dropout: float = 0.1
    ):
        """
        初始化 V1 模型
        
        Args:
            d_model: embedding 维度 (默认 256)
            K: 候选池大小 (默认 32)
            shot_num: 需要选择的样本数量 (默认 2)
            label_smoothing: 标签平滑系数 (默认 0.1)
            dropout: dropout 比例 (默认 0.1)
        """
        super().__init__()
        
        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        
        # 可选：query 投影层（用于增强表达能力）
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),  # 添加 LayerNorm 防止数值爆炸
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)   # 归一化前再加一层 LayerNorm
        )
        
        # 可选：候选投影层
        self.cand_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),  # 添加 LayerNorm 防止数值爆炸
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)   # 归一化前再加一层 LayerNorm
        )
        
        # 温度参数（用于控制 softmax 的尖锐度）
        # 修复：从0.07增大到0.1，避免在高维空间(d_model=768)时数值溢出
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        print(f"✓ PointerSelectorV1 初始化完成")
        print(f"  - d_model: {d_model}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
    
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
        
        # 投影 query 和 candidates
        query_proj = self.query_proj(query_emb)  # [B, d]
        cand_proj = self.cand_proj(cand_emb)     # [B, K, d]
        
        # L2 归一化（提高稳定性）
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
            scores = scores.squeeze(1) / self.temperature  # [B, K]，温度缩放
            
            # 应用 mask：将已选择的候选设为 -inf
            scores = scores.masked_fill(selected_mask, float('-inf'))
            
            # 保存 logits
            all_logits.append(scores)
            
            # 预测（训练时也计算，用于监控）
            pred = scores.argmax(dim=-1)  # [B]
            predictions.append(pred)
            
            # 更新 mask（训练时使用真实标签，推理时使用预测）
            # 🔧 修复：使用非就地操作，避免梯度计算错误
            if labels is not None and step < labels.shape[1]:
                # 训练模式：使用真实标签更新 mask（Teacher Forcing）
                true_indices = labels[:, step]  # [B]
                # 使用 scatter 而非 scatter_，创建新tensor而非就地修改
                selected_mask = selected_mask.scatter(1, true_indices.unsqueeze(1), True)
            else:
                # 推理模式：使用预测更新 mask
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
        
        # 🔧 关键修复：将 -inf 替换为一个非常小的值，避免与 label_smoothing 冲突
        # label_smoothing 会将部分概率分配给所有类别，包括被 mask 的（-inf）
        # 这会导致 log(0) = -inf，进而导致 loss = inf
        # 使用 -100：softmax(-100) ≈ 3.7e-44，接近0但不会导致数值问题
        logits_clamped = torch.clamp(logits, min=-100.0)  # 替换 -inf 为 -100
        
        # 重塑为 [B*S, K] 和 [B*S]
        logits_flat = logits_clamped.reshape(-1, K)
        labels_flat = labels.reshape(-1)
        
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


class PointerSelectorV1Config:
    """V1 模型配置类"""
    
    def __init__(
        self,
        d_model: int = 256,
        K: int = 32,
        shot_num: int = 2,
        label_smoothing: float = 0.1,
        dropout: float = 0.1
    ):
        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.dropout = dropout
    
    def to_dict(self):
        return {
            'd_model': self.d_model,
            'K': self.K,
            'shot_num': self.shot_num,
            'label_smoothing': self.label_smoothing,
            'dropout': self.dropout
        }


def build_model_v1(config: Optional[PointerSelectorV1Config] = None) -> PointerSelectorV1:
    """
    构建 V1 模型的工厂函数
    
    Args:
        config: 模型配置（可选）
    
    Returns:
        PointerSelectorV1 实例
    """
    if config is None:
        config = PointerSelectorV1Config()
    
    model = PointerSelectorV1(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV1 模型")
    print("="*70)
    
    # 创建模型
    model = build_model_v1()
    
    # 创建测试数据
    batch_size = 4
    d_model = 256
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

