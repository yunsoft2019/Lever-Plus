"""
Pointer Selector V1: Bi-Encoder 架构（恢复原版）

V1 特点：
- 使用独立的编码器分别编码 query 和 candidates（Bi-Encoder）
- 简单的 MLP 投影层
- Teacher Forcing 训练
- 指针网络选择机制

作者: Lever-Plus Team
日期: 2025-10-28
版本: V1 Bi-Encoder (恢复)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointerSelectorV1(nn.Module):
    """
    V1 版本：Bi-Encoder 指针网络
    使用独立的编码器分别编码 query 和 candidates
    """
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 6,
        label_smoothing: float = 0.0,
        dropout: float = 0.5
    ):
        super().__init__()

        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing

        # Query 投影层
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Candidate 投影层
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

        self._init_weights()

        print(f"✓ PointerSelectorV1 (Bi-Encoder) 初始化完成")
        print(f"  - d_model: {d_model}")
        print(f"  - K (候选池大小): {K}")
        print(f"  - shot_num: {shot_num}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - dropout: {dropout}")
        print(f"  - temperature: {self.temperature}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - 参数量: {total_params/1e6:.2f}M")
        print(f"  - 架构: Bi-Encoder (简单双塔)")


    def _init_weights(self):
        """初始化模型权重"""
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
            query_emb: [B, d_model]
            cand_emb: [B, K, d_model]
            labels: [B, shot_num] 可选，真实标签
            return_loss: 是否计算损失
        
        Returns:
            dict: {
                'logits': [B, shot_num, K],
                'predictions': [B, shot_num],
                'loss': scalar (如果 return_loss=True 且 labels 不为 None)
            }
        """
        batch_size = query_emb.shape[0]
        device = query_emb.device
        
        # 投影 query 和 candidates
        query_proj = self.query_proj(query_emb)  # [B, d_model]
        cand_proj = self.cand_proj(cand_emb)     # [B, K, d_model]
        
        # L2 归一化
        query_proj = F.normalize(query_proj, dim=-1)
        cand_proj = F.normalize(cand_proj, dim=-1)
        
        all_logits = []
        predictions = []
        
        # 当前 query 状态（初始为原始 query）
        current_query = query_proj
        
        # mask：记录已选择的候选（初始全为 False）
        selected_mask = torch.zeros(batch_size, self.K, dtype=torch.bool, device=device)
        
        for step in range(self.shot_num):
            # 计算相似度分数
            scores = torch.matmul(current_query.unsqueeze(1), cand_proj.transpose(1, 2)).squeeze(1)  # [B, K]
            scores = scores / self.temperature
            
            # 应用 mask
            scores = scores.masked_fill(selected_mask, -100.0)
            all_logits.append(scores)
            
            # 预测
            pred = scores.argmax(dim=-1)  # [B]
            predictions.append(pred)
            
            # 更新 mask (Teacher Forcing 或 Inference)
            if labels is not None and step < labels.shape[1]:
                true_indices = labels[:, step]
                selected_mask = selected_mask.scatter(1, true_indices.unsqueeze(1), True)
                # 更新 query（加入真实标签对应的候选）
                next_icd = torch.gather(cand_proj, 1, true_indices.view(batch_size, 1, 1).expand(-1, -1, self.d_model)).squeeze(1)
            else:
                selected_mask = selected_mask.scatter(1, pred.unsqueeze(1), True)
                # 更新 query（加入预测的候选）
                next_icd = torch.gather(cand_proj, 1, pred.view(batch_size, 1, 1).expand(-1, -1, self.d_model)).squeeze(1)
            
            # 更新 current_query（简单平均）
            current_query = (current_query + next_icd) / 2.0
            current_query = F.normalize(current_query, dim=-1)
        
        all_logits = torch.stack(all_logits, dim=1)  # [B, shot_num, K]
        predictions = torch.stack(predictions, dim=1)  # [B, shot_num]
        
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
        """
        计算交叉熵损失
        
        Args:
            logits: [B, shot_num, K]
            labels: [B, shot_num]
        
        Returns:
            loss: scalar
        """
        batch_size, shot_num, K = logits.shape
        logits_flat = logits.reshape(-1, K)
        labels_flat = labels.reshape(-1)
        
        # Clamp logits to prevent -inf with label_smoothing
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
        """
        推理：预测 top-k 个候选
        
        Args:
            query_emb: [B, d_model]
            cand_emb: [B, K, d_model]
            top_k: 返回 top-k 个候选
        
        Returns:
            predictions: [B, shot_num]
            scores: [B, shot_num]
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(query_emb, cand_emb, labels=None, return_loss=False)
            predictions = result['predictions']
            logits = result['logits']
            scores = logits.max(dim=-1)[0]  # [B, shot_num]
            return predictions, scores


class PointerSelectorV1Config:
    """V1 (Bi-Encoder) 模型配置类"""
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 6,
        label_smoothing: float = 0.0,
        dropout: float = 0.5
    ):
        self.d_model = d_model
        self.K = K
        self.shot_num = shot_num
        self.label_smoothing = label_smoothing
        self.dropout = dropout
    
    def to_dict(self):
        """转换为字典（用于保存检查点）"""
        return {
            'd_model': self.d_model,
            'K': self.K,
            'shot_num': self.shot_num,
            'label_smoothing': self.label_smoothing,
            'dropout': self.dropout
        }


def build_model_v1(config: PointerSelectorV1Config) -> PointerSelectorV1:
    """构建 V1 (Bi-Encoder) 模型"""
    model = PointerSelectorV1(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout
    )
    return model
