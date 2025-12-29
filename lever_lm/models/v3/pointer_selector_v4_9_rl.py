"""
Pointer Selector V4-9 RL: Two-Stage Coarse-to-Fine + 强化学习

继承 V4-9 的 Two-Stage 架构，添加 RCE + GRPO 支持

作者: Lever-Plus Team
日期: 2025-12-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lever_lm.models.v2.pointer_selector_v4_9 import PointerSelectorV4_9, PointerSelectorV4_9Config


class PointerSelectorV4_9_RL(PointerSelectorV4_9):
    """
    V4-9 RL 版本：Two-Stage Coarse-to-Fine + 强化学习
    
    继承 V4-9 的 Two-Stage 架构，添加 RCE + GRPO 支持
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
        top_m: int = 8,
        refine_type: str = "attn",
        use_gru: bool = True,
        use_step_emb: bool = True,
        # RL 相关参数
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
        advantage_clip: float = 5.0,
    ):
        """
        初始化 V4-9 RL 模型
        """
        super().__init__(
            d_model=d_model,
            K=K,
            shot_num=shot_num,
            label_smoothing=label_smoothing,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            num_layers=num_layers,
            top_m=top_m,
            refine_type=refine_type,
            use_gru=use_gru,
            use_step_emb=use_step_emb,
        )
        
        # RL 相关参数
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.advantage_clip = advantage_clip
        self.use_ppo_clip_only = False
        
        print(f"✓ PointerSelectorV4_9_RL 初始化完成（V4-9 + 强化学习）")
        print(f"  - clip_epsilon (PPO裁剪): {clip_epsilon}")
        print(f"  - kl_beta (KL权重): {kl_beta}")
        print(f"  - advantage_clip: [-{advantage_clip}, {advantage_clip}]")
    
    def compute_log_probs(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算给定标签序列的log概率"""
        result = self.forward(query_emb, cand_emb, labels=labels, return_loss=False)
        logits = result['logits']
        
        batch_size, shot_num, K = logits.shape
        log_probs_all = F.log_softmax(logits, dim=-1)
        labels_expanded = labels.unsqueeze(-1)
        selected_log_probs = log_probs_all.gather(dim=-1, index=labels_expanded)
        selected_log_probs = selected_log_probs.squeeze(-1)
        seq_log_probs = selected_log_probs.sum(dim=-1)
        
        return seq_log_probs
    
    def compute_log_probs_per_beam(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor
    ) -> torch.Tensor:
        """计算每个beam的log概率"""
        batch_size, num_beams, shot_num = beam_labels.shape
        K_actual = cand_emb.shape[1]
        
        query_expanded = query_emb.unsqueeze(1).expand(-1, num_beams, -1)
        query_expanded = query_expanded.reshape(batch_size * num_beams, -1)
        
        cand_expanded = cand_emb.unsqueeze(1).expand(-1, num_beams, -1, -1)
        cand_expanded = cand_expanded.reshape(batch_size * num_beams, K_actual, -1)
        
        labels_flat = beam_labels.reshape(batch_size * num_beams, shot_num)
        log_probs_flat = self.compute_log_probs(query_expanded, cand_expanded, labels_flat)
        log_probs = log_probs_flat.reshape(batch_size, num_beams)
        
        return log_probs
    
    def compute_advantage(
        self,
        rewards: torch.Tensor,
        normalize: bool = True,
        use_rank: bool = False
    ) -> torch.Tensor:
        """计算组内相对优势"""
        batch_size, num_beams = rewards.shape
        
        if use_rank:
            ranks = rewards.argsort(dim=-1, descending=True).argsort(dim=-1).float()
            if num_beams > 1:
                advantages = 1.0 - 2.0 * ranks / (num_beams - 1)
            else:
                advantages = torch.zeros_like(ranks)
        else:
            mean = rewards.mean(dim=-1, keepdim=True)
            std = rewards.std(dim=-1, keepdim=True)
            std = torch.clamp(std, min=0.1)
            advantages = (rewards - mean) / std
        
        advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        return advantages
    
    def compute_kl_divergence(
        self,
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor
    ) -> torch.Tensor:
        """计算KL散度（近似）"""
        log_ratio = log_probs_new - log_probs_old
        ratio = torch.exp(log_ratio)
        kl = ratio - 1 - log_ratio
        return kl.mean()
    
    def compute_rce_loss(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor,
        beam_rewards: torch.Tensor,
        temperature: float = 1.0,
        use_rank_normalization: bool = False,
        use_top1_only: bool = False
    ) -> torch.Tensor:
        """计算RCE预热损失"""
        batch_size, num_beams, shot_num = beam_labels.shape
        
        if use_top1_only:
            labels = beam_labels[:, 0, :]
            actual_K = cand_emb.shape[1]
            result = self.forward(query_emb, cand_emb, labels=labels, return_loss=False)
            logits = result['logits']
            logits_for_loss = logits.reshape(-1, actual_K)
            labels_for_loss = labels.reshape(-1)
            logits_for_loss = torch.clamp(logits_for_loss, min=-100.0)
            loss = F.cross_entropy(
                logits_for_loss, labels_for_loss,
                label_smoothing=self.label_smoothing
            )
            return loss
        
        if use_rank_normalization:
            ranks = beam_rewards.argsort(dim=-1).argsort(dim=-1).float()
            normalized_scores = ranks / (num_beams - 1) if num_beams > 1 else ranks
            weights = F.softmax(normalized_scores / temperature, dim=-1)
        else:
            weights = F.softmax(beam_rewards / temperature, dim=-1)
        
        labels_flat = beam_labels.reshape(batch_size * num_beams, shot_num)
        actual_K = cand_emb.shape[1]
        
        query_expanded = query_emb.unsqueeze(1).expand(-1, num_beams, -1)
        query_expanded = query_expanded.reshape(batch_size * num_beams, -1)
        
        cand_expanded = cand_emb.unsqueeze(1).expand(-1, num_beams, -1, -1)
        cand_expanded = cand_expanded.reshape(batch_size * num_beams, actual_K, -1)
        
        result = self.forward(query_expanded, cand_expanded, labels=labels_flat, return_loss=False)
        logits = result['logits']
        
        logits_for_loss = logits.reshape(-1, actual_K)
        labels_for_loss = labels_flat.reshape(-1)
        logits_for_loss = torch.clamp(logits_for_loss, min=-100.0)
        
        ce_losses = F.cross_entropy(
            logits_for_loss, labels_for_loss,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        ce_losses = ce_losses.reshape(batch_size, num_beams, shot_num).sum(dim=-1)
        weighted_loss = (weights * ce_losses).sum(dim=-1).mean()
        
        return weighted_loss

    def compute_grpo_loss(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor,
        beam_rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        use_top_k: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """计算GRPO损失"""
        batch_size, num_beams, shot_num = beam_labels.shape
        
        if use_top_k is not None and use_top_k < num_beams:
            _, top_indices = beam_rewards.topk(use_top_k, dim=-1)
            beam_labels = torch.gather(
                beam_labels, 
                dim=1, 
                index=top_indices.unsqueeze(-1).expand(-1, -1, shot_num)
            )
            beam_rewards = torch.gather(beam_rewards, dim=1, index=top_indices)
            old_log_probs = torch.gather(old_log_probs, dim=1, index=top_indices)
            num_beams = use_top_k
        
        new_log_probs = self.compute_log_probs_per_beam(query_emb, cand_emb, beam_labels)
        advantages = self.compute_advantage(beam_rewards, normalize=True)
        
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        ppo_obj = torch.min(obj1, obj2)
        ppo_loss = -ppo_obj.mean()
        
        kl = self.compute_kl_divergence(new_log_probs, old_log_probs)
        
        if self.use_ppo_clip_only:
            kl_loss = torch.tensor(0.0, device=ppo_loss.device)
            total_loss = ppo_loss
        else:
            kl_loss = self.kl_beta * kl
            total_loss = ppo_loss + kl_loss
        
        return {
            'loss': total_loss,
            'ppo_loss': ppo_loss,
            'kl_loss': kl_loss,
            'kl': kl,
            'mean_ratio': ratio.mean(),
            'mean_advantage': advantages.mean(),
            'std_advantage': advantages.std(),
            'max_advantage': advantages.abs().max()
        }


class PointerSelectorV4_9_RLConfig(PointerSelectorV4_9Config):
    """V4-9 RL 模型配置类"""
    
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
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
        advantage_clip: float = 5.0,
    ):
        super().__init__(
            d_model=d_model,
            K=K,
            shot_num=shot_num,
            label_smoothing=label_smoothing,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            num_layers=num_layers,
            top_m=top_m,
            refine_type=refine_type,
            use_gru=use_gru,
            use_step_emb=use_step_emb,
        )
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.advantage_clip = advantage_clip
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            'clip_epsilon': self.clip_epsilon,
            'kl_beta': self.kl_beta,
            'advantage_clip': self.advantage_clip,
        })
        return d


def build_model_v4_9_rl(config: Optional[PointerSelectorV4_9_RLConfig] = None) -> PointerSelectorV4_9_RL:
    """构建 V4-9 RL 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_9_RLConfig()
    
    return PointerSelectorV4_9_RL(**config.to_dict())
