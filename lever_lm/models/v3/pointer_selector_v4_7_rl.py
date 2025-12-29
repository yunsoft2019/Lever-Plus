"""
Pointer Selector V4-7 RL: V4-7 + 离线强化学习（RCE预热 + GRPO后训练）

特点：
- 继承V4-7的GRU Pointer Decoder + (N)DPP / log-det 风格集合增益架构
- 复用V3的RCE/GRPO方法
- 支持组内相对优势计算
- 支持KL散度计算和自适应调整

作者: Lever-Plus Team
日期: 2025-12-27
参考: Lever-Plus_PointerSelector_Upgrade_Plans_Keep_RCE_GRPO.md V4-7 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from ..v2.pointer_selector_v4_7 import PointerSelectorV4_7


class PointerSelectorV4_7_RL(PointerSelectorV4_7):
    """
    V4-7 RL 版本：V4-7 + 离线强化学习（RCE预热 + GRPO后训练）
    
    继承V4-7的GRU Pointer Decoder + (N)DPP / log-det 风格集合增益架构，新增强化学习相关方法：
    - compute_rce_loss(): RCE预热损失
    - compute_grpo_loss(): GRPO策略梯度损失
    - compute_advantage(): 组内相对优势计算
    - compute_kl_divergence(): KL散度计算
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
        dpp_rank: int = 32,
        dpp_lambda_init: float = 0.0,
        # RL参数
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
        advantage_clip: float = 5.0
    ):
        """
        初始化 V4-7 RL 模型
        
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
            dpp_rank: DPP 低秩投影维度 (默认 32)
            dpp_lambda_init: DPP 增益权重初始值 (默认 0.0)
            clip_epsilon: PPO裁剪参数ε (默认 0.2)
            kl_beta: KL散度权重β (默认 0.1)
            advantage_clip: 优势裁剪范围 (默认 5.0)
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
            use_step_emb=use_step_emb,
            use_gru=use_gru,
            dpp_rank=dpp_rank,
            dpp_lambda_init=dpp_lambda_init
        )
        
        # RL参数
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.advantage_clip = advantage_clip
        self.use_ppo_clip_only = False
        
        print(f"✓ PointerSelectorV4_7_RL 初始化完成（V4-7 + 强化学习）")
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
    
    def update_kl_beta(
        self,
        current_kl: float,
        kl_target_min: float = 0.01,
        kl_target_max: float = 0.1,
        adjustment_factor: float = 1.5
    ) -> float:
        """KL散度自适应调整"""
        if current_kl > kl_target_max:
            self.kl_beta = self.kl_beta * adjustment_factor
        elif current_kl < kl_target_min:
            self.kl_beta = self.kl_beta / adjustment_factor
        
        self.kl_beta = max(0.001, min(10.0, self.kl_beta))
        return self.kl_beta


class PointerSelectorV4_7_RLConfig:
    """V4-7 RL 模型配置类"""
    
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
        dpp_rank: int = 32,
        dpp_lambda_init: float = 0.0,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
        advantage_clip: float = 5.0
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
        self.dpp_rank = dpp_rank
        self.dpp_lambda_init = dpp_lambda_init
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.advantage_clip = advantage_clip
    
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
            'dpp_rank': self.dpp_rank,
            'dpp_lambda_init': self.dpp_lambda_init,
            'clip_epsilon': self.clip_epsilon,
            'kl_beta': self.kl_beta,
            'advantage_clip': self.advantage_clip
        }


def build_model_v4_7_rl(config: Optional[PointerSelectorV4_7_RLConfig] = None) -> PointerSelectorV4_7_RL:
    """构建 V4-7 RL 模型的工厂函数"""
    if config is None:
        config = PointerSelectorV4_7_RLConfig()
    
    model = PointerSelectorV4_7_RL(
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
        dpp_rank=config.dpp_rank,
        dpp_lambda_init=config.dpp_lambda_init,
        clip_epsilon=config.clip_epsilon,
        kl_beta=config.kl_beta,
        advantage_clip=config.advantage_clip
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV4_7_RL 模型")
    print("="*70)
    
    batch_size = 4
    d_model = 768
    K = 32
    shot_num = 4  # 测试多 shot 场景
    num_beams = 5
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    labels = torch.randint(0, K, (batch_size, shot_num))
    beam_labels = torch.randint(0, K, (batch_size, num_beams, shot_num))
    beam_rewards = torch.randn(batch_size, num_beams)
    
    # 测试不同的 dpp_rank
    for dpp_rank in [16, 32, 64]:
        print(f"\n{'='*70}")
        print(f"测试 dpp_rank = {dpp_rank}")
        print("="*70)
        
        config = PointerSelectorV4_7_RLConfig(
            shot_num=shot_num,
            dpp_rank=dpp_rank
        )
        model = build_model_v4_7_rl(config)
        
        print(f"\n测试前向传播...")
        result = model(query_emb, cand_emb, labels, return_loss=True)
        print(f"  loss: {result['loss'].item():.4f}")
        
        print(f"\n测试RCE损失...")
        rce_loss = model.compute_rce_loss(
            query_emb, cand_emb, beam_labels, beam_rewards, temperature=1.0
        )
        print(f"  RCE loss: {rce_loss.item():.4f}")
        
        print(f"\n测试GRPO损失...")
        old_log_probs = model.compute_log_probs_per_beam(query_emb, cand_emb, beam_labels).detach()
        grpo_result = model.compute_grpo_loss(
            query_emb, cand_emb, beam_labels, beam_rewards, old_log_probs
        )
        print(f"  GRPO loss: {grpo_result['loss'].item():.4f}")
        
        # 检查 dpp_lambda 的值
        print(f"\n  dpp_lambda (raw): {model.dpp_lambda.item():.4f}")
        print(f"  dpp_lambda (effective): {F.softplus(model.dpp_lambda).item():.4f}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n参数量: {total_params:,}")
    
    print("\n" + "="*70)
    print("✓ V4-7 RL 模型测试通过！")
    print("="*70)
