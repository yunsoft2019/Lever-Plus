"""
Pointer Selector V3: V2 + 离线强化学习（GRPO风格后训练）

特点：
- 基础：继承V2的所有功能（多层 Cross-Attention）
- 后训练：在V2有监督训练后进行GRPO风格后训练
- 阶段1：RCE (Reward-weighted Cross-Entropy) 热身
- 阶段2：GRPO-PPO (Group-Relative Policy Optimization with PPO-style clipping)
- 数据：完全离线，直接使用束搜索的beam和score
- 目标：进一步优化任务指标，最大化高分beam的概率

作者: Lever-Plus Team
日期: 2025-10-29
参考: yiyun.md V3 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import os
from collections import defaultdict
import numpy as np

# 导入V2作为基础
from ..v2.pointer_selector_v2 import PointerSelectorV2, PointerSelectorV2Config


class PointerSelectorV3Config:
    """V3模型配置"""
    
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 6,
        label_smoothing: float = 0.0,
        dropout: float = 0.5,
        hidden_dim: int = 256,
        num_heads: int = 1,
        attn_dropout: float = 0.1,
        num_layers: int = 3,
        # V3特有参数
        enable_rce: bool = False,  # 是否启用RCE
        enable_grpo: bool = False,  # 是否启用GRPO
        rce_temperature: float = 1.0,  # RCE温度
        ppo_epsilon: float = 0.2,  # PPO裁剪参数
        advantage_clip: float = 5.0,  # 优势函数裁剪
        kl_beta: float = 0.01,  # KL散度权重
        reward_norm: str = 'zscore'  # 奖励归一化方式：'zscore' 或 'minmax'
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
        # V3特有
        self.enable_rce = enable_rce
        self.enable_grpo = enable_grpo
        self.rce_temperature = rce_temperature
        self.ppo_epsilon = ppo_epsilon
        self.advantage_clip = advantage_clip
        self.kl_beta = kl_beta
        self.reward_norm = reward_norm

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
            'enable_rce': self.enable_rce,
            'enable_grpo': self.enable_grpo,
            'rce_temperature': self.rce_temperature,
            'ppo_epsilon': self.ppo_epsilon,
            'advantage_clip': self.advantage_clip,
            'kl_beta': self.kl_beta,
            'reward_norm': self.reward_norm
        }


class PointerSelectorV3(PointerSelectorV2):
    """
    V3 版本：V2 + 离线强化学习（GRPO）
    
    继承V2的所有功能，增加：
    1. RCE（Reward-weighted CE）热身
    2. GRPO-PPO后训练
    """
    
    def __init__(
        self,
        d_model: int = 768,
        K: int = 32,
        shot_num: int = 6,
        label_smoothing: float = 0.0,
        dropout: float = 0.5,
        hidden_dim: int = 256,
        num_heads: int = 1,
        attn_dropout: float = 0.1,
        num_layers: int = 3,
        # V3特有参数
        enable_rce: bool = False,
        enable_grpo: bool = False,
        rce_temperature: float = 1.0,
        ppo_epsilon: float = 0.2,
        advantage_clip: float = 5.0,
        kl_beta: float = 0.01,
        reward_norm: str = 'zscore'
    ):
        """
        初始化 V3 模型
        
        Args:
            [V2的所有参数...]
            enable_rce: 是否启用RCE加权
            enable_grpo: 是否启用GRPO
            rce_temperature: RCE温度参数
            ppo_epsilon: PPO裁剪参数
            advantage_clip: 优势函数裁剪范围
            kl_beta: KL散度正则化权重
            reward_norm: 奖励归一化方式
        """
        # 调用V2的初始化
        super().__init__(
            d_model=d_model,
            K=K,
            shot_num=shot_num,
            label_smoothing=label_smoothing,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            num_layers=num_layers
        )
        
        # V3特有属性
        self.enable_rce = enable_rce
        self.enable_grpo = enable_grpo
        self.rce_temperature = rce_temperature
        self.ppo_epsilon = ppo_epsilon
        self.advantage_clip = advantage_clip
        self.kl_beta = kl_beta
        self.reward_norm = reward_norm
        
        print(f"✓ PointerSelectorV3 初始化完成")
        print(f"  - 继承基础: V2")
        print(f"  - RCE启用: {enable_rce}")
        print(f"  - GRPO启用: {enable_grpo}")
        if enable_rce:
            print(f"  - RCE温度: {rce_temperature}")
        if enable_grpo:
            print(f"  - PPO ε: {ppo_epsilon}")
            print(f"  - Advantage clip: ±{advantage_clip}")
            print(f"  - KL β: {kl_beta}")
    
    def compute_log_probs(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算给定轨迹的log概率（用于GRPO）
        
        Args:
            query_emb: [B, d] query编码
            cand_emb: [B, K, d] 候选编码
            labels: [B, S] 轨迹（标签序列）
        
        Returns:
            log_probs: [B] 每个轨迹的总log概率
        """
        batch_size = query_emb.shape[0]
        device = query_emb.device
        
        # 前向传播获取logits
        output = self.forward(
            query_emb=query_emb,
            cand_emb=cand_emb,
            labels=None,  # 推理模式
            return_loss=False
        )
        
        logits = output['logits']  # [B, S, K]
        
        # 计算每步的log概率
        log_probs_per_step = F.log_softmax(logits, dim=-1)  # [B, S, K]
        
        # 收集每步的实际选择的log概率（保持梯度）
        # 使用gather操作而不是循环+.item()
        labels_expanded = labels.unsqueeze(-1)  # [B, S, 1]
        chosen_log_probs = torch.gather(log_probs_per_step, dim=2, index=labels_expanded)  # [B, S, 1]
        chosen_log_probs = chosen_log_probs.squeeze(-1)  # [B, S]
        
        # 裁剪每步的log_prob，防止极端值（-10对应概率约4.5e-5）
        chosen_log_probs = torch.clamp(chosen_log_probs, min=-10.0, max=0.0)
        
        # 对每个轨迹求和
        traj_log_probs = chosen_log_probs.sum(dim=1)  # [B]
        
        return traj_log_probs
    
    def compute_rce_loss(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        计算RCE（Reward-weighted Cross-Entropy）损失
        
        Args:
            query_emb: [B, d]
            cand_emb: [B, K, d]
            labels: [B, S]
            rewards: [B] 每个轨迹的奖励
        
        Returns:
            rce_loss: 加权CE损失
        """
        # 计算权重：softmax(reward / temperature)
        weights = F.softmax(rewards / self.rce_temperature, dim=0)  # [B]
        
        # 前向传播获取logits
        output = self.forward(
            query_emb=query_emb,
            cand_emb=cand_emb,
            labels=labels,
            return_loss=False
        )
        
        logits = output['logits']  # [B, S, K]
        
        # 计算每个样本的CE损失
        batch_size, shot_num, K = logits.shape
        logits_flat = logits.reshape(-1, K)
        labels_flat = labels.reshape(-1)
        
        # 逐样本CE
        ce_losses = []
        for b in range(batch_size):
            sample_logits = logits[b].reshape(-1, K)  # [S, K]
            sample_labels = labels[b].reshape(-1)  # [S]
            sample_ce = F.cross_entropy(sample_logits, sample_labels, reduction='mean')
            ce_losses.append(sample_ce)
        
        ce_losses = torch.stack(ce_losses)  # [B]
        
        # 加权
        rce_loss = (weights * ce_losses).sum()
        
        return rce_loss
    
    def compute_grpo_loss(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算GRPO-PPO损失
        
        Args:
            query_emb: [B, d]
            cand_emb: [B, K, d]
            labels: [B, S] 轨迹
            rewards: [B] 奖励
            old_log_probs: [B] 旧策略的log概率
        
        Returns:
            dict: {'loss': total_loss, 'ppo_loss': ppo_loss, 'kl_loss': kl_loss}
        """
        batch_size = query_emb.shape[0]
        
        # 1. 归一化奖励（添加数值稳定性检查）
        if self.reward_norm == 'zscore':
            reward_std = rewards.std()
            if reward_std < 1e-6:  # 避免除以接近0的数
                normalized_rewards = rewards - rewards.mean()
            else:
                normalized_rewards = (rewards - rewards.mean()) / (reward_std + 1e-8)
        elif self.reward_norm == 'minmax':
            reward_range = rewards.max() - rewards.min()
            if reward_range < 1e-6:  # 避免除以接近0的数
                normalized_rewards = torch.zeros_like(rewards)
            else:
                normalized_rewards = (rewards - rewards.min()) / (reward_range + 1e-8)
        else:
            normalized_rewards = rewards
        
        # 2. 计算组相对优势（Group-Relative Advantage）
        baseline = normalized_rewards.mean()
        advantages = normalized_rewards - baseline  # [B]
        
        # 3. 裁剪优势
        clipped_advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        
        # 4. 计算当前策略的log概率
        current_log_probs = self.compute_log_probs(query_emb, cand_emb, labels)  # [B]
        
        # 5. 计算概率比（添加数值稳定性）
        log_ratio = current_log_probs - old_log_probs
        # 裁剪log_ratio防止exp溢出（±5对应ratio在[0.0067, 148.4]之间）
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
        ratio = torch.exp(log_ratio)  # [B]
        
        # 6. PPO目标（裁剪版本）
        surr1 = ratio * clipped_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * clipped_advantages
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        # 7. KL散度正则（防止偏离太远）
        kl_div = (ratio - 1.0 - log_ratio).mean()  # 近似KL
        kl_loss = self.kl_beta * kl_div
        
        # 8. 总损失
        total_loss = ppo_loss + kl_loss
        
        return {
            'loss': total_loss,
            'ppo_loss': ppo_loss,
            'kl_loss': kl_loss,
            'mean_ratio': ratio.mean(),
            'mean_advantage': advantages.mean()
        }
    
    def forward_with_mode(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        mode: str = 'supervised'  # 'supervised', 'rce', 'grpo'
    ) -> Dict[str, torch.Tensor]:
        """
        根据模式选择损失计算方式
        
        Args:
            mode: 'supervised' (V2标准), 'rce' (RCE加权), 'grpo' (GRPO-PPO)
        """
        if mode == 'supervised':
            # 标准V2监督学习
            return super().forward(query_emb, cand_emb, labels, return_loss=True)
        
        elif mode == 'rce':
            # RCE加权CE
            if rewards is None:
                raise ValueError("RCE模式需要提供rewards")
            
            rce_loss = self.compute_rce_loss(query_emb, cand_emb, labels, rewards)
            return {'loss': rce_loss, 'rce_loss': rce_loss}
        
        elif mode == 'grpo':
            # GRPO-PPO后训练
            if rewards is None or old_log_probs is None:
                raise ValueError("GRPO模式需要提供rewards和old_log_probs")
            
            return self.compute_grpo_loss(query_emb, cand_emb, labels, rewards, old_log_probs)
        
        else:
            raise ValueError(f"未知模式: {mode}")


def build_model_v3(config: PointerSelectorV3Config) -> PointerSelectorV3:
    """构建V3模型的工厂函数"""
    return PointerSelectorV3(**config.to_dict())


def load_v3_from_v2_checkpoint(
    checkpoint_path: str,
    enable_rce: bool = False,
    enable_grpo: bool = False,
    rce_temperature: float = 1.0,
    ppo_epsilon: float = 0.2,
    kl_beta: float = 0.01
) -> PointerSelectorV3:
    """
    从V2 checkpoint加载V3模型（用于后训练）
    
    Args:
        checkpoint_path: V2模型checkpoint路径
        enable_rce: 是否启用RCE
        enable_grpo: 是否启用GRPO
        [其他V3参数...]
    
    Returns:
        初始化好的V3模型
    """
    print(f"从 V2 checkpoint 加载 V3 模型...")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到V2 checkpoint: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 获取配置
    if 'model_config' in checkpoint:
        config_dict = checkpoint['model_config']
    else:
        raise ValueError("Checkpoint中没有model_config")
    
    # 检测是V3还是V2 checkpoint
    is_v3_checkpoint = 'enable_rce' in config_dict or 'enable_grpo' in config_dict
    
    if is_v3_checkpoint:
        # 这是V3 checkpoint，允许覆盖V3参数
        print("检测到V3 checkpoint，直接加载...")
        if enable_rce is not None:
            config_dict['enable_rce'] = enable_rce
        if enable_grpo is not None:
            config_dict['enable_grpo'] = enable_grpo
        if rce_temperature is not None:
            config_dict['rce_temperature'] = rce_temperature
        if ppo_epsilon is not None:
            config_dict['ppo_epsilon'] = ppo_epsilon
        if kl_beta is not None:
            config_dict['kl_beta'] = kl_beta
        
        # 过滤掉不需要的参数（temperature在V3Config中不需要）
        v3_config_dict = {k: v for k, v in config_dict.items() if k != 'temperature'}
        
        # 创建V3配置
        v3_config = PointerSelectorV3Config(**v3_config_dict)
        # 创建V3模型
        model = build_model_v3(v3_config)
        # 加载V3权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 成功加载 V3 参数")
        else:
            raise ValueError("Checkpoint中没有model_state_dict")
        
        print("✓ V3模型已从V3 checkpoint加载")
    else:
        # 这是V2 checkpoint，转换为V3
        print("检测到V2 checkpoint，转换为V3...")
        
        # 创建V3配置（继承V2配置 + V3参数）
        v3_config = PointerSelectorV3Config(
            d_model=config_dict.get('d_model', 768),
            K=config_dict.get('K', 32),
            shot_num=config_dict.get('shot_num', 6),
            label_smoothing=config_dict.get('label_smoothing', 0.0),
            dropout=config_dict.get('dropout', 0.5),
            hidden_dim=config_dict.get('hidden_dim', 256),
            num_heads=config_dict.get('num_heads', 1),
            attn_dropout=config_dict.get('attn_dropout', 0.1),
            num_layers=config_dict.get('num_layers', 3),
            # V3特有
            enable_rce=enable_rce,
            enable_grpo=enable_grpo,
            rce_temperature=rce_temperature,
            ppo_epsilon=ppo_epsilon,
            kl_beta=kl_beta
        )
        
        # 创建V3模型
        model = build_model_v3(v3_config)
        
        # 加载V2的权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✓ 成功加载 V2 参数")
        else:
            raise ValueError("Checkpoint中没有model_state_dict")
        
        print("✓ V3模型已从V2加载并初始化")
    
    return model
