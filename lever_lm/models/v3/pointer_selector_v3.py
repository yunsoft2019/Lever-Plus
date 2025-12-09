"""
Pointer Selector V3: V2 + 离线强化学习（RCE预热 + GRPO后训练）

特点：
- 继承V2的Bi-Encoder + Cross-Attention架构
- 新增RCE（Reward-Conditioned Estimation）预热损失
- 新增GRPO（Group-Relative Policy Optimization）策略梯度损失
- 支持组内相对优势计算
- 支持KL散度计算和自适应调整

来自强化学习.md的核心算法：
- RCE损失：L_RCE = Σ w_i * CE(π_new, labels_i)，其中 w_i = softmax(score_i / τ)
- GRPO损失：L_GRPO = L_PPO + β * L_KL
  - L_PPO = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
  - L_KL = E[r - 1 - log(r)]（近似KL散度）

作者: Lever-Plus Team
日期: 2025-12-02
参考: 强化学习.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from ..v2.pointer_selector_v2 import PointerSelectorV2


class PointerSelectorV3(PointerSelectorV2):
    """
    V3 版本：V2 + 离线强化学习（RCE预热 + GRPO后训练）
    
    继承V2的Bi-Encoder + Cross-Attention架构，新增强化学习相关方法：
    - compute_rce_loss(): RCE预热损失
    - compute_grpo_loss(): GRPO策略梯度损失
    - compute_advantage(): 组内相对优势计算
    - compute_kl_divergence(): KL散度计算
    
    来自强化学习.md 2.1节的5个创新点：
    1. 多层级奖励设计
    2. 自适应温度调度
    3. 组内相对优势（Group-Relative Advantage）
    4. 课程学习策略
    5. KL散度自适应调整
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
        num_layers: int = 3,
        # V3新增参数
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
        advantage_clip: float = 5.0
    ):
        """
        初始化 V3 模型
        
        Args:
            d_model: 输入 embedding 维度 (默认 768, CLIP ViT-L/14 输出)
            K: 候选池大小 (默认 32)
            shot_num: 需要选择的样本数量 (默认 2)
            label_smoothing: 标签平滑系数 (默认 0.1)
            dropout: dropout 比例 (默认 0.1)
            hidden_dim: 输出维度 (默认 256)
            num_heads: Cross-Attention 的头数 (默认 1)
            attn_dropout: Attention 层的 dropout (默认 0.1)
            num_layers: Cross-Attention 的层数 (默认 3，V2使用3层)
            clip_epsilon: PPO裁剪参数ε (默认 0.2)
            kl_beta: KL散度权重β (默认 0.1)
            advantage_clip: 优势裁剪范围 (默认 5.0，即[-5, 5])
        """
        # 调用V2的初始化（不打印V2的信息）
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
        
        # V3新增参数
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.advantage_clip = advantage_clip
        
        print(f"✓ PointerSelectorV3 初始化完成（继承V2 + 强化学习）")
        print(f"  - clip_epsilon (PPO裁剪): {clip_epsilon}")
        print(f"  - kl_beta (KL权重): {kl_beta}")
        print(f"  - advantage_clip: [-{advantage_clip}, {advantage_clip}]")
    
    def compute_log_probs(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算给定标签序列的log概率
        
        Args:
            query_emb: [B, d] query embedding
            cand_emb: [B, K, d] 候选 embedding
            labels: [B, S] 标签序列
        
        Returns:
            log_probs: [B] 每个样本的log概率（所有步骤的和）
        """
        result = self.forward(query_emb, cand_emb, labels=labels, return_loss=False)
        logits = result['logits']  # [B, S, K]
        
        batch_size, shot_num, K = logits.shape
        
        # 计算每步的log softmax
        log_probs_all = F.log_softmax(logits, dim=-1)  # [B, S, K]
        
        # 收集每步选择的log概率
        # labels: [B, S] -> [B, S, 1]
        labels_expanded = labels.unsqueeze(-1)  # [B, S, 1]
        selected_log_probs = log_probs_all.gather(dim=-1, index=labels_expanded)  # [B, S, 1]
        selected_log_probs = selected_log_probs.squeeze(-1)  # [B, S]
        
        # 对所有步骤求和得到序列的log概率
        seq_log_probs = selected_log_probs.sum(dim=-1)  # [B]
        
        return seq_log_probs
    
    def compute_log_probs_per_beam(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个beam的log概率
        
        Args:
            query_emb: [B, d] query embedding
            cand_emb: [B, K, d] 候选 embedding
            beam_labels: [B, num_beams, S] 多个beam的标签序列
        
        Returns:
            log_probs: [B, num_beams] 每个beam的log概率
        """
        batch_size, num_beams, shot_num = beam_labels.shape
        
        # 展开beam维度
        # query_emb: [B, d] -> [B*num_beams, d]
        query_expanded = query_emb.unsqueeze(1).expand(-1, num_beams, -1)
        query_expanded = query_expanded.reshape(batch_size * num_beams, -1)
        
        # cand_emb: [B, K, d] -> [B*num_beams, K, d]
        cand_expanded = cand_emb.unsqueeze(1).expand(-1, num_beams, -1, -1)
        cand_expanded = cand_expanded.reshape(batch_size * num_beams, self.K, -1)
        
        # beam_labels: [B, num_beams, S] -> [B*num_beams, S]
        labels_flat = beam_labels.reshape(batch_size * num_beams, shot_num)
        
        # 计算log概率
        log_probs_flat = self.compute_log_probs(query_expanded, cand_expanded, labels_flat)
        
        # 恢复形状: [B*num_beams] -> [B, num_beams]
        log_probs = log_probs_flat.reshape(batch_size, num_beams)
        
        return log_probs
    
    def compute_advantage(
        self,
        rewards: torch.Tensor,
        normalize: bool = True,
        use_rank: bool = False,
        min_std: float = 0.1
    ) -> torch.Tensor:
        """
        计算组内相对优势（Group-Relative Advantage）
        
        来自强化学习.md 创新点3：
        - 在每个query的5个beam内计算相对优势
        - 避免跨query的奖励分布差异影响训练
        - 更稳定的梯度信号
        
        来自强化学习.md 2.3节 奖励归一化策略：
        - 组内Z-score：在每个query的5个beam内计算均值和标准差
        - 优势裁剪：限制在[-5, 5]范围内，防止极端梯度
        
        【优化】默认使用 Z-score 归一化而非排名归一化：
        - 排名归一化会丢失原始 reward 的绝对差异信息
        - 当所有候选都是负样本时，排名归一化仍会产生 [-1, 1] 的 advantage
        - Z-score 归一化保留了 reward 的相对大小关系
        - 设置 min_std 避免除零，同时保证有意义的 advantage
        
        Args:
            rewards: [B, num_beams] 每个beam的奖励（分数）
            normalize: 是否进行组内Z-score归一化
            use_rank: 是否使用基于排名的归一化（默认 False）
            min_std: Z-score 归一化时的最小标准差（默认 0.1）
        
        Returns:
            advantages: [B, num_beams] 组内相对优势
        """
        batch_size, num_beams = rewards.shape
        
        if use_rank:
            # 基于排名的归一化（可选，适用于 reward 差异极小的情况）
            # 获取排名（降序，最高reward排名为0）
            ranks = rewards.argsort(dim=-1, descending=True).argsort(dim=-1).float()
            # 归一化到 [-1, 1]
            # 排名最高(0) -> 1.0, 排名最低(num_beams-1) -> -1.0
            if num_beams > 1:
                advantages = 1.0 - 2.0 * ranks / (num_beams - 1)
            else:
                advantages = torch.zeros_like(ranks)
        else:
            # 【推荐】Z-score 归一化
            # 保留原始 reward 的绝对差异信息
            mean = rewards.mean(dim=-1, keepdim=True)
            
            if normalize:
                std = rewards.std(dim=-1, keepdim=True)
                # 设置最小 std，避免除零，同时保证有意义的 advantage
                std = torch.clamp(std, min=min_std)
                advantages = (rewards - mean) / std
            else:
                # 不归一化，只减去均值
                advantages = rewards - mean
        
        # 优势裁剪：限制在[-advantage_clip, advantage_clip]范围内
        advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        
        return advantages
    
    def compute_kl_divergence(
        self,
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor
    ) -> torch.Tensor:
        """
        计算KL散度（近似）
        
        来自强化学习.md 2.2节 阶段3：
        L_KL = E[r - 1 - log(r)]，其中 r = π_new / π_old = exp(log_π_new - log_π_old)
        
        这是KL散度的一阶近似：KL(π_new || π_old) ≈ E[r - 1 - log(r)]
        
        Args:
            log_probs_new: [B, num_beams] 新策略的log概率
            log_probs_old: [B, num_beams] 旧策略的log概率
        
        Returns:
            kl: 标量，平均KL散度
        """
        # 计算概率比 r = π_new / π_old
        log_ratio = log_probs_new - log_probs_old
        ratio = torch.exp(log_ratio)
        
        # KL散度近似：r - 1 - log(r)
        kl = ratio - 1 - log_ratio
        
        # 返回平均KL散度
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
        """
        计算RCE（Reward-Conditioned Estimation）预热损失
        
        来自强化学习.md 2.2节 阶段2：RCE预热
        - 目标：稳定地从监督学习过渡到强化学习
        - 损失：L_RCE = Σ w_i * CE(π_new, labels_i)
        - 其中 w_i = softmax(score_i / τ)
        
        【优化】默认关闭排名归一化：
        - 当使用 hard_plus_soft_v2 等 reward 模式时，reward 差异足够大
        - 排名归一化会丢失 reward 的绝对差异信息
        - 直接使用 reward 计算 softmax 权重更合理
        
        Args:
            query_emb: [B, d] query embedding
            cand_emb: [B, K, d] 候选 embedding
            beam_labels: [B, num_beams, S] 多个beam的标签序列
            beam_rewards: [B, num_beams] 每个beam的奖励（原始分数，未归一化）
            temperature: 温度参数τ（从2.0线性降到0.5）
            use_rank_normalization: 是否使用排名归一化（默认False）
            use_top1_only: 是否只使用Top-1 beam（回归V2监督学习方式）
        
        Returns:
            loss: 标量，RCE损失
        """
        batch_size, num_beams, shot_num = beam_labels.shape
        
        # 【回归V2方式】只使用Top-1 beam进行监督学习
        if use_top1_only:
            # 只取第一个beam（已按分数降序排列，第一个是最好的）
            labels = beam_labels[:, 0, :]  # [B, S]
            
            # 直接计算交叉熵损失
            result = self.forward(query_emb, cand_emb, labels=labels, return_loss=False)
            logits = result['logits']  # [B, S, K]
            
            logits_for_loss = logits.reshape(-1, self.K)  # [B*S, K]
            labels_for_loss = labels.reshape(-1)  # [B*S]
            logits_for_loss = torch.clamp(logits_for_loss, min=-100.0)
            
            loss = F.cross_entropy(
                logits_for_loss, labels_for_loss,
                label_smoothing=self.label_smoothing
            )
            return loss
        
        if use_rank_normalization:
            # 【修复】使用排名归一化
            # 将分数转换为排名，然后归一化到[0, 1]
            # 排名越高（分数越大），归一化值越大
            ranks = beam_rewards.argsort(dim=-1).argsort(dim=-1).float()  # [B, num_beams]
            # 归一化排名到[0, 1]，最高排名为1，最低排名为0
            normalized_scores = ranks / (num_beams - 1) if num_beams > 1 else ranks
            # 使用归一化排名计算权重
            weights = F.softmax(normalized_scores / temperature, dim=-1)  # [B, num_beams]
        else:
            # 原始方式：直接使用分数
            weights = F.softmax(beam_rewards / temperature, dim=-1)  # [B, num_beams]
        
        # 批量计算所有beam的损失（更高效）
        # 展开beam维度: [B, num_beams, S] -> [B*num_beams, S]
        labels_flat = beam_labels.reshape(batch_size * num_beams, shot_num)
        
        # 扩展query和cand: [B, d] -> [B*num_beams, d]
        query_expanded = query_emb.unsqueeze(1).expand(-1, num_beams, -1)
        query_expanded = query_expanded.reshape(batch_size * num_beams, -1)
        
        cand_expanded = cand_emb.unsqueeze(1).expand(-1, num_beams, -1, -1)
        cand_expanded = cand_expanded.reshape(batch_size * num_beams, self.K, -1)
        
        # 前向传播
        result = self.forward(query_expanded, cand_expanded, labels=labels_flat, return_loss=False)
        logits = result['logits']  # [B*num_beams, S, K]
        
        # 计算每个样本的交叉熵损失
        logits_for_loss = logits.reshape(-1, self.K)  # [B*num_beams*S, K]
        labels_for_loss = labels_flat.reshape(-1)  # [B*num_beams*S]
        logits_for_loss = torch.clamp(logits_for_loss, min=-100.0)
        
        # 不使用reduction，得到每个元素的损失
        ce_losses = F.cross_entropy(
            logits_for_loss, labels_for_loss,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )  # [B*num_beams*S]
        
        # 重塑并对shot维度求和: [B*num_beams*S] -> [B, num_beams]
        ce_losses = ce_losses.reshape(batch_size, num_beams, shot_num).sum(dim=-1)  # [B, num_beams]
        
        # 加权求和
        weighted_loss = (weights * ce_losses).sum(dim=-1).mean()  # 标量
        
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
        """
        计算GRPO（Group-Relative Policy Optimization）损失
        
        来自强化学习.md 2.2节 阶段3：GRPO训练
        - 目标：最大化高分beam的概率，同时保持策略稳定性
        - 损失：L_GRPO = L_PPO + β * L_KL
          - L_PPO = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
          - L_KL = E[r - 1 - log(r)]
        
        来自强化学习.md 创新点4 课程学习策略：
        - 阶段2（GRPO早期）：只使用top-3 beam，减少噪声
        - 阶段3（GRPO后期）：使用所有beam，精细优化
        
        Args:
            query_emb: [B, d] query embedding
            cand_emb: [B, K, d] 候选 embedding
            beam_labels: [B, num_beams, S] 多个beam的标签序列
            beam_rewards: [B, num_beams] 每个beam的奖励（分数）
            old_log_probs: [B, num_beams] 旧策略（SFT模型）的log概率
            use_top_k: 只使用top-k个beam（课程学习），None表示使用所有beam
        
        Returns:
            dict: {
                'loss': 总损失,
                'ppo_loss': PPO损失,
                'kl_loss': KL损失,
                'kl': KL散度值,
                'mean_ratio': 平均概率比,
                'mean_advantage': 平均优势
            }
        """
        batch_size, num_beams, shot_num = beam_labels.shape
        
        # 课程学习：只使用top-k个beam
        if use_top_k is not None and use_top_k < num_beams:
            # 按奖励排序，选择top-k
            _, top_indices = beam_rewards.topk(use_top_k, dim=-1)  # [B, use_top_k]
            
            # 收集top-k的数据
            beam_labels = torch.gather(
                beam_labels, 
                dim=1, 
                index=top_indices.unsqueeze(-1).expand(-1, -1, shot_num)
            )
            beam_rewards = torch.gather(beam_rewards, dim=1, index=top_indices)
            old_log_probs = torch.gather(old_log_probs, dim=1, index=top_indices)
            num_beams = use_top_k
        
        # 计算新策略的log概率
        new_log_probs = self.compute_log_probs_per_beam(query_emb, cand_emb, beam_labels)
        
        # 计算组内相对优势
        # 注意：beam_rewards应该是原始分数，这里进行归一化
        advantages = self.compute_advantage(beam_rewards, normalize=True)
        
        # 计算概率比 r = π_new / π_old
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # PPO裁剪目标
        # L_PPO = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # 计算两个目标
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        
        # 取最小值（悲观估计）
        ppo_obj = torch.min(obj1, obj2)
        
        # PPO损失（取负号，因为我们要最大化目标）
        ppo_loss = -ppo_obj.mean()
        
        # 计算KL散度
        kl = self.compute_kl_divergence(new_log_probs, old_log_probs)
        kl_loss = self.kl_beta * kl
        
        # 总损失
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
        """
        KL散度自适应调整
        
        来自强化学习.md 创新点5：
        - 监控KL散度，如果偏离过大（>0.1），增加kl_beta
        - 如果KL过小（<0.01），减少kl_beta，允许更大更新
        
        Args:
            current_kl: 当前KL散度值
            kl_target_min: KL目标下限 (默认 0.01)
            kl_target_max: KL目标上限 (默认 0.1)
            adjustment_factor: 调整因子 (默认 1.5)
        
        Returns:
            new_kl_beta: 调整后的kl_beta
        """
        if current_kl > kl_target_max:
            # KL过大，增加kl_beta以约束更新
            self.kl_beta = self.kl_beta * adjustment_factor
        elif current_kl < kl_target_min:
            # KL过小，减少kl_beta以允许更大更新
            self.kl_beta = self.kl_beta / adjustment_factor
        
        # 限制kl_beta的范围，防止极端值
        self.kl_beta = max(0.001, min(10.0, self.kl_beta))
        
        return self.kl_beta


class PointerSelectorV3Config:
    """V3 模型配置类
    
    继承V2配置，新增强化学习相关参数
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
        num_layers: int = 3,
        # V3新增参数
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
        # V3新增
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
            'clip_epsilon': self.clip_epsilon,
            'kl_beta': self.kl_beta,
            'advantage_clip': self.advantage_clip
        }


def build_model_v3(config: Optional[PointerSelectorV3Config] = None) -> PointerSelectorV3:
    """
    构建 V3 模型的工厂函数
    
    Args:
        config: 模型配置（可选）
    
    Returns:
        PointerSelectorV3 实例
    """
    if config is None:
        config = PointerSelectorV3Config()
    
    model = PointerSelectorV3(
        d_model=config.d_model,
        K=config.K,
        shot_num=config.shot_num,
        label_smoothing=config.label_smoothing,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        attn_dropout=config.attn_dropout,
        num_layers=config.num_layers,
        clip_epsilon=config.clip_epsilon,
        kl_beta=config.kl_beta,
        advantage_clip=config.advantage_clip
    )
    
    return model


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 PointerSelectorV3 模型")
    print("="*70)
    
    # 创建模型
    model = build_model_v3()
    
    # 创建测试数据
    batch_size = 4
    d_model = 768
    K = 32
    shot_num = 2
    num_beams = 5  # 5个beam
    
    query_emb = torch.randn(batch_size, d_model)
    cand_emb = torch.randn(batch_size, K, d_model)
    
    # 单个标签（用于基础测试）
    labels = torch.randint(0, K, (batch_size, shot_num))
    
    # 多个beam的标签和奖励
    beam_labels = torch.randint(0, K, (batch_size, num_beams, shot_num))
    beam_rewards = torch.tensor([
        [0.046, 0.045, 0.037, 0.035, 0.030],
        [0.050, 0.048, 0.040, 0.038, 0.035],
        [0.042, 0.041, 0.039, 0.036, 0.032],
        [0.055, 0.052, 0.045, 0.042, 0.038]
    ])
    
    print(f"\n输入形状:")
    print(f"  query_emb: {query_emb.shape}")
    print(f"  cand_emb: {cand_emb.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  beam_labels: {beam_labels.shape}")
    print(f"  beam_rewards: {beam_rewards.shape}")
    
    # 测试1：基础前向传播（继承自V2）
    print(f"\n测试1：基础前向传播...")
    result = model(query_emb, cand_emb, labels, return_loss=True)
    print(f"  logits: {result['logits'].shape}")
    print(f"  predictions: {result['predictions'].shape}")
    print(f"  loss: {result['loss'].item():.4f}")
    
    # 测试2：计算log概率
    print(f"\n测试2：计算log概率...")
    log_probs = model.compute_log_probs(query_emb, cand_emb, labels)
    print(f"  log_probs: {log_probs.shape}, values: {log_probs.tolist()}")
    
    # 测试3：计算每个beam的log概率
    print(f"\n测试3：计算每个beam的log概率...")
    beam_log_probs = model.compute_log_probs_per_beam(query_emb, cand_emb, beam_labels)
    print(f"  beam_log_probs: {beam_log_probs.shape}")
    print(f"  values: {beam_log_probs[0].tolist()}")
    
    # 测试4：计算组内相对优势
    print(f"\n测试4：计算组内相对优势...")
    advantages = model.compute_advantage(beam_rewards, normalize=True)
    print(f"  advantages: {advantages.shape}")
    print(f"  values: {advantages[0].tolist()}")
    print(f"  mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")
    assert advantages.min() >= -5 and advantages.max() <= 5, "优势应在[-5, 5]范围内"
    print(f"  ✓ 优势值在[-5, 5]范围内")
    
    # 测试5：计算RCE损失
    print(f"\n测试5：计算RCE损失...")
    rce_loss = model.compute_rce_loss(
        query_emb, cand_emb, beam_labels, beam_rewards, temperature=2.0
    )
    print(f"  RCE loss (τ=2.0): {rce_loss.item():.4f}")
    
    rce_loss_low_temp = model.compute_rce_loss(
        query_emb, cand_emb, beam_labels, beam_rewards, temperature=0.5
    )
    print(f"  RCE loss (τ=0.5): {rce_loss_low_temp.item():.4f}")
    
    # 测试6：计算GRPO损失
    print(f"\n测试6：计算GRPO损失...")
    # 模拟旧策略的log概率
    old_log_probs = beam_log_probs.detach() + torch.randn_like(beam_log_probs) * 0.1
    
    grpo_result = model.compute_grpo_loss(
        query_emb, cand_emb, beam_labels, beam_rewards, old_log_probs
    )
    print(f"  GRPO total loss: {grpo_result['loss'].item():.4f}")
    print(f"  PPO loss: {grpo_result['ppo_loss'].item():.4f}")
    print(f"  KL loss: {grpo_result['kl_loss'].item():.4f}")
    print(f"  KL: {grpo_result['kl'].item():.4f}")
    print(f"  mean_ratio: {grpo_result['mean_ratio'].item():.4f}")
    print(f"  mean_advantage: {grpo_result['mean_advantage'].item():.4f}")
    
    # 测试7：课程学习（只使用top-3 beam）
    print(f"\n测试7：课程学习（只使用top-3 beam）...")
    grpo_result_top3 = model.compute_grpo_loss(
        query_emb, cand_emb, beam_labels, beam_rewards, old_log_probs,
        use_top_k=3
    )
    print(f"  GRPO loss (top-3): {grpo_result_top3['loss'].item():.4f}")
    
    # 测试8：KL自适应调整
    print(f"\n测试8：KL自适应调整...")
    print(f"  初始 kl_beta: {model.kl_beta}")
    
    # 模拟KL过大
    model.update_kl_beta(current_kl=0.15, kl_target_max=0.1)
    print(f"  KL=0.15 (>0.1) 后 kl_beta: {model.kl_beta:.4f}")
    
    # 模拟KL过小
    model.kl_beta = 0.1  # 重置
    model.update_kl_beta(current_kl=0.005, kl_target_min=0.01)
    print(f"  KL=0.005 (<0.01) 后 kl_beta: {model.kl_beta:.4f}")
    
    print("\n" + "="*70)
    print("✓ V3 模型测试通过！")
    print("="*70)
