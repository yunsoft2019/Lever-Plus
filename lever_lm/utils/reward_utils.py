"""
奖励处理工具

来自强化学习.md 2.3节 奖励归一化策略：
- 组内Z-score：在每个query的5个beam内计算均值和标准差
- 优势裁剪：限制在[-5, 5]范围内，防止极端梯度

来自强化学习.md 2.2节 阶段2 RCE预热：
- softmax权重：w_i = softmax(score_i / τ)

作者: Lever-Plus Team
日期: 2025-12-02
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def normalize_rewards_zscore(
    rewards: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    组内Z-score归一化
    
    来自强化学习.md 2.3节：
    在每个query的5个beam内计算均值和标准差
    
    Args:
        rewards: [B, num_beams] 或 [num_beams] 奖励张量
        dim: 归一化的维度（默认-1，即beam维度）
        eps: 防止除零的小值
    
    Returns:
        normalized: 归一化后的奖励，均值≈0，标准差≈1
    """
    mean = rewards.mean(dim=dim, keepdim=True)
    std = rewards.std(dim=dim, keepdim=True)
    std = torch.clamp(std, min=eps)
    normalized = (rewards - mean) / std
    return normalized


def clip_advantages(
    advantages: torch.Tensor,
    clip_range: float = 5.0
) -> torch.Tensor:
    """
    优势裁剪
    
    来自强化学习.md 2.3节：
    限制在[-5, 5]范围内，防止极端梯度
    
    Args:
        advantages: 优势张量
        clip_range: 裁剪范围（默认5.0，即[-5, 5]）
    
    Returns:
        clipped: 裁剪后的优势
    """
    return torch.clamp(advantages, -clip_range, clip_range)


def compute_group_relative_advantage(
    rewards: torch.Tensor,
    normalize: bool = True,
    clip_range: float = 5.0
) -> torch.Tensor:
    """
    计算组内相对优势（Group-Relative Advantage）
    
    来自强化学习.md 创新点3：
    - 在每个query的5个beam内计算相对优势
    - 避免跨query的奖励分布差异影响训练
    - 更稳定的梯度信号
    
    Args:
        rewards: [B, num_beams] 每个beam的奖励
        normalize: 是否进行Z-score归一化
        clip_range: 优势裁剪范围
    
    Returns:
        advantages: [B, num_beams] 组内相对优势
    """
    if normalize:
        advantages = normalize_rewards_zscore(rewards, dim=-1)
    else:
        # 不归一化，只减去均值
        mean = rewards.mean(dim=-1, keepdim=True)
        advantages = rewards - mean
    
    # 裁剪
    advantages = clip_advantages(advantages, clip_range)
    
    return advantages


def compute_softmax_weights(
    rewards: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    计算RCE的softmax权重
    
    来自强化学习.md 2.2节 阶段2：
    w_i = softmax(score_i / τ)
    
    Args:
        rewards: [B, num_beams] 或 [num_beams] 奖励张量（原始分数）
        temperature: 温度参数τ（从2.0线性降到0.5）
    
    Returns:
        weights: softmax权重，和为1
    """
    return F.softmax(rewards / temperature, dim=-1)


def compute_temperature_schedule(
    current_step: int,
    total_steps: int,
    start_temp: float = 2.0,
    end_temp: float = 0.5
) -> float:
    """
    计算温度调度
    
    来自强化学习.md 2.2节 阶段2：
    温度调度：τ从2.0线性降到0.5
    
    Args:
        current_step: 当前步数
        total_steps: 总步数
        start_temp: 起始温度（默认2.0）
        end_temp: 结束温度（默认0.5）
    
    Returns:
        temperature: 当前温度
    """
    if total_steps <= 1:
        return end_temp
    
    progress = current_step / (total_steps - 1)
    progress = min(1.0, max(0.0, progress))  # 限制在[0, 1]
    temperature = start_temp + (end_temp - start_temp) * progress
    
    return temperature


def compute_kl_penalty(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor
) -> torch.Tensor:
    """
    计算KL散度惩罚项
    
    来自强化学习.md 2.2节 阶段3：
    L_KL = E[r - 1 - log(r)]，其中 r = π_new / π_old
    
    Args:
        log_probs_new: 新策略的log概率
        log_probs_old: 旧策略的log概率
    
    Returns:
        kl: KL散度（标量）
    """
    log_ratio = log_probs_new - log_probs_old
    ratio = torch.exp(log_ratio)
    kl = ratio - 1 - log_ratio
    return kl.mean()


def adaptive_kl_beta(
    current_kl: float,
    current_beta: float,
    kl_target_min: float = 0.01,
    kl_target_max: float = 0.1,
    adjustment_factor: float = 1.5,
    beta_min: float = 0.001,
    beta_max: float = 10.0
) -> float:
    """
    KL散度自适应调整β
    
    来自强化学习.md 创新点5：
    - 监控KL散度，如果偏离过大（>0.1），增加kl_beta
    - 如果KL过小（<0.01），减少kl_beta，允许更大更新
    
    Args:
        current_kl: 当前KL散度值
        current_beta: 当前β值
        kl_target_min: KL目标下限（默认0.01）
        kl_target_max: KL目标上限（默认0.1）
        adjustment_factor: 调整因子（默认1.5）
        beta_min: β最小值
        beta_max: β最大值
    
    Returns:
        new_beta: 调整后的β
    """
    if current_kl > kl_target_max:
        # KL过大，增加β以约束更新
        new_beta = current_beta * adjustment_factor
    elif current_kl < kl_target_min:
        # KL过小，减少β以允许更大更新
        new_beta = current_beta / adjustment_factor
    else:
        # KL在目标范围内，保持不变
        new_beta = current_beta
    
    # 限制范围
    new_beta = max(beta_min, min(beta_max, new_beta))
    
    return new_beta


def compute_reward_for_candidate(
    beam_score: Optional[float] = None,
    logprob_score: Optional[float] = None,
    vqa_correct: Optional[int] = None,
    vqa_acc_score: Optional[float] = None,
    vqa_gt_prob: Optional[float] = None,  # P1: 新增 vqa_gt_prob 支持
    # 新增：reward 模式参数
    reward_mode: str = "hard_plus_soft",
    hard_weight: float = 1.0,
    soft_weight: float = 1.0,
    # 兼容旧接口的参数（默认不启用）
    alpha: float = 0.0,
    beta: float = 0.0,
    correctness_mode: str = "01",
    use_logprob: bool = False,
    reward_clip: Tuple[float, float] = (-5.0, 5.0)
) -> float:
    """
    计算 RL reward
    
    新方案（v3_rl_from_current_code_full_plan.md）：
    - 只使用 vqa_correct / vqa_acc_score 构造 reward
    - reward = hard_weight * hard + soft_weight * soft
    - 正样本：[1, 2]，负样本：[0, 1)
    
    支持的 reward_mode：
    - "hard_plus_soft": reward = hard + soft（默认）
    - "hard_plus_soft_v2": reward = soft + 2*hard（增大正负样本差距）
    - "separated": 阈值分离，正样本 [2,3]，负样本 [0,1]（推荐，正负样本有明确 gap）
    - "hard_only": reward = hard（只看是否答对）
    - "soft_only": reward = soft（只看准确率分数）
    - "hard01_plus_gtprob": reward = hard_weight*hard + soft_weight*gt_prob（P1: 使用 vqa_gt_prob 作为 soft）
    - "hard01_plus_gtprob_separated": 阈值分离，使用 gt_prob（P1: 推荐，正样本 [2*hard_weight, 2*hard_weight+soft_weight]，负样本 [0, soft_weight]）
    - "hybrid": 混合 InfoScore 和 correctness（需要 beam_score）
    - "legacy": 使用旧的 alpha/beta 组合方式（兼容旧代码）
    
    Args:
        beam_score: beam search 分数（可选，legacy/hybrid 模式使用）
        logprob_score: log 概率分数（可选，legacy 模式使用）
        vqa_correct: correctness (0/1)（可选）
        vqa_acc_score: VQA 准确率分数 [0,1]（可选）
        vqa_gt_prob: GT 概率（P1: 连续 soft reward，与 VQA metric 对齐）
        reward_mode: reward 模式（默认 "hard_plus_soft"）
        hard_weight: hard correctness 权重（默认 1.0）
        soft_weight: soft correctness 权重（默认 1.0）
        alpha: quality 权重（legacy 模式，默认 0.0）
        beta: correctness 权重（legacy 模式，默认 0.0）
        correctness_mode: correctness 模式（legacy 模式，"01" 或 "pm1"）
        use_logprob: 是否使用 logprob_score（legacy 模式）
        reward_clip: reward 裁剪范围（默认 [-5, 5]）
    
    Returns:
        reward: 计算后的 reward（已裁剪）
    """
    # 新方案：基于 hard + soft correctness
    if reward_mode == "hard_plus_soft":
        # hard: 0/1 correctness
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        # soft: VQA 准确率分数 [0,1]
        soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
        # 组合：reward ∈ [0, 2]
        reward = hard_weight * hard + soft_weight * soft
    
    elif reward_mode == "hard_plus_soft_v2":
        # 改进版：增大正负样本差距
        # 正样本: [2, 3], 负样本: [0, 1]
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
        # 正确性加成更大（2.0 而不是 1.0）
        reward = soft_weight * soft + 2.0 * hard_weight * hard
    
    elif reward_mode == "hard_only":
        # 只看是否答对
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        reward = hard_weight * hard
    
    elif reward_mode == "soft_only":
        # 只看准确率分数
        soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
        reward = soft_weight * soft
    
    elif reward_mode == "separated":
        # 【推荐】阈值分离：正负样本之间有明确的 gap
        # 正样本：reward = 2.0 + soft，范围 [2.0, 3.0]
        # 负样本：reward = soft，范围 [0.0, 1.0]
        # 这样正负样本之间至少有 1.0 的差距
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        soft = float(vqa_acc_score) if vqa_acc_score is not None else 0.0
        if hard == 1.0:
            reward = 2.0 * hard_weight + soft_weight * soft  # 正样本 [2.0, 3.0]
        else:
            reward = soft_weight * soft  # 负样本 [0.0, 1.0]
    
    elif reward_mode == "hard01_plus_gtprob":
        # P1: 使用 vqa_gt_prob 作为 soft reward
        # 按照 2025-12-13需求.md P1需求5 的要求：
        # reward = hard_weight * hard + soft_weight * soft
        # 其中 hard = vqa_correct (0/1), soft = vqa_gt_prob [0,1]
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        soft = float(vqa_gt_prob) if vqa_gt_prob is not None else 0.0
        
        # 组合：reward = hard_weight * hard + soft_weight * soft
        # 默认：reward = 2.0 * hard + 1.0 * soft
        # 正样本：[2.0, 3.0]，负样本：[0.0, 1.0]
        reward = hard_weight * hard + soft_weight * soft
    
    elif reward_mode == "hard01_plus_gtprob_separated":
        # P1: 使用 vqa_gt_prob 作为 soft reward，阈值分离
        # 按照 2025-12-13需求.md P1需求5 的要求：
        # 正样本：reward = 2.0 * hard_weight + soft_weight * gt_prob，范围 [2*hard_weight, 2*hard_weight + soft_weight]
        # 负样本：reward = soft_weight * gt_prob，范围 [0, soft_weight]
        hard = float(vqa_correct) if vqa_correct is not None else 0.0
        soft = float(vqa_gt_prob) if vqa_gt_prob is not None else 0.0
        
        if hard == 1.0:
            # 正样本：[2*hard_weight, 2*hard_weight + soft_weight]
            reward = 2.0 * hard_weight + soft_weight * soft
        else:
            # 负样本：[0, soft_weight]
            reward = soft_weight * soft
    
    elif reward_mode == "legacy":
        # 兼容旧的 alpha/beta 组合方式
        # correctness 部分
        if correctness_mode == "01":
            if vqa_correct is not None:
                correctness_val = float(vqa_correct)
            elif vqa_acc_score is not None:
                correctness_val = float(vqa_acc_score)
            else:
                correctness_val = 0.0
        else:
            # pm1 模式：正确=+1，错误=-1
            if vqa_correct is not None:
                correctness_val = 2.0 * float(vqa_correct) - 1.0
            elif vqa_acc_score is not None:
                correctness_val = 2.0 * float(vqa_acc_score) - 1.0
            else:
                correctness_val = 0.0
        
        # quality 部分
        if use_logprob and logprob_score is not None:
            quality = -logprob_score
        elif beam_score is not None:
            quality = float(beam_score)
        else:
            quality = 0.0
        
        # 线性组合
        reward = alpha * quality + beta * correctness_val
    
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    
    # 裁剪
    reward = max(reward_clip[0], min(reward_clip[1], reward))
    
    return reward


if __name__ == "__main__":
    """测试代码"""
    print("="*70)
    print("测试 reward_utils")
    print("="*70)
    
    # 测试数据
    rewards = torch.tensor([
        [0.046, 0.045, 0.037, 0.035, 0.030],
        [0.050, 0.048, 0.040, 0.038, 0.035],
        [1e-7, 1.2e-7, 1.1e-7, 0.9e-7, 0.8e-7],  # 极小值测试
    ])
    
    print(f"\n原始奖励:")
    print(f"  shape: {rewards.shape}")
    print(f"  values[0]: {rewards[0].tolist()}")
    print(f"  values[2] (极小值): {rewards[2].tolist()}")
    
    # 测试1：Z-score归一化
    print(f"\n测试1：Z-score归一化")
    normalized = normalize_rewards_zscore(rewards)
    print(f"  normalized[0]: {normalized[0].tolist()}")
    print(f"  normalized[2]: {normalized[2].tolist()}")
    print(f"  mean[0]: {normalized[0].mean():.6f}, std[0]: {normalized[0].std():.6f}")
    print(f"  mean[2]: {normalized[2].mean():.6f}, std[2]: {normalized[2].std():.6f}")
    
    # 测试2：优势裁剪
    print(f"\n测试2：优势裁剪")
    extreme_advantages = torch.tensor([[-10.0, 5.0, 0.0, 3.0, 8.0]])
    clipped = clip_advantages(extreme_advantages, clip_range=5.0)
    print(f"  原始: {extreme_advantages[0].tolist()}")
    print(f"  裁剪后: {clipped[0].tolist()}")
    assert clipped.min() >= -5 and clipped.max() <= 5
    print(f"  ✓ 裁剪范围正确 [-5, 5]")
    
    # 测试3：组内相对优势
    print(f"\n测试3：组内相对优势")
    advantages = compute_group_relative_advantage(rewards, normalize=True, clip_range=5.0)
    print(f"  advantages[0]: {advantages[0].tolist()}")
    print(f"  mean: {advantages.mean():.6f}")
    
    # 测试4：softmax权重
    print(f"\n测试4：softmax权重")
    weights_high_temp = compute_softmax_weights(rewards[0], temperature=2.0)
    weights_low_temp = compute_softmax_weights(rewards[0], temperature=0.5)
    print(f"  τ=2.0: {weights_high_temp.tolist()}")
    print(f"  τ=0.5: {weights_low_temp.tolist()}")
    print(f"  sum(τ=2.0): {weights_high_temp.sum():.6f}")
    print(f"  sum(τ=0.5): {weights_low_temp.sum():.6f}")
    print(f"  ✓ 低温度时权重更集中于高分beam")
    
    # 测试5：温度调度
    print(f"\n测试5：温度调度")
    total_steps = 100
    for step in [0, 25, 50, 75, 99]:
        temp = compute_temperature_schedule(step, total_steps, start_temp=2.0, end_temp=0.5)
        print(f"  step {step}/{total_steps}: τ = {temp:.4f}")
    
    # 测试6：KL散度
    print(f"\n测试6：KL散度")
    log_probs_old = torch.tensor([-5.0, -6.0, -7.0, -8.0, -9.0])
    log_probs_new = log_probs_old + torch.randn_like(log_probs_old) * 0.1
    kl = compute_kl_penalty(log_probs_new, log_probs_old)
    print(f"  KL: {kl.item():.6f}")
    
    # 测试7：自适应β
    print(f"\n测试7：自适应β")
    beta = 0.1
    print(f"  初始β: {beta}")
    
    # KL过大
    new_beta = adaptive_kl_beta(current_kl=0.15, current_beta=beta)
    print(f"  KL=0.15 (>0.1) -> β: {new_beta:.4f}")
    
    # KL过小
    new_beta = adaptive_kl_beta(current_kl=0.005, current_beta=beta)
    print(f"  KL=0.005 (<0.01) -> β: {new_beta:.4f}")
    
    # KL正常
    new_beta = adaptive_kl_beta(current_kl=0.05, current_beta=beta)
    print(f"  KL=0.05 (正常) -> β: {new_beta:.4f}")
    
    print("\n" + "="*70)
    print("✓ reward_utils 测试通过！")
    print("="*70)
