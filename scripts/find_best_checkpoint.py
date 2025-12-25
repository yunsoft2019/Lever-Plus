#!/usr/bin/env python3
"""
根据训练日志找到最优的 GRPO checkpoint

使用方法:
    python scripts/find_best_checkpoint.py --log_file <训练日志文件> --checkpoint_dir <checkpoint目录>
    
或者直接指定 checkpoint 目录，脚本会自动查找训练日志:
    python scripts/find_best_checkpoint.py --checkpoint_dir ./results/okvqa/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct
"""

import argparse
import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Tuple


def parse_training_log(log_file: str) -> List[Dict]:
    """
    解析训练日志，提取每个 epoch 的指标
    
    Returns:
        List of dicts, each containing epoch metrics
    """
    metrics_list = []
    
    if not os.path.exists(log_file):
        print(f"警告: 训练日志文件不存在: {log_file}")
        return metrics_list
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找 GRPO 训练阶段的开始
    grpo_start = False
    header_found = False
    
    for line in lines:
        # 检测 GRPO 训练阶段开始
        if "阶段3：GRPO训练" in line or "GRPO训练" in line:
            grpo_start = True
            continue
        
        if not grpo_start:
            continue
        
        # 查找表头
        if "Epoch" in line and "Train Loss" in line and "Val Loss" in line:
            header_found = True
            continue
        
        if not header_found:
            continue
        
        # 跳过分隔线
        if line.strip().startswith("-"):
            continue
        
        # 解析 epoch 数据行
        # 格式: epoch_num train_loss val_loss ppo_loss kl adv_std adv_max beta
        pattern = r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.match(pattern, line.strip())
        
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            ppo_loss = float(match.group(4))
            kl = float(match.group(5))
            adv_std = float(match.group(6))
            adv_max = float(match.group(7))
            beta = float(match.group(8))
            
            metrics_list.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'ppo_loss': ppo_loss,
                'kl': kl,
                'adv_std': adv_std,
                'adv_max': adv_max,
                'beta': beta
            })
    
    return metrics_list


def find_best_checkpoint(metrics_list: List[Dict], strategy: str = 'val_loss') -> Tuple[int, Dict]:
    """
    根据策略找到最优的 checkpoint
    
    Args:
        metrics_list: 训练指标列表
        strategy: 选择策略
            - 'val_loss': 验证集损失最小（推荐）
            - 'ppo_loss': PPO 损失最小
            - 'kl': KL 散度适中（0.01-0.1）
            - 'adv_std': Advantage 标准差最大（梯度信号最强）
    
    Returns:
        (best_epoch, best_metrics)
    """
    if not metrics_list:
        return None, None
    
    if strategy == 'val_loss':
        # 验证集损失最小
        best_idx = min(range(len(metrics_list)), key=lambda i: metrics_list[i]['val_loss'])
    elif strategy == 'ppo_loss':
        # PPO 损失最小
        best_idx = min(range(len(metrics_list)), key=lambda i: metrics_list[i]['ppo_loss'])
    elif strategy == 'kl':
        # KL 散度最接近目标范围（0.01-0.1）
        def kl_score(m):
            kl = m['kl']
            if 0.01 <= kl <= 0.1:
                return abs(kl - 0.05)  # 最接近 0.05
            else:
                return float('inf')
        best_idx = min(range(len(metrics_list)), key=lambda i: kl_score(metrics_list[i]))
    elif strategy == 'adv_std':
        # Advantage 标准差最大（梯度信号最强）
        best_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['adv_std'])
    else:
        raise ValueError(f"未知的策略: {strategy}")
    
    best_epoch = metrics_list[best_idx]['epoch']
    best_metrics = metrics_list[best_idx]
    
    return best_epoch, best_metrics


def find_log_file(checkpoint_dir: str) -> str:
    """
    在 checkpoint 目录中查找训练日志文件
    """
    # 可能的日志文件名模式
    patterns = [
        '*.log',
        'training.log',
        'train.log',
        '*.txt'
    ]
    
    log_files = []
    for pattern in patterns:
        log_files.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))
    
    if log_files:
        # 返回最新的日志文件
        return max(log_files, key=os.path.getmtime)
    
    # 如果没找到，尝试在父目录查找
    parent_dir = os.path.dirname(checkpoint_dir)
    for pattern in patterns:
        log_files.extend(glob.glob(os.path.join(parent_dir, pattern)))
    
    if log_files:
        return max(log_files, key=os.path.getmtime)
    
    return None


def main():
    parser = argparse.ArgumentParser(description="找到最优的 GRPO checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Checkpoint 目录路径")
    parser.add_argument("--log_file", type=str, default=None,
                       help="训练日志文件路径（可选，如果不提供会自动查找）")
    parser.add_argument("--strategy", type=str, default="val_loss",
                       choices=['val_loss', 'ppo_loss', 'kl', 'adv_std'],
                       help="选择最优 checkpoint 的策略（默认: val_loss）")
    parser.add_argument("--show_all", action="store_true",
                       help="显示所有 epoch 的指标")
    
    args = parser.parse_args()
    
    # 查找日志文件
    if args.log_file is None:
        args.log_file = find_log_file(args.checkpoint_dir)
    
    if args.log_file is None:
        print("错误: 无法找到训练日志文件")
        print(f"请手动指定日志文件: --log_file <路径>")
        return
    
    print(f"使用训练日志: {args.log_file}")
    
    # 解析训练日志
    metrics_list = parse_training_log(args.log_file)
    
    if not metrics_list:
        print("警告: 无法从日志中解析到训练指标")
        print("可能的原因:")
        print("  1. 训练还未完成")
        print("  2. 日志格式不匹配")
        print("  3. GRPO 训练阶段未开始")
        return
    
    print(f"\n找到 {len(metrics_list)} 个 epoch 的训练指标")
    
    # 显示所有指标（如果请求）
    if args.show_all:
        print("\n所有 epoch 的指标:")
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'PPO Loss':>12} {'KL':>10} {'Adv Std':>10}")
        print("-" * 70)
        for m in metrics_list:
            print(f"{m['epoch']:6d} {m['train_loss']:12.5f} {m['val_loss']:12.5f} "
                  f"{m['ppo_loss']:12.5f} {m['kl']:10.5f} {m['adv_std']:10.4f}")
    
    # 找到最优 checkpoint
    best_epoch, best_metrics = find_best_checkpoint(metrics_list, args.strategy)
    
    if best_epoch is None:
        print("错误: 无法找到最优 checkpoint")
        return
    
    print(f"\n{'='*70}")
    print(f"最优 checkpoint (策略: {args.strategy}):")
    print(f"{'='*70}")
    print(f"  Epoch: {best_epoch}")
    print(f"  Checkpoint 文件: grpo_epoch{best_epoch}.pt")
    print(f"  完整路径: {os.path.join(args.checkpoint_dir, f'grpo_epoch{best_epoch}.pt')}")
    print(f"\n指标:")
    print(f"  Train Loss: {best_metrics['train_loss']:.5f}")
    print(f"  Val Loss: {best_metrics['val_loss']:.5f}")
    print(f"  PPO Loss: {best_metrics['ppo_loss']:.5f}")
    print(f"  KL: {best_metrics['kl']:.5f}")
    print(f"  Adv Std: {best_metrics['adv_std']:.4f}")
    print(f"  Adv Max: {best_metrics['adv_max']:.4f}")
    print(f"  Beta: {best_metrics['beta']:.4f}")
    print(f"{'='*70}")
    
    # 检查 checkpoint 文件是否存在
    checkpoint_path = os.path.join(args.checkpoint_dir, f"grpo_epoch{best_epoch}.pt")
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Checkpoint 文件存在")
    else:
        print(f"\n⚠️  警告: Checkpoint 文件不存在: {checkpoint_path}")
        print("   可能训练还未完成，或者 checkpoint 文件名格式不同")
    
    # 显示其他策略的结果（供参考）
    print(f"\n其他策略的结果（供参考）:")
    for strategy in ['val_loss', 'ppo_loss', 'kl', 'adv_std']:
        if strategy == args.strategy:
            continue
        epoch, metrics = find_best_checkpoint(metrics_list, strategy)
        if epoch:
            print(f"  {strategy:10s}: Epoch {epoch:3d} (Val Loss: {metrics['val_loss']:.5f}, "
                  f"PPO Loss: {metrics['ppo_loss']:.5f}, KL: {metrics['kl']:.5f})")


if __name__ == "__main__":
    main()


