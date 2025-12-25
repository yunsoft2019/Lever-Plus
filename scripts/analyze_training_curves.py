#!/usr/bin/env python3
"""
分析训练曲线，对比两个模型的训练策略
"""

import json
import numpy as np
from pathlib import Path

def load_metrics(json_file):
    """加载训练指标"""
    with open(json_file, 'r') as f:
        return json.load(f)

def safe_get(metrics, key, default=0):
    """安全获取指标值"""
    value = metrics.get(key, default)
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return default
    return value

def print_statistics(metrics1, metrics2, name1="KL_BETA=0.12", name2="KL_BETA=0.15"):
    """打印统计信息"""
    print("\n" + "=" * 80)
    print("训练曲线统计对比")
    print("=" * 80)
    
    # 提取所有指标
    epochs1 = [m.get('epoch', 0) for m in metrics1]
    epochs2 = [m.get('epoch', 0) for m in metrics2]
    
    kl1 = [safe_get(m, 'kl') for m in metrics1]
    kl2 = [safe_get(m, 'kl') for m in metrics2]
    
    kl_loss1 = [safe_get(m, 'kl_loss') for m in metrics1]
    kl_loss2 = [safe_get(m, 'kl_loss') for m in metrics2]
    
    ppo_loss1 = [safe_get(m, 'ppo_loss') for m in metrics1]
    ppo_loss2 = [safe_get(m, 'ppo_loss') for m in metrics2]
    
    grpo_loss1 = [safe_get(m, 'grpo_loss') for m in metrics1]
    grpo_loss2 = [safe_get(m, 'grpo_loss') for m in metrics2]
    
    val_loss1 = [safe_get(m, 'val_loss') for m in metrics1]
    val_loss2 = [safe_get(m, 'val_loss') for m in metrics2]
    
    mean_ratio1 = [safe_get(m, 'mean_ratio') for m in metrics1]
    mean_ratio2 = [safe_get(m, 'mean_ratio') for m in metrics2]
    
    mean_advantage1 = [safe_get(m, 'mean_advantage') for m in metrics1]
    mean_advantage2 = [safe_get(m, 'mean_advantage') for m in metrics2]
    
    # 1. KL散度对比
    print(f"\n1. KL散度 (KL Divergence):")
    print(f"   {name1}:")
    print(f"     平均: {np.mean(kl1):.6f}")
    print(f"     初始: {kl1[0]:.6f}")
    print(f"     最终: {kl1[-1]:.6f}")
    print(f"     变化: {kl1[-1] - kl1[0]:.6f} ({((kl1[-1] - kl1[0]) / kl1[0] * 100):.2f}%)")
    print(f"   {name2}:")
    print(f"     平均: {np.mean(kl2):.6f}")
    print(f"     初始: {kl2[0]:.6f}")
    print(f"     最终: {kl2[-1]:.6f}")
    print(f"     变化: {kl2[-1] - kl2[0]:.6f} ({((kl2[-1] - kl2[0]) / kl2[0] * 100):.2f}%)")
    print(f"   差异: 平均差异={np.mean(kl1) - np.mean(kl2):.6f}, 最终差异={kl1[-1] - kl2[-1]:.6f}")
    
    # 2. KL Loss对比
    print(f"\n2. KL Loss:")
    print(f"   {name1}:")
    print(f"     平均: {np.mean(kl_loss1):.8f}")
    print(f"     初始: {kl_loss1[0]:.8f}")
    print(f"     最终: {kl_loss1[-1]:.8f}")
    print(f"   {name2}:")
    print(f"     平均: {np.mean(kl_loss2):.8f}")
    print(f"     初始: {kl_loss2[0]:.8f}")
    print(f"     最终: {kl_loss2[-1]:.8f}")
    print(f"   差异: 平均差异={np.mean(kl_loss1) - np.mean(kl_loss2):.8f}, 最终差异={kl_loss1[-1] - kl_loss2[-1]:.8f}")
    
    # 3. PPO Loss对比
    print(f"\n3. PPO Loss:")
    print(f"   {name1}:")
    print(f"     平均: {np.mean(ppo_loss1):.8f}")
    print(f"     初始: {ppo_loss1[0]:.8f}")
    print(f"     最终: {ppo_loss1[-1]:.8f}")
    print(f"   {name2}:")
    print(f"     平均: {np.mean(ppo_loss2):.8f}")
    print(f"     初始: {ppo_loss2[0]:.8f}")
    print(f"     最终: {ppo_loss2[-1]:.8f}")
    print(f"   差异: 平均差异={np.mean(ppo_loss1) - np.mean(ppo_loss2):.8f}, 最终差异={ppo_loss1[-1] - ppo_loss2[-1]:.8f}")
    
    # 4. GRPO Loss对比
    print(f"\n4. GRPO Loss:")
    print(f"   {name1}:")
    print(f"     平均: {np.mean(grpo_loss1):.8f}")
    print(f"     初始: {grpo_loss1[0]:.8f}")
    print(f"     最终: {grpo_loss1[-1]:.8f}")
    print(f"   {name2}:")
    print(f"     平均: {np.mean(grpo_loss2):.8f}")
    print(f"     初始: {grpo_loss2[0]:.8f}")
    print(f"     最终: {grpo_loss2[-1]:.8f}")
    print(f"   差异: 平均差异={np.mean(grpo_loss1) - np.mean(grpo_loss2):.8f}, 最终差异={grpo_loss1[-1] - grpo_loss2[-1]:.8f}")
    
    # 5. Validation Loss对比
    print(f"\n5. Validation Loss:")
    print(f"   {name1}:")
    print(f"     平均: {np.mean(val_loss1):.6f}")
    print(f"     初始: {val_loss1[0]:.6f}")
    print(f"     最终: {val_loss1[-1]:.6f}")
    print(f"     最小: {np.min(val_loss1):.6f} (Epoch {epochs1[np.argmin(val_loss1)]})")
    print(f"   {name2}:")
    print(f"     平均: {np.mean(val_loss2):.6f}")
    print(f"     初始: {val_loss2[0]:.6f}")
    print(f"     最终: {val_loss2[-1]:.6f}")
    print(f"     最小: {np.min(val_loss2):.6f} (Epoch {epochs2[np.argmin(val_loss2)]})")
    print(f"   差异: 平均差异={np.mean(val_loss1) - np.mean(val_loss2):.6f}, 最终差异={val_loss1[-1] - val_loss2[-1]:.6f}")
    
    # 6. Mean Ratio对比
    print(f"\n6. Mean Ratio (Policy Update Ratio):")
    print(f"   {name1}:")
    print(f"     平均: {np.mean(mean_ratio1):.6f}")
    print(f"     初始: {mean_ratio1[0]:.6f}")
    print(f"     最终: {mean_ratio1[-1]:.6f}")
    print(f"   {name2}:")
    print(f"     平均: {np.mean(mean_ratio2):.6f}")
    print(f"     初始: {mean_ratio2[0]:.6f}")
    print(f"     最终: {mean_ratio2[-1]:.6f}")
    print(f"   差异: 平均差异={np.mean(mean_ratio1) - np.mean(mean_ratio2):.6f}, 最终差异={mean_ratio1[-1] - mean_ratio2[-1]:.6f}")
    
    # 7. Mean Advantage对比
    print(f"\n7. Mean Advantage:")
    print(f"   {name1}:")
    print(f"     平均: {np.mean(mean_advantage1):.8f}")
    print(f"     初始: {mean_advantage1[0]:.8f}")
    print(f"     最终: {mean_advantage1[-1]:.8f}")
    print(f"   {name2}:")
    print(f"     平均: {np.mean(mean_advantage2):.8f}")
    print(f"     初始: {mean_advantage2[0]:.8f}")
    print(f"     最终: {mean_advantage2[-1]:.8f}")
    print(f"   差异: 平均差异={np.mean(mean_advantage1) - np.mean(mean_advantage2):.8f}, 最终差异={mean_advantage1[-1] - mean_advantage2[-1]:.8f}")
    
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)
    
    # 分析关键差异
    kl_diff = np.mean(kl1) - np.mean(kl2)
    kl_loss_diff = np.mean(kl_loss1) - np.mean(kl_loss2)
    val_loss_diff = np.mean(val_loss1) - np.mean(val_loss2)
    
    print(f"\n1. KL散度差异: {kl_diff:.6f}")
    if abs(kl_diff) > 0.01:
        print(f"   → KL_BETA=0.12 的 KL散度 {'更高' if kl_diff > 0 else '更低'}")
        print(f"   → 说明两个模型学到了不同的策略")
    else:
        print(f"   → KL散度差异很小，两个模型可能学到了相似的策略")
    
    print(f"\n2. KL Loss差异: {kl_loss_diff:.8f}")
    if abs(kl_loss_diff) > 0.0001:
        print(f"   → KL_BETA=0.12 的 KL Loss {'更高' if kl_loss_diff > 0 else '更低'}")
    else:
        print(f"   → KL Loss差异很小")
    
    print(f"\n3. Validation Loss差异: {val_loss_diff:.6f}")
    if abs(val_loss_diff) > 0.1:
        print(f"   → KL_BETA=0.12 的 Val Loss {'更高' if val_loss_diff > 0 else '更低'}")
        print(f"   → 说明两个模型的性能有差异")
    else:
        print(f"   → Validation Loss差异很小，两个模型的性能相似")
    
    # 检查收敛性
    if len(kl1) >= 10:
        kl_trend1 = np.mean(np.array(kl1[-5:]) - np.array(kl1[0:5]))
    else:
        kl_trend1 = 0
    if len(kl2) >= 10:
        kl_trend2 = np.mean(np.array(kl2[-5:]) - np.array(kl2[0:5]))
    else:
        kl_trend2 = 0
    
    print(f"\n4. 收敛性分析:")
    print(f"   {name1}: KL散度变化趋势 = {kl_trend1:.6f}")
    print(f"   {name2}: KL散度变化趋势 = {kl_trend2:.6f}")
    if abs(kl_trend1) < 0.01 and abs(kl_trend2) < 0.01:
        print(f"   → 两个模型都已收敛")
    elif abs(kl_trend1) < abs(kl_trend2):
        print(f"   → {name1} 收敛更稳定")
    else:
        print(f"   → {name2} 收敛更稳定")
    
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析训练曲线")
    parser.add_argument("--metrics1", type=str, required=True, help="第一个模型的指标 JSON 文件")
    parser.add_argument("--metrics2", type=str, required=True, help="第二个模型的指标 JSON 文件")
    parser.add_argument("--name1", type=str, default="KL_BETA=0.12", help="第一个模型名称")
    parser.add_argument("--name2", type=str, default="KL_BETA=0.15", help="第二个模型名称")
    
    args = parser.parse_args()
    
    # 加载指标
    metrics1 = load_metrics(args.metrics1)
    metrics2 = load_metrics(args.metrics2)
    
    print(f"\n加载指标:")
    print(f"  {args.name1}: {len(metrics1)} 个 epoch")
    print(f"  {args.name2}: {len(metrics2)} 个 epoch")
    
    # 打印统计信息
    print_statistics(metrics1, metrics2, args.name1, args.name2)

