#!/usr/bin/env python3
"""
绘制训练曲线对比图
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(json_file):
    """加载训练指标"""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_training_curves(metrics1, metrics2, name1="KL_BETA=0.12", name2="KL_BETA=0.15", output_file=None):
    """绘制训练曲线对比图"""
    
    # 提取数据
    epochs1 = [m['epoch'] for m in metrics1]
    epochs2 = [m['epoch'] for m in metrics2]
    
    # 提取指标
    kl1 = [m.get('kl', 0) for m in metrics1]
    kl2 = [m.get('kl', 0) for m in metrics2]
    
    kl_loss1 = [m.get('kl_loss', 0) for m in metrics1]
    kl_loss2 = [m.get('kl_loss', 0) for m in metrics2]
    
    ppo_loss1 = [m.get('ppo_loss', 0) for m in metrics1]
    ppo_loss2 = [m.get('ppo_loss', 0) for m in metrics2]
    
    grpo_loss1 = [m.get('grpo_loss', 0) for m in metrics1]
    grpo_loss2 = [m.get('grpo_loss', 0) for m in metrics2]
    
    val_loss1 = [m.get('val_loss', 0) for m in metrics1]
    val_loss2 = [m.get('val_loss', 0) for m in metrics2]
    
    mean_ratio1 = [m.get('mean_ratio', 0) for m in metrics1]
    mean_ratio2 = [m.get('mean_ratio', 0) for m in metrics2]
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves Comparison: KL_BETA=0.12 vs 0.15', fontsize=16, fontweight='bold')
    
    # 1. KL散度
    ax = axes[0, 0]
    ax.plot(epochs1, kl1, 'o-', label=name1, linewidth=2, markersize=4)
    ax.plot(epochs2, kl2, 's-', label=name2, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KL Loss
    ax = axes[0, 1]
    ax.plot(epochs1, kl_loss1, 'o-', label=name1, linewidth=2, markersize=4)
    ax.plot(epochs2, kl_loss2, 's-', label=name2, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Loss')
    ax.set_title('KL Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PPO Loss
    ax = axes[0, 2]
    ax.plot(epochs1, ppo_loss1, 'o-', label=name1, linewidth=2, markersize=4)
    ax.plot(epochs2, ppo_loss2, 's-', label=name2, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PPO Loss')
    ax.set_title('PPO Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. GRPO Loss
    ax = axes[1, 0]
    ax.plot(epochs1, grpo_loss1, 'o-', label=name1, linewidth=2, markersize=4)
    ax.plot(epochs2, grpo_loss2, 's-', label=name2, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('GRPO Loss')
    ax.set_title('GRPO Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Validation Loss
    ax = axes[1, 1]
    ax.plot(epochs1, val_loss1, 'o-', label=name1, linewidth=2, markersize=4)
    ax.plot(epochs2, val_loss2, 's-', label=name2, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Mean Ratio
    ax = axes[1, 2]
    ax.plot(epochs1, mean_ratio1, 'o-', label=name1, linewidth=2, markersize=4)
    ax.plot(epochs2, mean_ratio2, 's-', label=name2, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Ratio')
    ax.set_title('Mean Ratio (Policy Update Ratio)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()

def print_statistics(metrics1, metrics2, name1="KL_BETA=0.12", name2="KL_BETA=0.15"):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("训练曲线统计对比")
    print("=" * 60)
    
    # KL散度
    kl1 = [m.get('kl', 0) for m in metrics1]
    kl2 = [m.get('kl', 0) for m in metrics2]
    print(f"\nKL散度:")
    print(f"  {name1}: 平均={np.mean(kl1):.4f}, 最终={kl1[-1] if kl1 else 'N/A':.4f}")
    print(f"  {name2}: 平均={np.mean(kl2):.4f}, 最终={kl2[-1] if kl2 else 'N/A':.4f}")
    
    # KL Loss
    kl_loss1 = [m.get('kl_loss', 0) for m in metrics1]
    kl_loss2 = [m.get('kl_loss', 0) for m in metrics2]
    print(f"\nKL Loss:")
    print(f"  {name1}: 平均={np.mean(kl_loss1):.6f}, 最终={kl_loss1[-1] if kl_loss1 else 'N/A':.6f}")
    print(f"  {name2}: 平均={np.mean(kl_loss2):.6f}, 最终={kl_loss2[-1] if kl_loss2 else 'N/A':.6f}")
    
    # Validation Loss
    val_loss1 = [m.get('val_loss', 0) for m in metrics1]
    val_loss2 = [m.get('val_loss', 0) for m in metrics2]
    print(f"\nValidation Loss:")
    print(f"  {name1}: 平均={np.mean(val_loss1):.4f}, 最终={val_loss1[-1] if val_loss1 else 'N/A':.4f}, 最小={np.min(val_loss1) if val_loss1 else 'N/A':.4f}")
    print(f"  {name2}: 平均={np.mean(val_loss2):.4f}, 最终={val_loss2[-1] if val_loss2 else 'N/A':.4f}, 最小={np.min(val_loss2) if val_loss2 else 'N/A':.4f}")
    
    # Mean Ratio
    mean_ratio1 = [m.get('mean_ratio', 0) for m in metrics1]
    mean_ratio2 = [m.get('mean_ratio', 0) for m in metrics2]
    print(f"\nMean Ratio:")
    print(f"  {name1}: 平均={np.mean(mean_ratio1):.4f}, 最终={mean_ratio1[-1] if mean_ratio1 else 'N/A':.4f}")
    print(f"  {name2}: 平均={np.mean(mean_ratio2):.4f}, 最终={mean_ratio2[-1] if mean_ratio2 else 'N/A':.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="绘制训练曲线对比图")
    parser.add_argument("--metrics1", type=str, required=True, help="第一个模型的指标 JSON 文件")
    parser.add_argument("--metrics2", type=str, required=True, help="第二个模型的指标 JSON 文件")
    parser.add_argument("--name1", type=str, default="KL_BETA=0.12", help="第一个模型名称")
    parser.add_argument("--name2", type=str, default="KL_BETA=0.15", help="第二个模型名称")
    parser.add_argument("--output", type=str, default=None, help="输出图片文件路径")
    
    args = parser.parse_args()
    
    # 加载指标
    metrics1 = load_metrics(args.metrics1)
    metrics2 = load_metrics(args.metrics2)
    
    # 打印统计信息
    print_statistics(metrics1, metrics2, args.name1, args.name2)
    
    # 绘制曲线
    plot_training_curves(metrics1, metrics2, args.name1, args.name2, args.output)




