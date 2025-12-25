#!/usr/bin/env python3
"""
诊断脚本：检查为什么不同配置的结果几乎完全相同
"""

import torch
import os
import json
import sys

PROJECT_ROOT = "/mnt/share/yiyun/Projects/Lever-Plus"
DATASET_NAME = "okvqa"

def check_checkpoints():
    """检查所有 checkpoint 的差异"""
    print("=" * 80)
    print("1. 检查所有 GRPO checkpoint")
    print("=" * 80)
    
    checkpoints = {
        'kl012_frozen_epoch1': f'{PROJECT_ROOT}/results/{DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl012/grpo_epoch1.pt',
        'kl015_unfrozen_epoch3': f'{PROJECT_ROOT}/results/{DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl015_unfrozen/grpo_epoch3.pt',
        'kl010_unfrozen_lr1e5_epoch3': f'{PROJECT_ROOT}/results/{DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl010_unfrozen_lr1e5/grpo_epoch3.pt',
        'kl010_unfrozen_lr2e5_epoch1': f'{PROJECT_ROOT}/results/{DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl010_unfrozen_lr2e5/grpo_epoch1.pt',
    }
    
    ckpt_data = {}
    for name, path in checkpoints.items():
        if os.path.exists(path):
            ckpt_data[name] = torch.load(path, map_location='cpu')
            print(f"✅ {name}: 存在")
        else:
            print(f"❌ {name}: 不存在")
    
    if len(ckpt_data) < 2:
        print("⚠️  可用的 checkpoint 太少，无法对比")
        return
    
    # 对比参数
    print("\n" + "=" * 80)
    print("2. 对比 checkpoint 参数")
    print("=" * 80)
    
    base_name = list(ckpt_data.keys())[0]
    base_state = ckpt_data[base_name]['model_state_dict']
    
    print(f"\n基准: {base_name}")
    print(f"  KL_BETA: {ckpt_data[base_name].get('kl_beta', 'N/A')}")
    print(f"  Epoch: {ckpt_data[base_name].get('epoch', 'N/A')}")
    
    for name, ckpt in ckpt_data.items():
        if name == base_name:
            continue
        
        state = ckpt['model_state_dict']
        
        print(f"\n对比: {base_name} vs {name}")
        print("-" * 80)
        print(f"  KL_BETA: {ckpt_data[base_name].get('kl_beta', 'N/A')} vs {ckpt.get('kl_beta', 'N/A')}")
        print(f"  Epoch: {ckpt_data[base_name].get('epoch', 'N/A')} vs {ckpt.get('epoch', 'N/A')}")
        
        # 对比所有参数
        total_diff = 0
        num_params = 0
        max_diff = 0
        max_diff_key = None
        
        for k in base_state.keys():
            if k in state:
                base_param = base_state[k]
                param = state[k]
                diff = (base_param - param).abs().mean().item()
                total_diff += diff
                num_params += 1
                if diff > max_diff:
                    max_diff = diff
                    max_diff_key = k
        
        avg_diff = total_diff / num_params if num_params > 0 else 0
        print(f"  平均参数差异: {avg_diff:.6f}")
        print(f"  最大参数差异: {max_diff:.6f} ({max_diff_key})")
        
        if avg_diff < 1e-5:
            print(f"  ⚠️  参数差异极小，可能不足以影响结果")
        elif avg_diff < 1e-3:
            print(f"  ⚠️  参数差异较小，可能影响有限")
        else:
            print(f"  ✅ 参数差异明显")

def check_rce_checkpoints():
    """检查 RCE checkpoint"""
    print("\n" + "=" * 80)
    print("3. 检查 RCE checkpoint（训练起点）")
    print("=" * 80)
    
    rce_path = f'{PROJECT_ROOT}/results/{DATASET_NAME}/model_cpk/v3_RandSampler_Qwen2_5-VL-3B-Instruct_kl012/rce_epoch5.pt'
    
    if os.path.exists(rce_path):
        print(f"✅ RCE checkpoint 存在: {rce_path}")
        rce_data = torch.load(rce_path, map_location='cpu')
        print(f"  Epoch: {rce_data.get('epoch', 'N/A')}")
        print(f"  Phase: {rce_data.get('phase', 'N/A')}")
    else:
        print(f"❌ RCE checkpoint 不存在: {rce_path}")

def check_data_files():
    """检查数据文件"""
    print("\n" + "=" * 80)
    print("4. 检查数据文件")
    print("=" * 80)
    
    data_path = f'{PROJECT_ROOT}/results/{DATASET_NAME}/generated_data/rl_data_k64_v3.json'
    
    if os.path.exists(data_path):
        print(f"✅ 数据文件存在: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"  Query 数量: {len(data)}")
        
        # 检查是否有 _meta
        if '_meta' in data:
            print(f"  包含 _meta 信息")
            meta = data['_meta']
            print(f"    Total queries: {meta.get('total_queries', 'N/A')}")
            print(f"    Positive queries: {meta.get('positive_queries', 'N/A')}")
    else:
        print(f"❌ 数据文件不存在: {data_path}")

def check_training_configs():
    """检查训练配置"""
    print("\n" + "=" * 80)
    print("5. 检查训练配置脚本")
    print("=" * 80)
    
    scripts = [
        'scripts/train_grpo_kl012_from_rce.sh',
        'scripts/train_grpo_kl015_unfrozen_from_rce.sh',
        'scripts/train_grpo_kl010_unfrozen_lr1e5_from_rce.sh',
        'scripts/train_grpo_kl010_unfrozen_lr2e5_from_rce.sh',
    ]
    
    for script in scripts:
        script_path = f'{PROJECT_ROOT}/{script}'
        if os.path.exists(script_path):
            print(f"\n✅ {script}:")
            with open(script_path, 'r') as f:
                content = f.read()
            
            # 提取关键配置
            if 'RCE_CKPT=' in content:
                import re
                match = re.search(r'RCE_CKPT="([^"]+)"', content)
                if match:
                    print(f"  RCE_CKPT: {match.group(1)}")
            
            if 'KL_BETA=' in content:
                import re
                match = re.search(r'KL_BETA=([0-9.e-]+)', content)
                if match:
                    print(f"  KL_BETA: {match.group(1)}")
            
            if 'GRPO_LR=' in content:
                import re
                match = re.search(r'GRPO_LR=([0-9.e-]+)', content)
                if match:
                    print(f"  GRPO_LR: {match.group(1)}")
        else:
            print(f"❌ {script}: 不存在")

def main():
    check_checkpoints()
    check_rce_checkpoints()
    check_data_files()
    check_training_configs()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("\n如果所有 checkpoint 的参数差异都很小（< 0.001），")
    print("那么即使训练了不同的配置，最终结果也可能几乎相同。")
    print("\n建议：")
    print("1. 检查是否所有实验都使用了相同的 RCE checkpoint")
    print("2. 检查训练时是否真的使用了不同的 KL_BETA 和学习率")
    print("3. 检查训练日志，确认训练过程是否正常")
    print("4. 尝试更大的参数差异（如更大的学习率或更小的 KL_BETA）")

if __name__ == '__main__':
    main()

