#!/usr/bin/env python
"""
将 V4-10 GRPO checkpoint 转换为 v2 格式，用于推理

V4-10 特点：
- 添加可学习的 STOP token
- 让模型自己决定何时停止选择
- 输出 logits 维度为 K+1（包含 STOP）

使用方法:
    python scripts/convert_v4_10_to_v2_format.py --input <input.pt> --output <output.ckpt>
"""

import argparse
import torch
import os
import sys

# 添加项目根目录到 Python 路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def convert_v4_10_to_v2_format(input_path: str, output_path: str):
    """
    将 V4-10 checkpoint 转换为 v2 格式
    
    Args:
        input_path: 输入的 V4-10 checkpoint 路径 (.pt)
        output_path: 输出的 v2 格式 checkpoint 路径 (.ckpt)
    """
    print(f"加载 V4-10 checkpoint: {input_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # 获取 state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # 获取元数据
    model_type = checkpoint.get("model_type", "v4_10")
    use_gru = checkpoint.get("use_gru", True)
    use_step_emb = checkpoint.get("use_step_emb", True)
    kl_beta = checkpoint.get("kl_beta", 0.1)
    epoch = checkpoint.get("epoch", 0)
    phase = checkpoint.get("phase", "unknown")
    metrics = checkpoint.get("metrics", {})
    
    print(f"  - Model Type: {model_type}")
    print(f"  - Use GRU: {use_gru}")
    print(f"  - Use Step Emb: {use_step_emb}")
    print(f"  - Phase: {phase}")
    print(f"  - Epoch: {epoch}")
    
    # 检测模型配置
    hidden_dim = 256
    d_model = 512
    
    for key in state_dict.keys():
        if "query_proj.weight" in key:
            hidden_dim = state_dict[key].shape[0]
        if "input_proj.weight" in key:
            d_model = state_dict[key].shape[1]
    
    print(f"  - d_model: {d_model}")
    print(f"  - hidden_dim: {hidden_dim}")
    
    # 检测 STOP token
    has_stop_token = any("stop_token" in k for k in state_dict.keys())
    print(f"  - Has STOP token: {has_stop_token}")
    
    # 检测 step_emb 大小
    step_emb_size = 2
    if "step_emb.weight" in state_dict:
        step_emb_size = state_dict["step_emb.weight"].shape[0]
    print(f"  - Step Emb Size: {step_emb_size}")
    
    # 创建新的 checkpoint（v2 格式）
    new_checkpoint = {
        "state_dict": {},
        "model_state_dict": state_dict,  # 保留原始 state_dict
        "model_type": "v4_10",
        "use_gru": use_gru,
        "use_step_emb": use_step_emb,
        "kl_beta": kl_beta,
        "epoch": epoch,
        "phase": phase,
        "metrics": metrics,
        "d_model": d_model,
        "hidden_dim": hidden_dim,
        "has_stop_token": has_stop_token,
    }
    
    # 添加前缀以兼容 v2 加载方式
    for key, value in state_dict.items():
        new_key = f"lever_lm.pointer_selector.{key}"
        new_checkpoint["state_dict"][new_key] = value
    
    # 保存
    print(f"\n保存转换后的 checkpoint: {output_path}")
    torch.save(new_checkpoint, output_path)
    
    # 验证
    print("\n验证转换结果...")
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    print(f"  - state_dict keys: {len(loaded['state_dict'])}")
    print(f"  - model_type: {loaded.get('model_type', 'unknown')}")
    print(f"  - has_stop_token: {loaded.get('has_stop_token', False)}")
    
    print("\n✓ 转换完成！")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert V4-10 checkpoint to v2 format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input V4-10 checkpoint path (.pt)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output v2 format checkpoint path (.ckpt)")
    args = parser.parse_args()
    
    # 默认输出路径
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_v2format.ckpt"
    
    convert_v4_10_to_v2_format(args.input, args.output)


if __name__ == "__main__":
    main()
