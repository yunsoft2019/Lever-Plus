#!/usr/bin/env python3
"""
将 V4-4 checkpoint 转换为 V2 格式，用于推理

V4-4 特有参数：
- cand_encoder: Candidate Self-Attention
- decoder_gru: GRU decoder
- step_emb: Step embedding
- div_lambda: MMR 多样性权重
"""

import argparse
import torch


def convert_v4_4_to_v2_format(v4_4_ckpt_path: str, output_path: str):
    """转换 V4-4 checkpoint 为 V2 格式"""
    
    print(f"加载 V4-4 checkpoint: {v4_4_ckpt_path}")
    ckpt = torch.load(v4_4_ckpt_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    print(f"V4-4 参数数量: {len(state_dict)}")
    print("V4-4 参数名:")
    for k in state_dict.keys():
        print(f"  {k}")
    
    # 检测 V4-4 特有参数
    has_cand_encoder = any("cand_encoder" in k for k in state_dict.keys())
    has_decoder_gru = any("decoder_gru" in k for k in state_dict.keys())
    has_step_emb = any("step_emb" in k for k in state_dict.keys())
    has_div_lambda = "div_lambda" in state_dict
    
    print(f"\n✓ 检测到 V4-4 参数:")
    print(f"  - cand_encoder: {has_cand_encoder}")
    print(f"  - decoder_gru: {has_decoder_gru}")
    print(f"  - step_emb: {has_step_emb}")
    print(f"  - div_lambda (MMR): {has_div_lambda}")
    
    if has_div_lambda:
        print(f"  - div_lambda 值: {state_dict['div_lambda'].tolist()}")
    
    # 转换为 V2 格式（添加 lever_lm.pointer_selector. 前缀）
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = f"lever_lm.pointer_selector.{k}"
        new_state_dict[new_key] = v
    
    print(f"\n转换后参数数量: {len(new_state_dict)}")
    print("转换后参数名:")
    for k in new_state_dict.keys():
        print(f"  {k}")
    
    # 保存
    torch.save(new_state_dict, output_path)
    print(f"\n保存转换后的 checkpoint: {output_path}")
    print("✓ 转换完成！")


def main():
    parser = argparse.ArgumentParser(description="Convert V4-4 checkpoint to V2 format")
    parser.add_argument("--v4_4_ckpt", type=str, required=True, help="V4-4 checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()
    
    convert_v4_4_to_v2_format(args.v4_4_ckpt, args.output)


if __name__ == "__main__":
    main()
