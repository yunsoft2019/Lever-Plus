#!/usr/bin/env python
"""
将 v3 GRPO checkpoint 转换为 v2 兼容的 PyTorch Lightning 格式

这样可以使用 inference.sh 脚本进行推理，确保与 v2 使用相同的推理流程
"""

import argparse
import torch
import os


def convert_v3_to_v2_format(v3_ckpt_path: str, output_path: str):
    """
    将 v3 checkpoint 转换为 v2 兼容格式
    
    v3 格式: {'model_state_dict': {...}, 'epoch': ..., ...}
    v2 格式: {'state_dict': {'lever_lm.pointer_selector.xxx': ...}, ...}
    """
    print(f"加载 v3 checkpoint: {v3_ckpt_path}")
    v3_ckpt = torch.load(v3_ckpt_path, map_location='cpu', weights_only=False)
    
    # 获取 v3 的 state_dict
    if 'model_state_dict' in v3_ckpt:
        v3_state = v3_ckpt['model_state_dict']
    else:
        v3_state = v3_ckpt
    
    print(f"v3 参数数量: {len(v3_state)}")
    print("v3 参数名:")
    for k in v3_state.keys():
        print(f"  {k}")
    
    # 转换为 v2 格式
    # v3: input_proj.weight -> v2: lever_lm.pointer_selector.input_proj.weight
    # v3: cross_attn_layers.0.xxx -> v2: lever_lm.pointer_selector.cross_attn.xxx (只取第一层)
    # v3: attn_norms.0.xxx -> v2: lever_lm.pointer_selector.attn_norm.xxx (只取第一层)
    v2_state = {}
    
    for k, v in v3_state.items():
        # 跳过第2、3层的参数（v2只有1层）
        if 'cross_attn_layers.1' in k or 'cross_attn_layers.2' in k:
            continue
        if 'attn_norms.1' in k or 'attn_norms.2' in k:
            continue
        
        # 转换参数名
        new_key = k
        # cross_attn_layers.0.xxx -> cross_attn.xxx
        new_key = new_key.replace('cross_attn_layers.0.', 'cross_attn.')
        # attn_norms.0.xxx -> attn_norm.xxx
        new_key = new_key.replace('attn_norms.0.', 'attn_norm.')
        
        # 添加前缀
        new_key = f'lever_lm.pointer_selector.{new_key}'
        
        v2_state[new_key] = v
    
    # 【V4-1】检查是否包含 query_update_gate 参数
    has_v4_1 = any('query_update_gate' in k for k in v2_state.keys())
    if has_v4_1:
        print("\n✓ 检测到 V4-1 参数 (query_update_gate)")
    
    print(f"\n转换后 v2 参数数量: {len(v2_state)}")
    print("v2 参数名:")
    for k in v2_state.keys():
        print(f"  {k}")
    
    # 加载原始 v2 checkpoint 获取其他必要的参数（如 CLIP 模型权重）
    # 这里我们只保存 pointer_selector 的权重，CLIP 权重会在推理时重新加载
    
    # 创建 v2 兼容的 checkpoint
    v2_ckpt = {
        'epoch': v3_ckpt.get('epoch', 0),
        'global_step': 0,
        'pytorch-lightning_version': '2.0.0',
        'state_dict': v2_state,
        'hyper_parameters': {'lr': 1e-4, 'weight_decay': 0.001, 'warm_steps': 0.05},
    }
    
    # 保存
    print(f"\n保存转换后的 checkpoint: {output_path}")
    torch.save(v2_ckpt, output_path)
    print("✓ 转换完成！")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="将 v3 checkpoint 转换为 v2 格式")
    parser.add_argument("--v3_ckpt", type=str, required=True, help="v3 checkpoint 路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径（默认在同目录下添加 _v2format 后缀）")
    args = parser.parse_args()
    
    if args.output is None:
        base, ext = os.path.splitext(args.v3_ckpt)
        args.output = f"{base}_v2format.ckpt"
    
    convert_v3_to_v2_format(args.v3_ckpt, args.output)


if __name__ == "__main__":
    main()
