#!/usr/bin/env python3
"""
将 V4-8 checkpoint 转换为 V2 格式

V4-8 使用 Slot/Set Decoder 架构，需要转换以下参数：
- slot_emb: slot embeddings
- slot_self_attn_layers: slot self-attention 层
- slot_norms: slot layer norms
- slot_cand_attn: slot-candidate cross-attention (可选)
- slot_cand_norm: slot-candidate layer norm (可选)

使用方法:
    python scripts/convert_v4_8_to_v2_format.py --input <v4_8_ckpt> --output <v2_ckpt>
"""

import argparse
import torch
import os
import sys

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def convert_v4_8_to_v2(input_path: str, output_path: str):
    """转换 V4-8 checkpoint 到 V2 格式"""
    
    print(f"加载 V4-8 checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')
    
    # 获取 model state dict
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    print(f"原始参数数量: {len(state_dict)}")
    
    # V4-8 特有的参数前缀
    v4_8_prefixes = [
        'slot_emb',
        'slot_self_attn_layers',
        'slot_norms',
        'slot_cand_attn',
        'slot_cand_norm',
    ]
    
    # 创建新的 state dict
    new_state_dict = {}
    skipped_keys = []
    kept_keys = []
    
    for key, value in state_dict.items():
        # 检查是否是 V4-8 特有参数
        is_v4_8_param = any(key.startswith(prefix) for prefix in v4_8_prefixes)
        
        if is_v4_8_param:
            # 保留 V4-8 特有参数（推理时需要）
            new_state_dict[key] = value
            kept_keys.append(key)
        else:
            # 保留共享参数
            new_state_dict[key] = value
            kept_keys.append(key)
    
    print(f"\n转换后参数数量: {len(new_state_dict)}")
    print(f"保留的参数: {len(kept_keys)}")
    
    # 打印 V4-8 特有参数
    v4_8_params = [k for k in kept_keys if any(k.startswith(p) for p in v4_8_prefixes)]
    if v4_8_params:
        print(f"\nV4-8 特有参数 ({len(v4_8_params)} 个):")
        for k in v4_8_params[:10]:
            print(f"  - {k}: {new_state_dict[k].shape}")
        if len(v4_8_params) > 10:
            print(f"  ... 还有 {len(v4_8_params) - 10} 个")
    
    # 保存转换后的 checkpoint
    output_ckpt = {
        'model_state_dict': new_state_dict,
        'config': ckpt.get('config', {}),
        'epoch': ckpt.get('epoch', 0),
        'val_loss': ckpt.get('val_loss', 0),
        'source': 'v4_8',
        'converted': True
    }
    
    torch.save(output_ckpt, output_path)
    print(f"\n✓ 保存转换后的 checkpoint: {output_path}")
    
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert V4-8 checkpoint to V2 format")
    parser.add_argument("--input", type=str, required=True, help="Input V4-8 checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output V2 format checkpoint path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    convert_v4_8_to_v2(args.input, args.output)


if __name__ == "__main__":
    main()
