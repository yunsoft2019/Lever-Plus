#!/usr/bin/env python
"""
将 V4-3 GRPO checkpoint 转换为 v2 兼容的 PyTorch Lightning 格式

V4-3 使用 GRU Decoder + MMR 多样性残差，与 V2 的结构不同
转换时需要特殊处理这些参数
"""

import argparse
import torch
import os


def convert_v4_3_to_v2_format(v4_3_ckpt_path: str, output_path: str):
    """
    将 V4-3 checkpoint 转换为 v2 兼容格式
    
    V4-3 特有参数:
    - decoder_gru.weight_ih, decoder_gru.weight_hh, decoder_gru.bias_ih, decoder_gru.bias_hh
    - step_emb.weight (如果 use_step_emb=True)
    - div_lambda (MMR 多样性权重)
    
    V2 特有参数:
    - query_update_gate.weight, query_update_gate.bias
    """
    print(f"加载 V4-3 checkpoint: {v4_3_ckpt_path}")
    v4_3_ckpt = torch.load(v4_3_ckpt_path, map_location='cpu', weights_only=False)
    
    # 获取 state_dict
    if 'model_state_dict' in v4_3_ckpt:
        v4_3_state = v4_3_ckpt['model_state_dict']
    else:
        v4_3_state = v4_3_ckpt
    
    print(f"V4-3 参数数量: {len(v4_3_state)}")
    print("V4-3 参数名:")
    for k in v4_3_state.keys():
        print(f"  {k}")
    
    # 转换为 v2 格式
    v2_state = {}
    
    for k, v in v4_3_state.items():
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
    
    # 检查 V4-3 特有参数
    has_gru = any('decoder_gru' in k for k in v2_state.keys())
    has_step_emb = any('step_emb' in k for k in v2_state.keys())
    has_div_lambda = any('div_lambda' in k for k in v2_state.keys())
    
    print(f"\n✓ 检测到 V4-3 参数:")
    print(f"  - decoder_gru: {has_gru}")
    print(f"  - step_emb: {has_step_emb}")
    print(f"  - div_lambda (MMR): {has_div_lambda}")
    
    # 打印 div_lambda 值
    if has_div_lambda:
        for k, v in v2_state.items():
            if 'div_lambda' in k:
                print(f"  - div_lambda 值: {v.tolist()}")
    
    print(f"\n转换后参数数量: {len(v2_state)}")
    print("转换后参数名:")
    for k in v2_state.keys():
        print(f"  {k}")
    
    # 创建 v2 兼容的 checkpoint
    v2_ckpt = {
        'epoch': v4_3_ckpt.get('epoch', 0),
        'global_step': 0,
        'pytorch-lightning_version': '2.0.0',
        'state_dict': v2_state,
        'hyper_parameters': {'lr': 1e-4, 'weight_decay': 0.001, 'warm_steps': 0.05},
        'model_type': 'v4_3',  # 标记模型类型
        'div_lambda_values': v4_3_ckpt.get('div_lambda_values', None)
    }
    
    # 保存
    print(f"\n保存转换后的 checkpoint: {output_path}")
    torch.save(v2_ckpt, output_path)
    print("✓ 转换完成！")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="将 V4-3 checkpoint 转换为 v2 格式")
    parser.add_argument("--v4_3_ckpt", type=str, required=True, help="V4-3 checkpoint 路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    args = parser.parse_args()
    
    if args.output is None:
        base, ext = os.path.splitext(args.v4_3_ckpt)
        args.output = f"{base}_v2format.ckpt"
    
    convert_v4_3_to_v2_format(args.v4_3_ckpt, args.output)


if __name__ == "__main__":
    main()
