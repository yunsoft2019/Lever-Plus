#!/usr/bin/env python
"""
将 V4-5 GRPO checkpoint 转换为 v2 兼容的 PyTorch Lightning 格式

V4-5 使用 GRU Decoder + Additive/Bilinear Attention 打分头，与 V2 的结构不同
转换时需要特殊处理这些参数
"""

import argparse
import torch
import os


def convert_v4_5_to_v2_format(v4_5_ckpt_path: str, output_path: str, attention_type: str = 'additive'):
    """
    将 V4-5 checkpoint 转换为 v2 兼容格式
    
    V4-5 特有参数:
    - decoder_gru.weight_ih, decoder_gru.weight_hh, decoder_gru.bias_ih, decoder_gru.bias_hh
    - step_emb.weight (如果 use_step_emb=True)
    - Additive Attention: attn_Wq.weight, attn_Wq.bias, attn_Wc.weight, attn_v.weight
    - Bilinear Attention: bilinear.weight
    
    V2 特有参数:
    - query_update_gate.weight, query_update_gate.bias
    """
    print(f"加载 V4-5 checkpoint: {v4_5_ckpt_path}")
    v4_5_ckpt = torch.load(v4_5_ckpt_path, map_location='cpu', weights_only=False)
    
    # 获取 state_dict
    if 'model_state_dict' in v4_5_ckpt:
        v4_5_state = v4_5_ckpt['model_state_dict']
    else:
        v4_5_state = v4_5_ckpt
    
    print(f"V4-5 参数数量: {len(v4_5_state)}")
    print("V4-5 参数名:")
    for k in v4_5_state.keys():
        print(f"  {k}")
    
    # 转换为 v2 格式
    v2_state = {}
    
    for k, v in v4_5_state.items():
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
    
    # 检查 V4-5 特有参数
    has_gru = any('decoder_gru' in k for k in v2_state.keys())
    has_step_emb = any('step_emb' in k for k in v2_state.keys())
    has_additive = any('attn_Wq' in k or 'attn_Wc' in k or 'attn_v' in k for k in v2_state.keys())
    has_bilinear = any('bilinear' in k for k in v2_state.keys())
    
    print(f"\n✓ 检测到 V4-5 参数:")
    print(f"  - decoder_gru: {has_gru}")
    print(f"  - step_emb: {has_step_emb}")
    print(f"  - Additive Attention: {has_additive}")
    print(f"  - Bilinear Attention: {has_bilinear}")
    
    # 确定实际的 attention_type
    if has_additive:
        detected_type = 'additive'
    elif has_bilinear:
        detected_type = 'bilinear'
    else:
        detected_type = 'dot'
    
    if detected_type != attention_type:
        print(f"⚠️  警告: 指定的 attention_type ({attention_type}) 与检测到的 ({detected_type}) 不一致")
        print(f"   使用检测到的类型: {detected_type}")
        attention_type = detected_type
    
    print(f"\n转换后参数数量: {len(v2_state)}")
    print("转换后参数名:")
    for k in v2_state.keys():
        print(f"  {k}")
    
    # 创建 v2 兼容的 checkpoint
    v2_ckpt = {
        'epoch': v4_5_ckpt.get('epoch', 0),
        'global_step': 0,
        'pytorch-lightning_version': '2.0.0',
        'state_dict': v2_state,
        'hyper_parameters': {'lr': 1e-4, 'weight_decay': 0.001, 'warm_steps': 0.05},
        'model_type': 'v4_5',  # 标记模型类型
        'attention_type': attention_type
    }
    
    # 保存
    print(f"\n保存转换后的 checkpoint: {output_path}")
    torch.save(v2_ckpt, output_path)
    print("✓ 转换完成！")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="将 V4-5 checkpoint 转换为 v2 格式")
    parser.add_argument("--v4_5_ckpt", type=str, required=True, help="V4-5 checkpoint 路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--attention_type", type=str, default="additive", 
                        choices=["dot", "additive", "bilinear"], help="打分头类型")
    args = parser.parse_args()
    
    if args.output is None:
        base, ext = os.path.splitext(args.v4_5_ckpt)
        args.output = f"{base}_v2format.ckpt"
    
    convert_v4_5_to_v2_format(args.v4_5_ckpt, args.output, args.attention_type)


if __name__ == "__main__":
    main()
