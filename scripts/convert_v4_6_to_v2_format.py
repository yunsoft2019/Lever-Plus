#!/usr/bin/env python
"""
将 V4-6 GRPO checkpoint 转换为 v2 兼容的 PyTorch Lightning 格式

V4-6 使用 GRU Decoder + Coverage / Topic 原型覆盖，与 V2 的结构不同
转换时需要特殊处理这些参数
"""

import argparse
import torch
import os


def convert_v4_6_to_v2_format(v4_6_ckpt_path: str, output_path: str, num_topics: int = 16):
    """
    将 V4-6 checkpoint 转换为 v2 兼容格式
    
    V4-6 特有参数:
    - decoder_gru.weight_ih, decoder_gru.weight_hh, decoder_gru.bias_ih, decoder_gru.bias_hh
    - step_emb.weight (如果 use_step_emb=True)
    - topic_prototypes (Coverage 原型向量)
    - query_topic_head.weight, query_topic_head.bias
    - cover_lambda (覆盖增益权重)
    
    V2 特有参数:
    - query_update_gate.weight, query_update_gate.bias
    """
    print(f"加载 V4-6 checkpoint: {v4_6_ckpt_path}")
    v4_6_ckpt = torch.load(v4_6_ckpt_path, map_location='cpu', weights_only=False)
    
    # 获取 state_dict
    if 'model_state_dict' in v4_6_ckpt:
        v4_6_state = v4_6_ckpt['model_state_dict']
    else:
        v4_6_state = v4_6_ckpt
    
    print(f"V4-6 参数数量: {len(v4_6_state)}")
    print("V4-6 参数名:")
    for k in v4_6_state.keys():
        print(f"  {k}")
    
    # 转换为 v2 格式
    v2_state = {}
    
    for k, v in v4_6_state.items():
        # 跳过第2、3层的参数（v2只有1层）
        if 'cross_attn_layers.1' in k or 'cross_attn_layers.2' in k:
            continue
        if 'attn_norms.1' in k or 'attn_norms.2' in k:
            continue
        
        # 转换参数名
        new_key = k
        # 注意：V4-6 模型使用 cross_attn_layers 和 attn_norms（ModuleList）
        # 不需要转换参数名，保持原样
        # 旧的错误转换（已删除）：
        # new_key = new_key.replace('cross_attn_layers.0.', 'cross_attn.')
        # new_key = new_key.replace('attn_norms.0.', 'attn_norm.')
        
        # 添加前缀
        new_key = f'lever_lm.pointer_selector.{new_key}'
        
        v2_state[new_key] = v
    
    # 检查 V4-6 特有参数
    has_gru = any('decoder_gru' in k for k in v2_state.keys())
    has_step_emb = any('step_emb' in k for k in v2_state.keys())
    has_topic_prototypes = any('topic_prototypes' in k for k in v2_state.keys())
    has_query_topic_head = any('query_topic_head' in k for k in v2_state.keys())
    has_cover_lambda = any('cover_lambda' in k for k in v2_state.keys())

    print(f"\n✓ 检测到 V4-6 参数:")
    print(f"  - decoder_gru: {has_gru}")
    print(f"  - step_emb: {has_step_emb}")
    print(f"  - topic_prototypes: {has_topic_prototypes}")
    print(f"  - query_topic_head: {has_query_topic_head}")
    print(f"  - cover_lambda: {has_cover_lambda}")
    
    # 获取实际的 num_topics
    detected_num_topics = None
    for k, v in v2_state.items():
        if 'topic_prototypes' in k:
            detected_num_topics = v.shape[0]
            break
    
    if detected_num_topics is not None and detected_num_topics != num_topics:
        print(f"⚠️  警告: 指定的 num_topics ({num_topics}) 与检测到的 ({detected_num_topics}) 不一致")
        print(f"   使用检测到的值: {detected_num_topics}")
        num_topics = detected_num_topics
    
    # 获取 cover_lambda 的值
    cover_lambda_value = None
    for k, v in v2_state.items():
        if 'cover_lambda' in k:
            cover_lambda_value = v.item()
            break
    
    if cover_lambda_value is not None:
        print(f"  - cover_lambda 值: {cover_lambda_value:.4f}")
    
    print(f"\n转换后参数数量: {len(v2_state)}")
    print("转换后参数名:")
    for k in v2_state.keys():
        print(f"  {k}")
    
    # 创建 v2 兼容的 checkpoint
    v2_ckpt = {
        'epoch': v4_6_ckpt.get('epoch', 0),
        'global_step': 0,
        'pytorch-lightning_version': '2.0.0',
        'state_dict': v2_state,
        'hyper_parameters': {'lr': 1e-4, 'weight_decay': 0.001, 'warm_steps': 0.05},
        'model_type': 'v4_6',  # 标记模型类型
        'num_topics': num_topics,
        'cover_lambda': cover_lambda_value
    }
    
    # 保存
    print(f"\n保存转换后的 checkpoint: {output_path}")
    torch.save(v2_ckpt, output_path)
    print("✓ 转换完成！")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="将 V4-6 checkpoint 转换为 v2 格式")
    parser.add_argument("--v4_6_ckpt", type=str, required=True, help="V4-6 checkpoint 路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--num_topics", type=int, default=16, help="Topic/原型数量")
    args = parser.parse_args()
    
    if args.output is None:
        base, ext = os.path.splitext(args.v4_6_ckpt)
        args.output = f"{base}_v2format.ckpt"
    
    convert_v4_6_to_v2_format(args.v4_6_ckpt, args.output, args.num_topics)


if __name__ == "__main__":
    main()
