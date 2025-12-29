#!/usr/bin/env python
"""
将 V4-7 checkpoint 转换为 V2 格式，用于推理

注意：保留 decoder_gru 和 step_emb 参数，因为推理时需要检测这些参数来正确创建模型。
只跳过 DPP 特有的参数（dpp_proj, dpp_lambda），因为推理时使用 V2 的打分方式。

使用方法:
    python scripts/convert_v4_7_to_v2_format.py \
        --v4_7_ckpt results/okvqa/model_cpk/v3_plan_v4_7_rank32/grpo_epoch1.pt \
        --output results/okvqa/model_cpk/v3_plan_v4_7_rank32/grpo_epoch1_v2format.ckpt \
        --dpp_rank 32
"""

import argparse
import torch


def convert_v4_7_to_v2(v4_7_ckpt_path: str, output_path: str, dpp_rank: int = 32):
    """将 V4-7 checkpoint 转换为 V2 格式"""
    print(f"加载 V4-7 checkpoint: {v4_7_ckpt_path}")
    ckpt = torch.load(v4_7_ckpt_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    # 只跳过 DPP 特有的参数（推理时不使用 DPP 打分）
    # 保留 decoder_gru 和 step_emb，因为推理时需要这些参数
    v4_7_dpp_only_keys = [
        "dpp_proj",
        "dpp_lambda",
    ]
    
    # 过滤掉 DPP 特有参数，保留其他所有参数
    v2_state_dict = {}
    skipped_keys = []
    
    for k, v in state_dict.items():
        skip = False
        for specific_key in v4_7_dpp_only_keys:
            if specific_key in k:
                skip = True
                skipped_keys.append(k)
                break
        
        if not skip:
            # 保持原始 key 名称，不做转换
            # V4-7 使用 cross_attn_layers.0. 格式，推理时也使用相同格式
            v2_state_dict[k] = v
    
    print(f"\n跳过的 V4-7 特有参数 ({len(skipped_keys)} 个):")
    for k in skipped_keys:
        print(f"  - {k}")
    
    print(f"\n保留的参数 ({len(v2_state_dict)} 个):")
    for k in list(v2_state_dict.keys())[:10]:
        print(f"  - {k}")
    if len(v2_state_dict) > 10:
        print(f"  ... 还有 {len(v2_state_dict) - 10} 个")
    
    # 保存为 V2 格式
    v2_ckpt = {
        "state_dict": {f"lever_lm.pointer_selector.{k}": v for k, v in v2_state_dict.items()},
        "model_type": "v4_7_converted_to_v2",
        "original_dpp_rank": dpp_rank
    }
    
    # 保留原始 checkpoint 的元信息
    if "metrics" in ckpt:
        v2_ckpt["metrics"] = ckpt["metrics"]
    if "epoch" in ckpt:
        v2_ckpt["epoch"] = ckpt["epoch"]
    if "phase" in ckpt:
        v2_ckpt["phase"] = ckpt["phase"]
    
    torch.save(v2_ckpt, output_path)
    print(f"\n✓ V2 格式 checkpoint 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert V4-7 checkpoint to V2 format")
    parser.add_argument("--v4_7_ckpt", type=str, required=True, help="V4-7 checkpoint 路径")
    parser.add_argument("--output", type=str, required=True, help="输出路径")
    parser.add_argument("--dpp_rank", type=int, default=32, help="DPP 低秩投影维度")
    args = parser.parse_args()
    
    convert_v4_7_to_v2(args.v4_7_ckpt, args.output, args.dpp_rank)


if __name__ == "__main__":
    main()
