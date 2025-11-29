#!/usr/bin/env python3
"""
从完整的 checkpoint 文件中提取 LoRA 权重

用法:
    python extract_lora_from_checkpoint.py <checkpoint_path> <output_dir>
    
示例:
    python extract_lora_from_checkpoint.py \
        results/okvqa/model_cpk/v2_lora/Qwen2_5_VL_3B_Instruct_TextSimSampler_...ckpt \
        results/okvqa/model_cpk/v2_lora/lora/Qwen2_5_VL_3B_Instruct_TextSimSampler
"""

import sys
import os
import torch
from pathlib import Path

def extract_lora_from_checkpoint(checkpoint_path: str, output_dir: str):
    """
    从完整的 checkpoint 文件中提取 LoRA 权重
    
    Args:
        checkpoint_path: 完整 checkpoint 文件路径
        output_dir: LoRA 权重输出目录
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 获取模型状态字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 提取 LoRA 权重
    # LoRA 权重通常以 'lever_lm.sen_model.base_model.model.*.lora_' 或类似的前缀开头
    text_lora_state_dict = {}
    vision_lora_state_dict = {}
    
    for key, value in state_dict.items():
        # 文本编码器的 LoRA 权重
        if 'lever_lm.sen_model.base_model.model' in key and ('lora_A' in key or 'lora_B' in key):
            # 移除 'lever_lm.sen_model.base_model.model.' 前缀
            new_key = key.replace('lever_lm.sen_model.base_model.model.', '')
            text_lora_state_dict[new_key] = value
        
        # 视觉编码器的 LoRA 权重
        elif 'lever_lm.img_model.base_model.model' in key and ('lora_A' in key or 'lora_B' in key):
            # 移除 'lever_lm.img_model.base_model.model.' 前缀
            new_key = key.replace('lever_lm.img_model.base_model.model.', '')
            vision_lora_state_dict[new_key] = value
    
    # 保存文本编码器的 LoRA 权重
    if text_lora_state_dict:
        text_output_dir = os.path.join(output_dir, 'text_encoder_lora')
        os.makedirs(text_output_dir, exist_ok=True)
        
        # 保存权重
        from safetensors.torch import save_file
        save_file(text_lora_state_dict, os.path.join(text_output_dir, 'adapter_model.safetensors'))
        
        # 保存配置（需要从 checkpoint 中提取或使用默认配置）
        import json
        adapter_config = {
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "base_model_name_or_path": "openai/clip-vit-base-patch32",
            "task_type": "FEATURE_EXTRACTION"
        }
        with open(os.path.join(text_output_dir, 'adapter_config.json'), 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        print(f"✓ Extracted text encoder LoRA to: {text_output_dir}")
        print(f"  - Found {len(text_lora_state_dict)} LoRA parameters")
    else:
        print("⚠ No text encoder LoRA weights found in checkpoint")
    
    # 保存视觉编码器的 LoRA 权重
    if vision_lora_state_dict:
        vision_output_dir = os.path.join(output_dir, 'vision_encoder_lora')
        os.makedirs(vision_output_dir, exist_ok=True)
        
        # 保存权重
        from safetensors.torch import save_file
        save_file(vision_lora_state_dict, os.path.join(vision_output_dir, 'adapter_model.safetensors'))
        
        # 保存配置
        import json
        adapter_config = {
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "base_model_name_or_path": "openai/clip-vit-base-patch32",
            "task_type": "FEATURE_EXTRACTION"
        }
        with open(os.path.join(vision_output_dir, 'adapter_config.json'), 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        print(f"✓ Extracted vision encoder LoRA to: {vision_output_dir}")
        print(f"  - Found {len(vision_lora_state_dict)} LoRA parameters")
    else:
        print("⚠ No vision encoder LoRA weights found in checkpoint")
    
    if not text_lora_state_dict and not vision_lora_state_dict:
        print("\n⚠ Warning: No LoRA weights found in checkpoint!")
        print("  This might mean:")
        print("  1. The checkpoint was not trained with LoRA")
        print("  2. The LoRA weights are stored in a different format")
        print("\n  Checking checkpoint keys...")
        print("  First 20 keys in checkpoint:")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"    {key}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    extract_lora_from_checkpoint(checkpoint_path, output_dir)
    print(f"\n✓ LoRA extraction complete! Output directory: {output_dir}")

