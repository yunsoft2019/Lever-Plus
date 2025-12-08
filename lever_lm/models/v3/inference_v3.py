"""
V3模型推理适配

V3模型继承自V2，推理时使用相同的predict方法。
本模块提供：
1. 从GRPO检查点加载V3模型
2. 与现有推理流程的兼容性

作者: Lever-Plus Team
日期: 2025-12-02
"""

import os
import torch
from typing import Optional, Tuple
from loguru import logger

from lever_lm.models.v3 import PointerSelectorV3, PointerSelectorV3Config


def load_v3_from_grpo_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> PointerSelectorV3:
    """
    从GRPO检查点加载V3模型
    
    Args:
        checkpoint_path: GRPO检查点路径（.pt文件）
        device: 目标设备
    
    Returns:
        加载好的V3模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"加载GRPO检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型配置
    # 检查点可能包含不同的格式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # PyTorch Lightning格式，需要去掉前缀
        state_dict = {k.replace("lever_lm.", "").replace("model.", ""): v 
                      for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # 从state_dict推断模型配置
    # 查找hidden_dim
    hidden_dim = 256  # 默认值
    for key in state_dict.keys():
        if "query_proj.weight" in key:
            hidden_dim = state_dict[key].shape[0]
            break
    
    # 查找d_model
    d_model = 768  # 默认值
    for key in state_dict.keys():
        if "input_proj.weight" in key:
            d_model = state_dict[key].shape[1]
            break
    
    # 查找num_layers
    num_layers = 0
    for key in state_dict.keys():
        if "cross_attn_layers" in key:
            layer_idx = int(key.split(".")[1])
            num_layers = max(num_layers, layer_idx + 1)
    if num_layers == 0:
        num_layers = 3  # 默认值
    
    logger.info(f"推断模型配置: d_model={d_model}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    # 创建模型
    model = PointerSelectorV3(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # 恢复kl_beta（如果有）
    if "kl_beta" in checkpoint:
        model.kl_beta = checkpoint["kl_beta"]
        logger.info(f"恢复kl_beta: {model.kl_beta}")
    
    logger.info(f"✓ V3模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_v3_from_sft_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> PointerSelectorV3:
    """
    从SFT检查点加载V3模型（用于继续GRPO训练或推理）
    
    SFT检查点通常是PyTorch Lightning格式
    
    Args:
        checkpoint_path: SFT检查点路径（.ckpt文件）
        device: 目标设备
    
    Returns:
        加载好的V3模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"加载SFT检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # PyTorch Lightning格式
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # 去掉前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉 "lever_lm." 和 "pointer_selector." 前缀
        new_key = k
        if new_key.startswith("lever_lm."):
            new_key = new_key[len("lever_lm."):]
        if new_key.startswith("pointer_selector."):
            new_key = new_key[len("pointer_selector."):]
        new_state_dict[new_key] = v
    
    # 从state_dict推断模型配置
    hidden_dim = 256
    d_model = 768
    num_layers = 3
    
    for key in new_state_dict.keys():
        if "query_proj.weight" in key:
            hidden_dim = new_state_dict[key].shape[0]
        if "input_proj.weight" in key:
            d_model = new_state_dict[key].shape[1]
        if "cross_attn_layers" in key:
            layer_idx = int(key.split(".")[1])
            num_layers = max(num_layers, layer_idx + 1)
    
    logger.info(f"推断模型配置: d_model={d_model}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    # 创建V3模型
    model = PointerSelectorV3(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    # 加载权重（V3继承V2，权重兼容）
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ V3模型从SFT检查点加载完成")
    
    return model


@torch.inference_mode()
def predict_with_v3(
    model: PointerSelectorV3,
    query_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    top_k: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用V3模型进行预测
    
    V3继承V2的predict方法，接口完全兼容
    
    Args:
        model: V3模型
        query_emb: [B, d] query embedding
        cand_emb: [B, K, d] 候选embedding
        top_k: 返回top-k个预测
    
    Returns:
        predictions: [B, shot_num] 预测的候选索引
        scores: [B, shot_num] 预测的分数
    """
    model.eval()
    return model.predict(query_emb, cand_emb, top_k=top_k)


if __name__ == "__main__":
    """测试推理适配"""
    print("="*70)
    print("测试V3推理适配")
    print("="*70)
    
    # 创建模拟模型
    model = PointerSelectorV3(d_model=768, K=32, shot_num=2)
    print(f"✓ 模型创建成功")
    
    # 测试预测
    batch_size = 4
    query_emb = torch.randn(batch_size, 768)
    cand_emb = torch.randn(batch_size, 32, 768)
    
    predictions, scores = predict_with_v3(model, query_emb, cand_emb)
    print(f"✓ 预测成功")
    print(f"  predictions shape: {predictions.shape}")
    print(f"  scores shape: {scores.shape}")
    print(f"  predictions[0]: {predictions[0].tolist()}")
    print(f"  scores[0]: {scores[0].tolist()}")
    
    # 测试保存和加载
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name
    
    # 保存GRPO格式检查点
    torch.save({
        "model_state_dict": model.state_dict(),
        "kl_beta": model.kl_beta,
        "epoch": 3,
        "phase": "grpo"
    }, temp_path)
    print(f"✓ 检查点保存成功: {temp_path}")
    
    # 加载检查点
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = load_v3_from_grpo_checkpoint(temp_path, device=device)
    print(f"✓ 检查点加载成功")
    
    # 验证预测一致性（需要将输入移到同一设备）
    query_emb_gpu = query_emb.to(device)
    cand_emb_gpu = cand_emb.to(device)
    predictions2, scores2 = predict_with_v3(loaded_model, query_emb_gpu, cand_emb_gpu)
    assert torch.allclose(predictions.to(device), predictions2), "预测不一致！"
    print(f"✓ 预测一致性验证通过")
    
    # 清理
    os.remove(temp_path)
    
    print("\n" + "="*70)
    print("✓ V3推理适配测试通过！")
    print("="*70)
