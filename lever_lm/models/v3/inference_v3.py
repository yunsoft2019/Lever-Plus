"""
V3模型推理适配

V3模型继承自V2，推理时使用相同的predict方法。
本模块提供：
1. 从GRPO检查点加载V3模型
2. 从GRPO检查点加载V4-2模型（GRU Pointer Decoder）
3. 从GRPO检查点加载V4-3模型（GRU + MMR 多样性残差）
4. 从GRPO检查点加载V4-4模型（Candidate Set Encoder + GRU + MMR）
5. 从GRPO检查点加载V4-5模型（GRU + Additive/Bilinear Attention）
6. 从GRPO检查点加载V4-6模型（GRU + Coverage / Topic 原型覆盖）
7. 与现有推理流程的兼容性

作者: Lever-Plus Team
日期: 2025-12-02
更新: 2025-12-27 - 添加V4-6支持
"""

import os
import torch
from typing import Optional, Tuple, Union
from loguru import logger

from lever_lm.models.v3 import PointerSelectorV3, PointerSelectorV3Config
from lever_lm.models.v3 import PointerSelectorV4_2_RL
from lever_lm.models.v3.pointer_selector_v4_3_rl import PointerSelectorV4_3_RL
from lever_lm.models.v3.pointer_selector_v4_4_rl import PointerSelectorV4_4_RL
from lever_lm.models.v3.pointer_selector_v4_5_rl import PointerSelectorV4_5_RL
from lever_lm.models.v3.pointer_selector_v4_6_rl import PointerSelectorV4_6_RL


def load_v3_from_grpo_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    model_type: Optional[str] = None
) -> Union[PointerSelectorV3, PointerSelectorV4_2_RL, PointerSelectorV4_3_RL, PointerSelectorV4_4_RL, PointerSelectorV4_5_RL, PointerSelectorV4_6_RL]:
    """
    从GRPO检查点加载V3、V4-2、V4-3、V4-4或V4-5模型
    
    Args:
        checkpoint_path: GRPO检查点路径（.pt文件）
        device: 目标设备
        model_type: 模型类型（"v3", "v4_2", "v4_3", "v4_4", "v4_5"），如果为None则自动检测
    
    Returns:
        加载好的V3、V4-2、V4-3、V4-4或V4-5模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查环境变量中的模型类型
    if model_type is None:
        model_type = os.environ.get("LEVER_LM_MODEL_TYPE", None)
    
    logger.info(f"加载GRPO检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型配置
    # 检查点可能包含不同的格式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # PyTorch Lightning格式，需要去掉前缀
        state_dict = {k.replace("lever_lm.", "").replace("model.", "").replace("pointer_selector.", ""): v 
                      for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # 自动检测模型类型
    if model_type is None:
        # 检查checkpoint中是否有model_type标记
        model_type = checkpoint.get("model_type", None)
        
        # 如果没有标记，通过参数名检测
        if model_type is None:
            has_cand_encoder = any("cand_encoder" in k for k in state_dict.keys())
            has_gru = any("decoder_gru" in k for k in state_dict.keys())
            has_step_emb = any("step_emb" in k for k in state_dict.keys())
            has_div_lambda = any("div_lambda" in k for k in state_dict.keys())
            has_query_update_gate = any("query_update_gate" in k for k in state_dict.keys())
            # V4-5 特有参数
            has_additive_attn = any("attn_Wq" in k or "attn_Wc" in k or "attn_v" in k for k in state_dict.keys())
            has_bilinear_attn = any("bilinear" in k for k in state_dict.keys())
            # V4-6 特有参数
            has_topic_prototypes = any("topic_prototypes" in k for k in state_dict.keys())
            has_cover_lambda = any("cover_lambda" in k for k in state_dict.keys())
            
            if has_topic_prototypes or has_cover_lambda:
                model_type = "v4_6"
                logger.info("检测到 V4-6 模型参数 (topic_prototypes/cover_lambda)")
            elif has_cand_encoder:
                model_type = "v4_4"
                logger.info("检测到 V4-4 模型参数 (cand_encoder)")
            elif has_additive_attn or has_bilinear_attn:
                model_type = "v4_5"
                logger.info("检测到 V4-5 模型参数 (Additive/Bilinear Attention)")
            elif has_div_lambda:
                model_type = "v4_3"
                logger.info("检测到 V4-3 模型参数 (div_lambda/MMR)")
            elif has_gru or has_step_emb:
                model_type = "v4_2"
                logger.info("检测到 V4-2 模型参数 (decoder_gru/step_emb)")
            elif has_query_update_gate:
                model_type = "v3"  # V3 继承 V2，V2 有 query_update_gate
                logger.info("检测到 V3/V2 模型参数 (query_update_gate)")
            else:
                model_type = "v3"  # 默认
                logger.info("使用默认模型类型: V3")
    
    # 从state_dict推断模型配置
    hidden_dim = 256  # 默认值
    for key in state_dict.keys():
        if "query_proj.weight" in key:
            hidden_dim = state_dict[key].shape[0]
            break
    
    d_model = 768  # 默认值
    for key in state_dict.keys():
        if "input_proj.weight" in key:
            d_model = state_dict[key].shape[1]
            break
    
    num_layers = 0
    for key in state_dict.keys():
        if "cross_attn_layers" in key:
            layer_idx = int(key.split(".")[1])
            num_layers = max(num_layers, layer_idx + 1)
    if num_layers == 0:
        num_layers = 1  # 默认值
    
    # 检测 cand_encoder 层数
    cand_encoder_layers = 0
    for key in state_dict.keys():
        if "cand_encoder.layers" in key:
            layer_idx = int(key.split("layers.")[1].split(".")[0])
            cand_encoder_layers = max(cand_encoder_layers, layer_idx + 1)
    if cand_encoder_layers == 0:
        cand_encoder_layers = 1  # 默认值
    
    logger.info(f"推断模型配置: d_model={d_model}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    logger.info(f"模型类型: {model_type}")
    
    # 创建模型
    if model_type == "v4_6":
        # V4-6 模型
        use_step_emb = any("step_emb" in k for k in state_dict.keys())
        use_gru = any("decoder_gru" in k for k in state_dict.keys())
        
        # 检测 num_topics
        num_topics = 16  # 默认值
        for key in state_dict.keys():
            if "topic_prototypes" in key:
                num_topics = state_dict[key].shape[0]
                break
        
        # 检查环境变量中的 num_topics
        env_num_topics = os.environ.get("LEVER_LM_NUM_TOPICS", None)
        if env_num_topics:
            num_topics = int(env_num_topics)
        
        # 检查 checkpoint 中的 num_topics
        ckpt_num_topics = checkpoint.get("num_topics", None)
        if ckpt_num_topics:
            num_topics = ckpt_num_topics
        
        model = PointerSelectorV4_6_RL(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_step_emb=use_step_emb,
            use_gru=use_gru,
            num_topics=num_topics,
            cover_lambda_init=-2.0,  # 使用 softplus，初始化为负数
            label_smoothing=0.0,
            dropout=0.5
        )
        logger.info(f"创建 V4-6 模型 (GRU + Coverage {num_topics} topics, use_step_emb={use_step_emb}, use_gru={use_gru})")
    elif model_type == "v4_5":
        # V4-5 模型
        use_step_emb = any("step_emb" in k for k in state_dict.keys())
        use_gru = any("decoder_gru" in k for k in state_dict.keys())
        has_additive_attn = any("attn_Wq" in k for k in state_dict.keys())
        has_bilinear_attn = any("bilinear" in k for k in state_dict.keys())
        
        # 确定 attention_type
        if has_additive_attn:
            attention_type = 'additive'
        elif has_bilinear_attn:
            attention_type = 'bilinear'
        else:
            attention_type = 'dot'
        
        # 检查环境变量中的 attention_type
        env_attention_type = os.environ.get("LEVER_LM_ATTENTION_TYPE", None)
        if env_attention_type:
            attention_type = env_attention_type
        
        # 检查 checkpoint 中的 attention_type
        ckpt_attention_type = checkpoint.get("attention_type", None)
        if ckpt_attention_type:
            attention_type = ckpt_attention_type
        
        model = PointerSelectorV4_5_RL(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_step_emb=use_step_emb,
            use_gru=use_gru,
            attention_type=attention_type,
            label_smoothing=0.0,
            dropout=0.5
        )
        logger.info(f"创建 V4-5 模型 (GRU + {attention_type.upper()} Attention, use_step_emb={use_step_emb}, use_gru={use_gru})")
    elif model_type == "v4_4":
        # V4-4 模型
        use_step_emb = any("step_emb" in k for k in state_dict.keys())
        use_gru = any("decoder_gru" in k for k in state_dict.keys())
        model = PointerSelectorV4_4_RL(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_step_emb=use_step_emb,
            use_gru=use_gru,
            mmr_reduction='max',
            cand_encoder_layers=cand_encoder_layers,
            cand_encoder_heads=1,
            label_smoothing=0.0,
            dropout=0.5
        )
        logger.info(f"创建 V4-4 模型 (Cand Encoder + GRU + MMR, cand_encoder_layers={cand_encoder_layers})")
    elif model_type == "v4_3":
        # V4-3 模型
        use_step_emb = any("step_emb" in k for k in state_dict.keys())
        use_gru = any("decoder_gru" in k for k in state_dict.keys())
        model = PointerSelectorV4_3_RL(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_step_emb=use_step_emb,
            use_gru=use_gru,
            mmr_reduction='max',  # 默认使用 max
            label_smoothing=0.0,
            dropout=0.5
        )
        logger.info(f"创建 V4-3 模型 (GRU + MMR, use_step_emb={use_step_emb}, use_gru={use_gru})")
    elif model_type == "v4_2":
        # V4-2 模型
        use_step_emb = any("step_emb" in k for k in state_dict.keys())
        model = PointerSelectorV4_2_RL(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_step_emb=use_step_emb,
            label_smoothing=0.0,
            dropout=0.5
        )
        logger.info(f"创建 V4-2 模型 (GRU Pointer Decoder, use_step_emb={use_step_emb})")
    else:
        # V3 模型（默认）
        model = PointerSelectorV3(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            label_smoothing=0.0,
            dropout=0.5
        )
        logger.info("创建 V3 模型")
    
    # 加载权重
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"缺失参数: {len(missing)} 个")
    if unexpected:
        logger.warning(f"多余参数: {len(unexpected)} 个")
    
    model = model.to(device)
    model.eval()
    
    # 恢复kl_beta（如果有）
    if "kl_beta" in checkpoint:
        model.kl_beta = checkpoint["kl_beta"]
        logger.info(f"恢复kl_beta: {model.kl_beta}")
    
    logger.info(f"✓ 模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
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


# 测试代码已移除，避免在导入时意外执行
# 如需测试，请使用单独的测试脚本
