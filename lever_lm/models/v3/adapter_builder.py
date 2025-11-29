"""
适配器构建函数：创建适配 v0 训练流程的 v3 模型
"""
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from ..adapter import PointerSelectorAdapter
from .pointer_selector_v3 import PointerSelectorV3, PointerSelectorV3Config, build_model_v3


def build_model_v3_with_adapter(
    config: DictConfig,
    clip_name: str = 'openai/clip-vit-base-patch32',
    query_encoding_flag: Optional[list] = None,
    icd_encoding_flag: Optional[list] = None,
    adapter: bool = False,
    norm: bool = True,
    device: Optional[str] = None,  # 设备参数，用于确保 CLIP 模型加载到正确的 GPU
    **kwargs
) -> PointerSelectorAdapter:
    """
    构建适配 v0 训练流程的 v3 模型
    
    Args:
        config: PointerSelectorV3Config 参数（DictConfig 或已实例化的对象）
        clip_name: CLIP 模型名称
        query_encoding_flag: 查询编码标志（如 ['image', 'text']）
        icd_encoding_flag: ICD 编码标志（如 ['image', 'text']）
        adapter: 是否使用 adapter（默认 False）
        norm: 是否归一化（默认 True）
        **kwargs: 其他参数（忽略）
    
    Returns:
        PointerSelectorAdapter: 适配器包装的 v3 模型
    """
    # 处理配置参数：可能是 DictConfig 或已实例化的 PointerSelectorV3Config
    if isinstance(config, PointerSelectorV3Config):
        # 已经实例化，直接使用
        pointer_config = config
    else:
        # 是 DictConfig，需要转换
        config_dict = OmegaConf.to_container(config, resolve=True)
        pointer_config = PointerSelectorV3Config(
            d_model=config_dict['d_model'],
            K=config_dict['K'],
            shot_num=config_dict['shot_num'],
            label_smoothing=config_dict.get('label_smoothing', 0.0),
            dropout=config_dict.get('dropout', 0.5),
            hidden_dim=config_dict.get('hidden_dim', 256),
            num_heads=config_dict.get('num_heads', 1),
            attn_dropout=config_dict.get('attn_dropout', 0.1),
            num_layers=config_dict.get('num_layers', 3),
            # V3 特有参数
            enable_rce=config_dict.get('enable_rce', False),
            enable_grpo=config_dict.get('enable_grpo', False),
            rce_temperature=config_dict.get('rce_temperature', 1.0),
            ppo_epsilon=config_dict.get('ppo_epsilon', 0.2),
            advantage_clip=config_dict.get('advantage_clip', 5.0),
            kl_beta=config_dict.get('kl_beta', 0.01),
            reward_norm=config_dict.get('reward_norm', 'zscore')
        )
    
    # 构建 v3 模型
    pointer_selector = build_model_v3(pointer_config)
    
    # 处理编码标志
    if query_encoding_flag is None:
        query_encoding_flag = ['image', 'text']
    elif isinstance(query_encoding_flag, DictConfig):
        query_encoding_flag = OmegaConf.to_container(query_encoding_flag, resolve=True)
    
    if icd_encoding_flag is None:
        icd_encoding_flag = ['image', 'text']
    elif isinstance(icd_encoding_flag, DictConfig):
        icd_encoding_flag = OmegaConf.to_container(icd_encoding_flag, resolve=True)
    
    # 创建适配器
    adapter_model = PointerSelectorAdapter(
        pointer_selector_model=pointer_selector,
        clip_name=clip_name,
        query_encoding_flag=query_encoding_flag,
        icd_encoding_flag=icd_encoding_flag,
        adapter=adapter,
        norm=norm,
        K=pointer_config.K,
        device=device,  # 传递 device 参数，确保 CLIP 模型加载到正确的 GPU
    )
    
    return adapter_model

