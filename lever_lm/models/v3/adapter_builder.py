"""
适配器构建函数：创建适配 v2 训练流程的 v3 模型

V3 继承自 V2，架构完全相同，只是新增了强化学习相关方法。
V3 可以从 V2 的检查点加载（权重兼容）。
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
    use_lora: bool = False,  # 是否使用 LoRA
    lora_config: Optional[dict] = None,  # LoRA 配置参数
    cache_dir: Optional[str] = None,  # CLIP 模型缓存目录（可选）
    **kwargs
) -> PointerSelectorAdapter:
    """
    构建适配 v2 训练流程的 v3 模型
    
    V3 继承自 V2，架构完全相同，只是新增了强化学习相关方法（RCE、GRPO）。
    V3 可以从 V2 的检查点加载（权重兼容）。
    
    Args:
        config: PointerSelectorV3Config 参数（DictConfig 或已实例化的对象）
        clip_name: CLIP 模型名称
        query_encoding_flag: 查询编码标志（如 ['image', 'text']）
        icd_encoding_flag: ICD 编码标志（如 ['image', 'text']）
        adapter: 是否使用 adapter（默认 False）
        norm: 是否归一化（默认 True）
        device: 设备参数，用于确保 CLIP 模型加载到正确的 GPU
        use_lora: 是否使用 LoRA（默认 False）
        lora_config: LoRA 配置参数（可选）
        cache_dir: CLIP 模型缓存目录（可选）
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
            d_model=config_dict.get('d_model', 512),  # 默认512（CLIP base-patch32）
            K=config_dict.get('K', 2),
            shot_num=config_dict.get('shot_num', 2),
            label_smoothing=config_dict.get('label_smoothing', 0.0),
            dropout=config_dict.get('dropout', 0.5),
            hidden_dim=config_dict.get('hidden_dim', 256),
            num_heads=config_dict.get('num_heads', 1),
            attn_dropout=config_dict.get('attn_dropout', 0.1),
            num_layers=config_dict.get('num_layers', 3),
            # V3新增参数
            clip_epsilon=config_dict.get('clip_epsilon', 0.2),
            kl_beta=config_dict.get('kl_beta', 0.1),
            advantage_clip=config_dict.get('advantage_clip', 5.0)
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
    
    # 处理 lora_config：可能是 DictConfig 或 dict
    if lora_config is not None and isinstance(lora_config, DictConfig):
        lora_config = OmegaConf.to_container(lora_config, resolve=True)
    
    # 从 kwargs 中获取 cache_dir（如果配置文件中提供了）
    if cache_dir is None:
        cache_dir = kwargs.get('cache_dir', None)
    # 如果 cache_dir 是 DictConfig，转换为字符串
    if isinstance(cache_dir, DictConfig):
        cache_dir = OmegaConf.to_container(cache_dir, resolve=True)
        if cache_dir == 'null' or cache_dir is None:
            cache_dir = None
    
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
        use_lora=use_lora,  # 传递 use_lora 参数
        lora_config=lora_config,  # 传递 lora_config 参数
        cache_dir=cache_dir,  # 传递 cache_dir 参数
    )
    
    return adapter_model
