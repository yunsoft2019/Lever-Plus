"""
V2 模型模块

Bi-Encoder + Cross-Attention 指针选择器

包含：
- V2: 基础版本（Cross-Attention + query_update_gate）
- V4-2: GRU Pointer Decoder 版本
"""

from .pointer_selector_v2 import (
    PointerSelectorV2,
    PointerSelectorV2Config,
    build_model_v2
)
from .pointer_selector_v4_2 import (
    PointerSelectorV4_2,
    PointerSelectorV4_2Config,
    build_model_v4_2
)
from .adapter_builder import build_model_v2_with_adapter

__all__ = [
    'PointerSelectorV2',
    'PointerSelectorV2Config',
    'build_model_v2',
    'PointerSelectorV4_2',
    'PointerSelectorV4_2Config',
    'build_model_v4_2',
    'build_model_v2_with_adapter'
]

__version__ = '2.1.0'


