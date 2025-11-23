"""
V2 模型模块

Bi-Encoder + Cross-Attention 指针选择器
"""

from .pointer_selector_v2 import (
    PointerSelectorV2,
    PointerSelectorV2Config,
    build_model_v2
)
from .adapter_builder import build_model_v2_with_adapter

__all__ = [
    'PointerSelectorV2',
    'PointerSelectorV2Config',
    'build_model_v2',
    'build_model_v2_with_adapter'
]

__version__ = '2.0.0'


