"""
V1 模型模块

Bi-Encoder 指针选择器基础版本
"""

from .pointer_selector_v1 import (
    PointerSelectorV1,
    PointerSelectorV1Config,
    build_model_v1
)

__all__ = [
    'PointerSelectorV1',
    'PointerSelectorV1Config',
    'build_model_v1'
]

__version__ = '1.0.0'






