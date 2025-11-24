"""
Pointer Selector V3: Bi-Encoder + 排序学习（Ranking Learning）
"""

from .pointer_selector_v3 import (
    PointerSelectorV3,
    PointerSelectorV3Config,
    build_model_v3,
    load_v3_from_checkpoint
)
from .adapter_builder import build_model_v3_with_adapter

__all__ = [
    'PointerSelectorV3',
    'PointerSelectorV3Config',
    'build_model_v3',
    'load_v3_from_checkpoint',
    'build_model_v3_with_adapter'
]
