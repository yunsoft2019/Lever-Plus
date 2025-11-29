"""
Pointer Selector V3: 离线强化学习（GRPO风格后训练）
"""

from .pointer_selector_v3 import (
    PointerSelectorV3,
    PointerSelectorV3Config,
    build_model_v3,
    load_v3_from_v2_checkpoint
)
from .adapter_builder import build_model_v3_with_adapter

__all__ = [
    'PointerSelectorV3',
    'PointerSelectorV3Config',
    'build_model_v3',
    'load_v3_from_v2_checkpoint',
    'build_model_v3_with_adapter'
]
