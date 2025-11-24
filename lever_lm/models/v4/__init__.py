"""
Pointer Selector V4: 离线强化学习（GRPO风格后训练）
"""

from .pointer_selector_v4 import (
    PointerSelectorV4,
    PointerSelectorV4Config,
    build_model_v4,
    load_v4_from_v3_checkpoint
)
from .adapter_builder import build_model_v4_with_adapter

__all__ = [
    'PointerSelectorV4',
    'PointerSelectorV4Config',
    'build_model_v4',
    'load_v4_from_v3_checkpoint',
    'build_model_v4_with_adapter'
]
