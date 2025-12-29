"""
V3 模型模块

v2 + 离线强化学习（RCE预热 + GRPO后训练）

包含：
- V3: 基于V2的RL版本
- V4-2 RL: 基于V4-2 (GRU Decoder) 的RL版本
"""

from .pointer_selector_v3 import (
    PointerSelectorV3,
    PointerSelectorV3Config,
    build_model_v3
)
from .pointer_selector_v4_2_rl import (
    PointerSelectorV4_2_RL,
    PointerSelectorV4_2_RLConfig,
    build_model_v4_2_rl
)
from .pointer_selector_v4_6_rl import (
    PointerSelectorV4_6_RL,
    PointerSelectorV4_6_RLConfig,
    build_model_v4_6_rl
)
from .adapter_builder import (
    build_model_v3_with_adapter
)
from .dataset_v3 import (
    BeamDataset,
    BeamDatasetWithEmbedding,
    RLBeamDatasetWithEmbedding,
    collate_fn_v3,
    collate_fn_rl_v3,
    load_beam_data,
    split_beam_data
)
from .inference_v3 import (
    load_v3_from_grpo_checkpoint,
    load_v3_from_sft_checkpoint,
    predict_with_v3
)
from .rl_data_generation import (
    compute_step_logits,
    sample_pointer_with_temperature,
    generate_pointer_candidates_for_query,
    evaluate_pointer_candidate
)

__all__ = [
    'PointerSelectorV3',
    'PointerSelectorV3Config',
    'build_model_v3',
    # V4-2 RL
    'PointerSelectorV4_2_RL',
    'PointerSelectorV4_2_RLConfig',
    'build_model_v4_2_rl',
    # V4-6 RL
    'PointerSelectorV4_6_RL',
    'PointerSelectorV4_6_RLConfig',
    'build_model_v4_6_rl',
    # Adapter
    'build_model_v3_with_adapter',
    # Dataset
    'BeamDataset',
    'BeamDatasetWithEmbedding',
    'RLBeamDatasetWithEmbedding',
    'collate_fn_v3',
    'collate_fn_rl_v3',
    'load_beam_data',
    'split_beam_data',
    # Inference
    'load_v3_from_grpo_checkpoint',
    'load_v3_from_sft_checkpoint',
    'predict_with_v3',
    # RL 数据生成
    'compute_step_logits',
    'sample_pointer_with_temperature',
    'generate_pointer_candidates_for_query',
    'evaluate_pointer_candidate'
]

__version__ = '3.1.0'
