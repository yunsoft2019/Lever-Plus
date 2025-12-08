"""
V3 模型模块

v2 + 离线强化学习（RCE预热 + GRPO后训练）
"""

from .pointer_selector_v3 import (
    PointerSelectorV3,
    PointerSelectorV3Config,
    build_model_v3
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
    'build_model_v3_with_adapter',
    'BeamDataset',
    'BeamDatasetWithEmbedding',
    'RLBeamDatasetWithEmbedding',
    'collate_fn_v3',
    'collate_fn_rl_v3',
    'load_beam_data',
    'split_beam_data',
    'load_v3_from_grpo_checkpoint',
    'load_v3_from_sft_checkpoint',
    'predict_with_v3',
    # RL 数据生成
    'compute_step_logits',
    'sample_pointer_with_temperature',
    'generate_pointer_candidates_for_query',
    'evaluate_pointer_candidate'
]

__version__ = '3.0.0'
