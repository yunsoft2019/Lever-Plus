"""
Lever-LM 工具模块
"""

from .reward_utils import (
    normalize_rewards_zscore,
    clip_advantages,
    compute_group_relative_advantage,
    compute_softmax_weights,
    compute_temperature_schedule,
    compute_kl_penalty,
    adaptive_kl_beta
)

# 从 lever_lm/utils.py 文件导入 init_interface
# 注意：lever_lm/utils.py 是一个独立文件，与 lever_lm/utils/ 目录并存
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# 直接从 lever_lm.utils 模块文件导入
import importlib.util
_utils_file = os.path.join(_parent_dir, 'utils.py')
_spec = importlib.util.spec_from_file_location("lever_lm_utils_file", _utils_file)
_utils_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_module)
init_interface = _utils_module.init_interface
beam_filter = _utils_module.beam_filter
encode_image = _utils_module.encode_image
encode_text = _utils_module.encode_text
recall_sim_feature = _utils_module.recall_sim_feature
data_split = _utils_module.data_split
collate_fn = _utils_module.collate_fn

__all__ = [
    'normalize_rewards_zscore',
    'clip_advantages',
    'compute_group_relative_advantage',
    'compute_softmax_weights',
    'compute_temperature_schedule',
    'compute_kl_penalty',
    'adaptive_kl_beta',
    'init_interface',
    'beam_filter',
    'encode_image',
    'encode_text',
    'recall_sim_feature',
    'data_split',
    'collate_fn',
]
