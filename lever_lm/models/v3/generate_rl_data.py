"""
RL 数据生成脚本：生成包含 beam + 温度采样 + correctness 的完整 RL 数据

按照强化学习.md §3 实现：
1. 对每个 query 生成 pointer 候选（beam + 温度采样 + 随机组合）
2. 对每个 pointer 调用 VQA 模型计算 correctness
3. 保存为新的数据格式（包含 vqa_correct 和 vqa_acc_score）

使用方法：
    python -m lever_lm.models.v3.generate_rl_data \
        --sft_ckpt <path_to_v2_checkpoint> \
        --beam_data <existing_beam_data.json> \
        --output_path <output_rl_data.json> \
        --vqa_model <vqa_model_name> \
        --dataset <dataset_name> \
        --num_beams 5 \
        --temps 1.0 1.3 \
        --num_samples_per_temp 2 \
        --num_random 1

作者: Lever-Plus Team
日期: 2025-12-06
参考: 强化学习.md
"""

import json
import argparse
import torch
import os
import tempfile
import sys
import contextlib
import re
from io import StringIO
from typing import Dict, List, Optional, Tuple
from collections import Counter
from difflib import SequenceMatcher
from datetime import datetime
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# 优化：使用/mnt/share作为临时文件目录（有更多空间）
# 如果/mnt/share可用，使用它；否则使用系统默认的/tmp
TEMP_DIR = "/mnt/share/yiyun/Projects/Lever-Plus/tmp" if os.path.exists("/mnt/share") else None
if TEMP_DIR:
    os.makedirs(TEMP_DIR, exist_ok=True)

from lever_lm.models.v3.rl_data_generation import (
    generate_pointer_candidates_for_query,
    evaluate_pointer_candidate
)
from lever_lm.models.v3 import PointerSelectorV3
from lever_lm.utils import init_interface
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy as compute_vqa_accuracy_metric, VQA, VQAEval

# 导入根目录的utils.py（避免与lever_lm/utils/冲突）
import importlib.util
import os as _os
_utils_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
vqa_postprocess = _root_utils.vqa_postprocess
load_ds = _root_utils.load_ds


def load_sft_model(checkpoint_path: str, device: torch.device) -> PointerSelectorV3:
    """
    加载 SFT 模型（v2 checkpoint）
    
    Args:
        checkpoint_path: checkpoint 路径
        device: 设备
    
    Returns:
        model: PointerSelectorV3 模型
    """
    # TODO: 根据实际 checkpoint 格式加载
    # 这里需要根据你的 checkpoint 格式进行适配
    model = PointerSelectorV3(
        d_model=512,  # 根据实际配置调整
        K=64,
        shot_num=2
    )
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 处理不同格式的 checkpoint
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        # 去掉前缀
        state_dict = {k.replace('lever_lm.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # 检查缺失的参数
    if missing:
        print(f"[警告] 加载 checkpoint 时有 {len(missing)} 个参数缺失，例如：")
        for k in list(missing)[:10]:
            print(f"  - missing: {k}")
        if len(missing) > 10:
            print(f"  ... 还有 {len(missing) - 10} 个参数缺失")
    
    # 检查多余的参数
    if unexpected:
        print(f"[警告] 有 {len(unexpected)} 个多余参数，例如：")
        for k in list(unexpected)[:10]:
            print(f"  - unexpected: {k}")
        if len(unexpected) > 10:
            print(f"  ... 还有 {len(unexpected) - 10} 个多余参数")
    
    # 如果缺失的关键参数太多，直接 raise
    if len(missing) > 1000:
        raise RuntimeError(
            f"Checkpoint 与当前模型结构差异过大（缺失 {len(missing)} 个参数），请检查模型配置。"
        )
    
    # 如果没有missing和unexpected，说明checkpoint完全匹配
    if not missing and not unexpected:
        print("✓ Checkpoint 参数完全匹配，加载成功")
    
    model.to(device)
    model.eval()
    
    return model


def load_vqa_model(model_name: str, device: torch.device, cfg: Optional[DictConfig] = None):
    """
    加载 VQA 模型
    
    Args:
        model_name: 模型名称（如 "qwen2.5_vl_3B" 或 "flamingo_3B"）
        device: 设备
        cfg: 配置对象（可选，如果提供则使用，否则从config文件加载）
    
    Returns:
        vqa_model: VQA 模型 interface
    """
    # 映射模型名称到 init_interface 期望的格式
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        infer_model_name = "Qwen2.5-VL"  # init_interface 需要这个格式
        hf_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif "flamingo" in model_name_lower:
        infer_model_name = "flamingo"
        hf_model_name = None
    elif "idefics" in model_name_lower:
        infer_model_name = "idefics"
        hf_model_name = None
    else:
        infer_model_name = model_name
        hf_model_name = None
    
    if cfg is None:
        # 【重要】从config文件加载配置，确保与v0/v1/v2完全一致
        # 参考 configs/infer_model/qwen2.5_vl_3B.yaml 和 configs/task/vqa.yaml
        infer_model_config_path = "configs/infer_model/qwen2.5_vl_3B.yaml"
        task_config_path = "configs/task/vqa.yaml"
        
        # 加载infer_model配置
        if os.path.exists(infer_model_config_path):
            infer_model_cfg = OmegaConf.load(infer_model_config_path)
        else:
            raise FileNotFoundError(f"配置文件不存在: {infer_model_config_path}")
        
        # 加载task配置
        if os.path.exists(task_config_path):
            task_cfg = OmegaConf.load(task_config_path)
        else:
            raise FileNotFoundError(f"配置文件不存在: {task_config_path}")
        
        # 合并配置，确保与v0/v1/v2完全一致
        cfg = OmegaConf.create({
            "infer_model": {
                "name": infer_model_name,  # 使用映射后的名称
                "model_name": hf_model_name,
                "load_from_local": False,
                "precision": "bf16",
                # 【关键】使用config文件中的值，确保与v0/v1/v2一致
                "icd_join_char": infer_model_cfg.get("icd_join_char", "<|endofchunk|>"),  # 从config加载
                "system_prompt": infer_model_cfg.get("system_prompt", ""),  # 从config加载
            },
            "task": {
                # 【关键】使用config文件中的vqa_prompt_template，确保与v0/v1/v2一致
                # configs/infer_model/qwen2.5_vl_3B.yaml: vqa_prompt_template: "Question:<Q> Short answer:<A>"
                "template": infer_model_cfg.get("vqa_prompt_template", "Question:<Q> Short answer:<A>"),
                # 【关键】使用config文件中的column_token_map，确保与v0/v1/v2一致
                # configs/task/vqa.yaml: column_token_map: {question: "<Q>", answer: "<A>"}
                "column_token_map": OmegaConf.to_container(task_cfg.get("column_token_map", {"question": "<Q>", "answer": "<A>"})),
                "instruction": infer_model_cfg.get("vqa_instruction", ""),
                "image_field": "image",
                "output_column": "answer",
            },
            "precision": "bf16",
        })
    
    # 使用 init_interface 加载模型
    interface = init_interface(cfg, device=device)
    return interface


def build_vqa_prompt_with_label(
    interface,
    image,
    question: str,
    answer: str,
    ex1: Dict,
    ex2: Dict
) -> Dict:
    """
    构建带标签的 VQA prompt（用于计算 cond_prob）
    
    按照 2025-12-13需求.md P1需求4 的要求：
    - 用于计算 vqa_gt_prob（GT 概率）
    - 使用 teacher-forcing 方式，将 GT answer 作为标签
    
    Args:
        interface: VQA 模型 interface
        image: 查询图像
        question: 查询问题
        answer: GT 答案（标签）
        ex1: 第一个示例（包含 image, question, answer）
        ex2: 第二个示例（包含 image, question, answer）
    
    Returns:
        input_dict: 准备好的输入字典（用于 get_cond_prob）
        mask_length: mask 长度（用于 get_cond_prob）
    """
    # 构造 data_sample_list（示例 + 查询）
    query_sample = {
        "image": image,
        "question": question,
        "answer": answer,  # 添加标签
    }
    data_sample_list = [ex1, ex2, query_sample]
    
    # 使用 transfer_prompts 转换为 prompt 格式（is_last_for_generation=False，包含标签）
    prompts = interface.transfer_prompts(
        [data_sample_list], 
        is_last_for_generation=False,
        query_label=answer  # 传入标签
    )
    
    # 使用 prepare_input 转换为 tensor 格式
    input_dict = interface.prepare_input(
        prompts, is_last_for_generation=False
    )
    
    # 处理 BatchFeature 对象，转换为 dict
    if hasattr(input_dict, 'data'):
        input_dict = dict(input_dict.data)
    elif not isinstance(input_dict, dict):
        input_dict = dict(input_dict)
    
    # 将数据移动到设备
    data = {k: v.to(interface.device) if isinstance(v, torch.Tensor) else v 
           for k, v in input_dict.items()}
    
    # 处理 Qwen2.5-VL 的特殊情况（image_grid_thw）
    if 'image_grid_thw' in data:
        if data['image_grid_thw'].dim() == 3 and data['image_grid_thw'].shape[0] == 1:
            data['image_grid_thw'] = data['image_grid_thw'].squeeze(0)
            if 'image_nums' in data:
                if isinstance(data['image_nums'], torch.Tensor) and data['image_nums'].numel() > 1:
                    data['image_nums'] = data['image_nums'][0:1]
                elif isinstance(data['image_nums'], list) and len(data['image_nums']) > 0:
                    data['image_nums'] = torch.tensor([data['image_nums'][0]], dtype=torch.long)
        elif data['image_grid_thw'].dim() == 2:
            if 'image_nums' not in data:
                num_images = data['image_grid_thw'].shape[0]
                data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
    
    # 计算 mask_length（prompt 长度，不包括答案部分）
    # 参考 utils.py 中 get_info_score 的计算方式：
    # mask_length 是 prompt 的长度（不包括答案部分）
    # 对于带标签的情况，需要计算 prompt 的长度（不包括答案）
    
    # 获取 prompt 的长度（不包括答案）
    # 先构建不带答案的 prompt，计算其长度
    query_sample_no_label = {
        "image": image,
        "question": question,
    }
    data_sample_list_no_label = [ex1, ex2, query_sample_no_label]
    
    # 使用 concat_prompt 构建完整的 prompt 字符串
    mask_context = interface.concat_prompt(
        data_sample_list_no_label,
        add_eos_token=False,
        is_last_for_generation=True
    )
    
    # 计算 mask_length（prompt 长度，不包括答案）
    mask_length = interface.get_input_token_num(mask_context)
    
    return data, mask_length


def build_vqa_prompt_and_generate(
    interface,
    image,
    question: str,
    ex1: Dict,
    ex2: Dict,
    generation_kwargs: Optional[Dict] = None
) -> str:
    """
    构建 VQA prompt 并生成答案
    
    按照 icl_inference.py 的方式：
    1. 构造 data_sample_list（示例 + 查询）
    2. 使用 interface.transfer_prompts() 转换为 prompt 格式
    3. 使用 interface.prepare_input() 转换为 tensor 格式
    4. 调用 interface.generate() 生成答案
    5. 解码并后处理
    
    Args:
        interface: VQA 模型 interface
        image: 查询图像
        question: 查询问题
        ex1: 第一个示例（包含 image, question, answer）
        ex2: 第二个示例（包含 image, question, answer）
        generation_kwargs: 生成参数（可选）
    
    Returns:
        answer: 生成的答案字符串
    """
    if generation_kwargs is None:
        generation_kwargs = {}
    
    # 构造 data_sample_list（示例 + 查询）
    query_sample = {
        "image": image,
        "question": question,
    }
    data_sample_list = [ex1, ex2, query_sample]
    
    # 使用 transfer_prompts 转换为 prompt 格式
    prompts = interface.transfer_prompts(
        [data_sample_list], is_last_for_generation=True
    )
    
    # 使用 prepare_input 转换为 messages 格式（tensor）
    input_dict = interface.prepare_input(
        prompts, is_last_for_generation=True
    )
    
    # 处理 BatchFeature 对象，转换为 dict
    if hasattr(input_dict, 'data'):
        input_dict = dict(input_dict.data)
    elif not isinstance(input_dict, dict):
        input_dict = dict(input_dict)
    
    # 将数据移动到设备
    data = {k: v.to(interface.device) if isinstance(v, torch.Tensor) else v 
           for k, v in input_dict.items()}
    
    # 处理 Qwen2.5-VL 的特殊情况（image_grid_thw）
    if 'image_grid_thw' in data:
        if data['image_grid_thw'].dim() == 3 and data['image_grid_thw'].shape[0] == 1:
            data['image_grid_thw'] = data['image_grid_thw'].squeeze(0)
            if 'image_nums' in data:
                if isinstance(data['image_nums'], torch.Tensor) and data['image_nums'].numel() > 1:
                    data['image_nums'] = data['image_nums'][0:1]
                elif isinstance(data['image_nums'], list) and len(data['image_nums']) > 0:
                    data['image_nums'] = torch.tensor([data['image_nums'][0]], dtype=torch.long)
        elif data['image_grid_thw'].dim() == 2:
            if 'image_nums' not in data:
                num_images = data['image_grid_thw'].shape[0]
                data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
    
    # 获取 prompt 长度
    prompt_len = int(data["attention_mask"].shape[1])
    
    # 生成答案
    with torch.inference_mode():
        outputs = interface.generate(
            **data,
            eos_token_id=interface.tokenizer.eos_token_id,
            pad_token_id=interface.tokenizer.pad_token_id,
            **generation_kwargs,
        )
    
    # 解码生成结果
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.tolist()
    
    # 确保 outputs 是列表格式
    if not isinstance(outputs, list):
        outputs = [outputs]
    if len(outputs) > 0 and not isinstance(outputs[0], list):
        outputs = [outputs]
    
    # 解码：只取 prompt 之后的部分
    generated = interface.tokenizer.batch_decode(
        [output[prompt_len:] for output in outputs],
        skip_special_tokens=True,
    )
    
    # 后处理得到 answer
    prediction = generated[0] if generated else ""
    # 确保prediction是字符串（严格检查）
    if prediction is None:
        prediction = ""
    if not isinstance(prediction, str):
        try:
            prediction = str(prediction) if prediction is not None else ""
        except Exception:
            prediction = ""
    # 再次确保prediction不是None且是字符串
    if prediction is None or not isinstance(prediction, str):
        prediction = ""
    
    model_name = interface.__class__.__name__.lower()
    try:
        if "qwen" in model_name:
            answer = vqa_postprocess(prediction, model_name="qwen2.5_vl_3B")
            if answer is None or not isinstance(answer, str):
                answer = ""
        elif "flamingo" in model_name:
            answer = vqa_postprocess(prediction, model_name="flamingo_3B")
            if answer is None or not isinstance(answer, str):
                answer = ""
        else:
            # 确保prediction是字符串后再调用strip
            if prediction and isinstance(prediction, str):
                answer = prediction.strip()
            else:
                answer = ""
    except Exception as e:
        # 如果后处理失败，使用原始prediction（确保安全）
        if prediction and isinstance(prediction, str):
            answer = prediction.strip()
        else:
            answer = ""
    
    # 确保answer是字符串
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""
    
    # 获取原始生成结果（postprocess前），确保是字符串
    raw_generation = prediction if prediction is not None else ""
    if not isinstance(raw_generation, str):
        raw_generation = str(raw_generation) if raw_generation is not None else ""
    
    # 获取prompt文本（可选）
    prompt_text = None
    try:
        # 使用concat_prompt构建完整的prompt字符串
        prompt_text = interface.concat_prompt(
            data_sample_list,
            add_eos_token=False,
            is_last_for_generation=True
        )
    except Exception as e:
        # 如果获取prompt失败，设为None
        prompt_text = None
    
    return {
        "pred_answer": answer,
        "raw_generation": raw_generation,
        "prompt_text": prompt_text,
        "prompt_len": prompt_len,
    }


def _tok(s: str) -> List[str]:
    """分词函数：按空格/数字/字母分词"""
    return re.findall(r"[a-z0-9]+", s.lower())


def token_f1(a: str, b: str) -> float:
    """计算token级别的F1分数"""
    A, B = _tok(a), _tok(b)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    from collections import Counter
    ca, cb = Counter(A), Counter(B)
    common = sum((ca & cb).values())
    prec = common / max(1, len(A))
    rec = common / max(1, len(B))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def edit_sim(a: str, b: str) -> float:
    """计算字符级编辑相似度"""
    return SequenceMatcher(None, a, b).ratio()


def compute_relevance(pred: str, gt_list: List[str]) -> Dict[str, float]:
    """
    计算pred与gt_list的相关性指标
    
    Args:
        pred: 预测答案（postprocessed）
        gt_list: ground truth答案列表（normalized）
    
    Returns:
        dict包含: vqa_rel_token_f1, vqa_rel_edit_sim, vqa_rel_score
    """
    if pred is None:
        pred = ""
    pred = pred.strip().lower()
    gt_list = [(g or "").strip().lower() for g in gt_list]
    
    if pred == "" or len(gt_list) == 0:
        return {
            "vqa_rel_token_f1": 0.0,
            "vqa_rel_edit_sim": 0.0,
            "vqa_rel_score": 0.0
        }
    
    # 计算与每个gt的最大相似度
    f1_max = max(token_f1(pred, g) for g in gt_list)
    ed_max = max(edit_sim(pred, g) for g in gt_list)
    
    # rel_score取两者最大值（或平均，这里用最大值）
    rel_score = max(f1_max, ed_max)
    
    return {
        "vqa_rel_token_f1": f1_max,
        "vqa_rel_edit_sim": ed_max,
        "vqa_rel_score": rel_score
    }


def compute_expected_vqa_acc_prob(
    interface,
    image,
    question: str,
    ground_truth_answers: List[str],
    ex1: Dict,
    ex2: Dict
) -> float:
    """
    计算期望 VQA 准确率概率（vqa_gt_prob）
    
    按照 2025-12-13需求.md P1需求4 的要求：
    - 计算给定 prompt（含 ICD）下，模型对任一 GT 答案字符串的概率质量
    - 使用 teacher-forcing / cond_prob 方式
    - 按 VQA 计分方式加权求和
    
    Args:
        interface: VQA 模型 interface
        image: 查询图像
        question: 查询问题
        ground_truth_answers: GT 答案列表（可能包含重复）
        ex1: 第一个示例（包含 image, question, answer）
        ex2: 第二个示例（包含 image, question, answer）
    
    Returns:
        vqa_gt_prob: float [0, 1]，期望 VQA 准确率概率
    """
    # 1) 统计 GT answers 频次（已做 normalization）
    # 处理 ground_truth_answers：如果是字典列表，提取 answer 字段
    if ground_truth_answers and isinstance(ground_truth_answers[0], dict):
        gt_answers_str = [ans.get("answer", "") if isinstance(ans, dict) else str(ans) 
                         for ans in ground_truth_answers]
    elif ground_truth_answers and isinstance(ground_truth_answers[0], str):
        gt_answers_str = ground_truth_answers
    else:
        gt_answers_str = [str(ans) for ans in ground_truth_answers] if ground_truth_answers else []
    
    # 归一化答案（使用 vqa_postprocess）
    gt_answers_normalized = [vqa_postprocess(ans, model_name="qwen2.5_vl_3B") for ans in gt_answers_str]
    
    # 统计去重后的答案及其频次
    from collections import Counter
    uniq_counter = Counter(gt_answers_normalized)
    
    # 2) 对每个 uniq answer 做 teacher forcing cond_prob
    probs = {}
    for ans, cnt in uniq_counter.items():
        try:
            # 构建带标签的输入
            x_input, mask_len = build_vqa_prompt_with_label(
                interface=interface,
                image=image,
                question=question,
                answer=ans,
                ex1=ex1,
                ex2=ex2
            )
            
            # 计算 cond_prob（返回概率，不是 logprob）
            # get_cond_prob 返回 exp(-ce_loss)，已经是概率
            p = interface.get_cond_prob(x_input, mask_length=[mask_len])
            
            # 处理返回值：可能是标量或 tensor
            if isinstance(p, torch.Tensor):
                p = p.item() if p.numel() == 1 else p.mean().item()
            
            probs[ans] = float(p)
        except Exception as e:
            # 如果计算失败，设为 0
            print(f"警告：计算 cond_prob 失败 (answer={ans}): {e}")
            probs[ans] = 0.0
    
    # 3) 按 VQA 计分权重加权求和
    # VQA 计分方式：w(a) = min(count(a)/3.0, 1.0)
    score = 0.0
    for ans, cnt in uniq_counter.items():
        w = min(cnt / 3.0, 1.0)
        p = probs.get(ans, 0.0)
        score += p * w
    
    return float(score)


def compute_vqa_accuracy(
    pred_answer: str,
    ground_truth_answers,
    question_id: Optional[str] = None,
    val_ques_path: Optional[str] = None,
    val_ann_path: Optional[str] = None,
    vqa_cache: Optional[VQA] = None
) -> Tuple[int, float, bool]:
    """
    计算 VQA 准确率
    
    使用 open_mmicl.metrics.vqa_metrics.compute_vqa_accuracy
    如果提供了 val_ques_path 和 val_ann_path，使用文件方式计算
    否则使用简单的字符串匹配方式
    
    Args:
        pred_answer: 预测答案
        ground_truth_answers: 标准答案列表（可以是字符串列表或字典列表）
        question_id: 问题 ID（可选，用于文件方式）
        val_ques_path: 验证集问题文件路径（可选）
        val_ann_path: 验证集标注文件路径（可选）
    
    Returns:
        correct: 0/1（是否正确）
        acc_score: float [0,1]（准确率分数）
        used_file_metric: bool（是否使用了文件方式计算）
    """
    # 处理 ground_truth_answers：如果是字典列表，提取 answer 字段
    if not ground_truth_answers or len(ground_truth_answers) == 0:
        # 空列表：返回空列表
        gt_answers_str = []
    elif isinstance(ground_truth_answers[0], dict):
        # 字典格式：提取 "answer" 字段
        gt_answers_str = [ans.get("answer", "") if isinstance(ans, dict) else str(ans) 
                         for ans in ground_truth_answers]
    elif isinstance(ground_truth_answers[0], str):
        # 字符串格式：直接使用
        gt_answers_str = ground_truth_answers
    else:
        # 其他格式：转换为字符串列表
        gt_answers_str = [str(ans) for ans in ground_truth_answers] if ground_truth_answers else []
    
    # 如果提供了文件路径，使用文件方式计算（更准确）
    if val_ques_path and val_ann_path and question_id:
        try:
            # 如果提供了缓存的 VQA 对象，使用缓存（避免重复加载）
            if vqa_cache is not None:
                # 创建临时结果文件（使用指定的临时目录，避免磁盘空间不足）
                temp_kwargs = {'mode': 'w', 'suffix': '.json', 'delete': False}
                if TEMP_DIR:
                    temp_kwargs['dir'] = TEMP_DIR
                with tempfile.NamedTemporaryFile(**temp_kwargs) as f:
                    temp_result_file = f.name
                    json.dump([{
                        "answer": pred_answer,
                        "question_id": question_id,
                    }], f, indent=4)
                
                try:
                    # 使用缓存的 VQA 对象，抑制打印输出（避免干扰 tqdm 进度条）
                    # 使用 StringIO 捕获输出，而不是完全重定向，这样不会影响 tqdm
                    with contextlib.redirect_stdout(StringIO()):
                        vqaRes = vqa_cache.loadRes(temp_result_file, val_ques_path)
                        vqaEval = VQAEval(vqa_cache, vqaRes, n=2)
                        # 只评估当前问题
                        vqaEval.params = {"question_id": [int(question_id)]}
                        vqaEval.evaluate()
                    
                    # 获取单个问题的准确率
                    if int(question_id) in vqaEval.evalQA:
                        accuracy = vqaEval.evalQA[int(question_id)]
                    else:
                        accuracy = 0.0
                    
                    # 处理准确率格式
                    if accuracy > 1:
                        acc_score = accuracy / 100.0
                    else:
                        acc_score = accuracy
                    
                    correct = 1 if acc_score > 0.0 else 0
                    
                    return correct, acc_score, True  # 使用了文件方式
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_result_file):
                        os.remove(temp_result_file)
            else:
                # 没有缓存，使用标准评估函数（会重新加载文件）
                # 创建临时结果文件（使用指定的临时目录，避免磁盘空间不足）
                temp_kwargs = {'mode': 'w', 'suffix': '.json', 'delete': False}
                if TEMP_DIR:
                    temp_kwargs['dir'] = TEMP_DIR
                with tempfile.NamedTemporaryFile(**temp_kwargs) as f:
                    temp_result_file = f.name
                    json.dump([{
                        "answer": pred_answer,
                        "question_id": question_id,
                    }], f, indent=4)
                
                try:
                    # 使用标准评估函数
                    accuracy = compute_vqa_accuracy_metric(
                        temp_result_file,
                        val_ques_path,
                        val_ann_path
                    )
                    
                    # 处理准确率格式
                    if accuracy > 1:
                        acc_score = accuracy / 100.0
                    else:
                        acc_score = accuracy
                    
                    correct = 1 if acc_score > 0.0 else 0
                    
                    return correct, acc_score, True  # 使用了文件方式
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_result_file):
                        os.remove(temp_result_file)
        except Exception as e:
            pass  # 静默回退到简单匹配
    
    # 简单匹配方式：检查预测答案是否在标准答案列表中（不区分大小写）
    # 确保 pred_answer 是字符串
    if pred_answer is None:
        pred_answer = ""
    if not isinstance(pred_answer, str):
        pred_answer = str(pred_answer)
    pred_answer_lower = pred_answer.lower().strip()
    gt_answers_lower = [ans.lower().strip() if ans else "" for ans in gt_answers_str]
    
    # 精确匹配
    if pred_answer_lower in gt_answers_lower:
        return 1, 1.0, False  # 使用了 fallback
    
    # 部分匹配（检查预测答案是否包含标准答案，或标准答案是否包含预测答案）
    for gt_ans in gt_answers_lower:
        if pred_answer_lower in gt_ans or gt_ans in pred_answer_lower:
            # 部分匹配，给予较低的分数
            return 1, 0.5, False  # 使用了 fallback
    
    # 不匹配
    return 0, 0.0, False  # 使用了 fallback


def generate_rl_data(
    sft_model: PointerSelectorV3,
    vqa_model,
    beam_data: Dict,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_indices: List[int],
    dataset,
    num_beams: int = 5,
    temps: tuple = (1.0, 1.3),
    num_samples_per_temp: int = 2,
    num_random: int = 1,
    num_retrieval: int = 5,  # 新增：retrieval方法的数量
    device: torch.device = None,
    generation_kwargs: Optional[Dict] = None,
    val_ques_path: Optional[str] = None,
    val_ann_path: Optional[str] = None,
    train_ques_path: Optional[str] = None,
    train_ann_path: Optional[str] = None,
    strict_eval: bool = True,  # 严格模式：禁用fallback
    save_prompts: bool = False,  # 是否保存prompt文本
    start_idx: int = 0,  # 分片起始索引
    end_idx: int = -1,  # 分片结束索引（-1 表示全部）
    output_path: Optional[str] = None,  # 输出路径（用于断点续传）
    save_interval: int = 50,  # 保存间隔
) -> Dict:
    """
    生成完整的 RL 数据
    
    Args:
        sft_model: SFT 模型
        vqa_model: VQA 模型 interface
        beam_data: 现有的 beam 数据（用于获取 query_id 和候选池信息）
        query_embeddings: [N, d] query embeddings
        candidate_embeddings: [K, d] candidate embeddings
        candidate_indices: candidate 索引列表
        dataset: 数据集对象（用于获取图像、问题、答案等）
        num_beams: beam 数量
        temps: 温度列表
        num_samples_per_temp: 每个温度的采样数量
        num_random: 随机组合数量
        device: 设备
        generation_kwargs: VQA 生成参数（可选）
        val_ques_path: 验证集问题文件路径（可选，用于准确率计算）
        val_ann_path: 验证集标注文件路径（可选，用于准确率计算）
        start_idx: 分片起始索引（用于多 GPU 并行）
        end_idx: 分片结束索引（-1 表示全部）
        output_path: 输出路径（用于断点续传）
        save_interval: 保存间隔
    
    Returns:
        rl_data: 新的 RL 数据格式
    """
    if device is None:
        device = next(sft_model.parameters()).device
    
    # 断点续传：加载已有数据
    rl_data = {}
    completed_queries = set()
    if output_path and os.path.exists(output_path):
        try:
            print(f"发现已有输出文件，加载断点: {output_path}")
            with open(output_path, "r") as f:
                existing_data = json.load(f)
            # 过滤掉 _meta 键
            rl_data = {k: v for k, v in existing_data.items() if not k.startswith("_")}
            completed_queries = set(rl_data.keys())
            print(f"✓ 已加载 {len(completed_queries)} 个已完成的 query")
        except Exception as e:
            print(f"警告：加载断点失败: {e}，将从头开始")
    
    # 构建 candidate 索引映射
    cand_idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}
    
    # 统计变量：用于跟踪文件方式 vs fallback 的使用情况
    total_accuracy_computations = 0
    file_metric_count = 0
    fallback_count = 0
    
    # 优化1: 预加载embeddings到device（避免每个query重复移动）
    print("优化：预加载embeddings到device...")
    if query_embeddings.device != device:
        query_embeddings = query_embeddings.to(device)
    if candidate_embeddings.device != device:
        candidate_embeddings = candidate_embeddings.to(device)
    print("✓ Embeddings已预加载到device")
    
    # 优化2: 使用按需加载的candidate_pool（避免内存爆炸）
    # 对于大数据集（如VQAv2有443k候选），预加载所有候选会导致内存不足
    # 改为按需加载：只在需要时才从dataset中获取
    print(f"优化：使用按需加载的candidate_pool（共{len(candidate_indices)}个候选）...")
    
    class LazyDatasetPool:
        """按需加载的数据集池，避免预加载所有候选导致内存爆炸"""
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self._cache = {}  # 简单的LRU缓存
            self._cache_max_size = 1000  # 最多缓存1000个样本
        
        def __getitem__(self, pos):
            """按position获取候选，pos是在indices中的位置"""
            if pos in self._cache:
                return self._cache[pos]
            
            # 从dataset加载
            global_idx = self.indices[pos]
            item = self.dataset[global_idx]
            
            # 缓存（简单的FIFO策略）
            if len(self._cache) >= self._cache_max_size:
                # 删除最早的一个
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[pos] = item
            
            return item
        
        def __len__(self):
            return len(self.indices)
    
    candidate_pool = LazyDatasetPool(dataset, candidate_indices)
    print(f"✓ Candidate pool已初始化（按需加载模式，共{len(candidate_pool)}个候选）")
    
    # 优化：预先加载 VQA 对象（只加载一次，避免重复加载）
    vqa_train_cache = None
    vqa_val_cache = None
    if train_ques_path and train_ann_path:
        try:
            print("预加载训练集 VQA 标注文件（优化：只加载一次）...")
            vqa_train_cache = VQA(train_ann_path, train_ques_path)
            print("✓ 训练集 VQA 对象已缓存")
        except Exception as e:
            print(f"警告：预加载训练集 VQA 对象失败: {e}")
    
    if val_ques_path and val_ann_path:
        try:
            print("预加载验证集 VQA 标注文件（优化：只加载一次）...")
            vqa_val_cache = VQA(val_ann_path, val_ques_path)
            print("✓ 验证集 VQA 对象已缓存")
        except Exception as e:
            print(f"警告：预加载验证集 VQA 对象失败: {e}")
    
    # 分片处理
    all_query_ids = list(beam_data.keys())
    if end_idx == -1:
        end_idx = len(all_query_ids)
    query_ids_to_process = all_query_ids[start_idx:end_idx]
    
    print(f"开始生成 RL 数据...")
    print(f"  - 总 Query 数: {len(all_query_ids)}")
    print(f"  - 分片范围: {start_idx} - {end_idx}")
    print(f"  - 本次处理: {len(query_ids_to_process)} 个 query")
    print(f"  - 已完成: {len(completed_queries)} 个 query")
    print(f"  - 待处理: {len(query_ids_to_process) - len(completed_queries & set(query_ids_to_process))} 个 query")
    print(f"  - Beam 数量: {num_beams}")
    print(f"  - 温度: {temps}")
    print(f"  - 每个温度采样数: {num_samples_per_temp}")
    print(f"  - 随机组合数: {num_random}")
    print(f"  - Retrieval数量: {num_retrieval}")
    print(f"  - Strict eval: {strict_eval}")
    
    # 严格模式检查：必须提供至少一个官方评测文件
    if strict_eval:
        if (train_ques_path is None or train_ann_path is None) and (val_ques_path is None or val_ann_path is None):
            raise ValueError(
                "strict_eval enabled but no official VQA eval files provided. "
                "Please provide at least one of: (train_ques_path, train_ann_path) or (val_ques_path, val_ann_path)"
            )
    
    processed_count = 0
    for query_id_str in tqdm(query_ids_to_process, desc="生成 RL 数据"):
        # 断点续传：跳过已完成的 query
        if query_id_str in completed_queries:
            continue
        
        query_data = beam_data[query_id_str]
        query_id = int(query_id_str)
        
        # 优化1: embeddings已在device上，直接使用（不需要.to(device)）
        query_emb = query_embeddings[query_id]  # [d]，已在device上
        cand_emb = candidate_embeddings  # [K, d]，已在device上
        
        # 生成 pointer 候选（beam + 温度采样 + 随机组合）
        # 关键修复：需要排除query_id本身，因为ICD不应该包含query
        # 但candidate_embeddings包含了所有样本（包括query），所以需要在生成时排除
        # 注意：这里query_id是数据集索引，需要映射到candidate_indices中的位置
        query_id_in_cand_pos = None
        if query_id in candidate_indices:
            query_id_in_cand_pos = candidate_indices.index(query_id)
        
        pointer_candidates = generate_pointer_candidates_for_query(
            model=sft_model,
            query_emb=query_emb,
            cand_emb=cand_emb,
            num_beams=num_beams,
            temps=temps,
            num_samples_per_temp=num_samples_per_temp,
            num_random=num_random,
            num_retrieval=num_retrieval,  # 使用retrieval方法选择ICD组合
            beam_search_fn=None,  # TODO: 如果已有 beam search 函数，可以传入
            exclude_indices=[query_id_in_cand_pos] if query_id_in_cand_pos is not None else None  # 排除query_id本身
        )
        
        # 获取 query 的原始数据（图像、问题、答案等）
        query_item = dataset[query_id]
        image = query_item.get("image")
        question = query_item.get("question")
        gt_answers_raw = query_item.get("answers", [])
        
        # 处理gt_answers_raw：如果是字典列表，提取answer字段；如果是字符串列表，直接使用
        if gt_answers_raw and isinstance(gt_answers_raw[0], dict):
            gt_answers_raw = [ans.get("answer", "") if isinstance(ans, dict) else str(ans) 
                             for ans in gt_answers_raw]
        elif gt_answers_raw and isinstance(gt_answers_raw[0], str):
            gt_answers_raw = gt_answers_raw
        else:
            gt_answers_raw = [str(ans) for ans in gt_answers_raw] if gt_answers_raw else []
        
        # 归一化ground truth答案（保存到query级别）
        gt_answers_norm = [vqa_postprocess(ans, model_name="qwen2.5_vl_3B") for ans in gt_answers_raw]
        
        # 优化2: candidate_pool已预构建，直接使用
        # candidate_pool已在循环外预构建
        
        # 对每个 pointer 候选计算 correctness
        pointer_candidates_with_correctness = []
        for c in pointer_candidates:
            # 【修复A】pointer是position，不是global id
            pointer_pos = c["pointer"]  # position in candidate_pool (0..K-1)
            pointer_global = [candidate_indices[p] for p in pointer_pos]  # global id
            
            try:
                # 【修复A】使用position作为list下标，不是global id
                ex1 = candidate_pool[pointer_pos[0]]
                ex2 = candidate_pool[pointer_pos[1]]
                
                # 构建 prompt 并生成答案（返回更多信息）
                out = build_vqa_prompt_and_generate(
                    interface=vqa_model,
                    image=image,
                    question=question,
                    ex1=ex1,
                    ex2=ex2,
                    generation_kwargs=generation_kwargs or {}
                )
                
                pred_answer = out["pred_answer"]
                raw_generation = out["raw_generation"]
                prompt_text = out.get("prompt_text") if save_prompts else None
                prompt_len = out.get("prompt_len")
                
                # 计算准确率
                question_id_str = query_item.get("question_id", str(query_id))
                
                # 【改动E】严格模式：禁用fallback
                used_file_metric = False
                eval_failed = False
                eval_split_used = None
                
                # 先尝试训练集文件（如果提供）
                if train_ques_path and train_ann_path:
                    try:
                        correct, acc_score, used_file_metric = compute_vqa_accuracy(
                            pred_answer=pred_answer,
                            ground_truth_answers=gt_answers_raw,
                            question_id=question_id_str,
                            val_ques_path=train_ques_path,
                            val_ann_path=train_ann_path,
                            vqa_cache=vqa_train_cache
                        )
                        if used_file_metric:
                            eval_split_used = "train"
                    except Exception as e:
                        used_file_metric = False
                
                # 如果训练集文件未使用或失败，尝试验证集文件
                if not used_file_metric and val_ques_path and val_ann_path:
                    try:
                        correct, acc_score, used_file_metric = compute_vqa_accuracy(
                            pred_answer=pred_answer,
                            ground_truth_answers=gt_answers_raw,
                            question_id=question_id_str,
                            val_ques_path=val_ques_path,
                            val_ann_path=val_ann_path,
                            vqa_cache=vqa_val_cache
                        )
                        if used_file_metric:
                            eval_split_used = "val"
                    except Exception as e:
                        used_file_metric = False
                
                # 【改动E】严格模式：如果都失败，直接跳过或报错
                if not used_file_metric:
                    if strict_eval:
                        # 严格模式下，跳过这个candidate
                        eval_failed = True
                        correct = 0
                        acc_score = 0.0
                    else:
                        # 非严格模式：使用fallback
                        correct, acc_score, used_file_metric = compute_vqa_accuracy(
                            pred_answer=pred_answer,
                            ground_truth_answers=gt_answers_raw,
                            question_id=question_id_str,
                            val_ques_path=None,
                            val_ann_path=None
                        )
                        eval_split_used = "fallback"
                
                # 【改动E】严格模式下，跳过eval_failed的candidate
                if strict_eval and eval_failed:
                    continue
                
                # 统计使用情况
                total_accuracy_computations += 1
                if used_file_metric:
                    file_metric_count += 1
                else:
                    fallback_count += 1
                
                # P1: 计算 vqa_gt_prob（期望 VQA 准确率概率）
                try:
                    vqa_gt_prob = compute_expected_vqa_acc_prob(
                        interface=vqa_model,
                        image=image,
                        question=question,
                        ground_truth_answers=gt_answers_raw,
                        ex1=ex1,
                        ex2=ex2
                    )
                except Exception as e:
                    print(f"警告：计算 vqa_gt_prob 失败 (query_id={query_id}, pointer={pointer_pos}): {e}")
                    import traceback
                    traceback.print_exc()
                    vqa_gt_prob = 0.0
                
                # 【改动D】计算relevance
                rel = compute_relevance(pred_answer, gt_answers_norm)
                
                # 【改动A】保存pointer信息（pos和global）
                c["pointer_pos"] = pointer_pos
                c["pointer"] = pointer_global
                
                # 【改动B】保存raw generation和prompt
                c["vqa_raw_generation"] = raw_generation
                c["vqa_pred_answer"] = pred_answer
                if save_prompts and prompt_text:
                    c["vqa_prompt_text"] = prompt_text
                    c["vqa_prompt_len"] = prompt_len
                
                # 添加 correctness 信息
                c["vqa_correct"] = correct
                c["vqa_acc_score"] = acc_score
                c["vqa_gt_prob"] = vqa_gt_prob
                
                # 【改动D】添加relevance信息
                c.update(rel)
                
                # 【改动E】添加eval信息
                c["vqa_eval_mode"] = "vqaEval" if used_file_metric else "fallback"
                c["eval_split_used"] = eval_split_used
                c["eval_failed"] = eval_failed
                
            except Exception as e:
                print(f"警告：计算 correctness 失败 (query_id={query_id}, pointer={pointer_pos}): {e}")
                import traceback
                traceback.print_exc()
                # P0: 错误不要默默变成负样本，添加 eval_failed 标记
                # 严格模式下跳过，非严格模式下标记
                if strict_eval:
                    continue  # 严格模式下直接跳过
                
                # 非严格模式：标记为错误
                c["pointer_pos"] = pointer_pos
                c["pointer"] = pointer_global
                c["vqa_raw_generation"] = ""
                c["vqa_pred_answer"] = ""
                c["vqa_correct"] = 0
                c["vqa_acc_score"] = 0.0
                c["vqa_gt_prob"] = 0.0
                c["vqa_rel_token_f1"] = 0.0
                c["vqa_rel_edit_sim"] = 0.0
                c["vqa_rel_score"] = 0.0
                c["vqa_eval_mode"] = "error"
                c["eval_split_used"] = None
                c["eval_failed"] = True
            
            pointer_candidates_with_correctness.append(c)
        
        # 【改动C】保存到 rl_data，包含query级别的gt_answers
        rl_data[query_id_str] = {
            "query": {
                "query_id": query_id,
                "question_id": query_item.get("question_id", str(query_id)),
                "image_id": query_item.get("image_id", None),
                "question": question,
                "gt_answers_raw": gt_answers_raw,
                "gt_answers_norm": gt_answers_norm,
            },
            "pointer_candidates": pointer_candidates_with_correctness
        }
        
        # 定期保存（断点续传）
        processed_count += 1
        if output_path and processed_count % save_interval == 0:
            try:
                with open(output_path, "w") as f:
                    json.dump(rl_data, f, indent=2)
                print(f"  [checkpoint] 已保存 {len(rl_data)} 个 query 到 {output_path}")
            except Exception as e:
                print(f"  [checkpoint] 保存失败: {e}")
    
    # 最终保存
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(rl_data, f, indent=2)
            print(f"  [final] 已保存 {len(rl_data)} 个 query 到 {output_path}")
        except Exception as e:
            print(f"  [final] 保存失败: {e}")
    
    # 打印统计信息
    print(f"\n✓ RL 数据生成完成！")
    print(f"  - 总准确率计算次数: {total_accuracy_computations}")
    if total_accuracy_computations > 0:
        file_metric_ratio = file_metric_count / total_accuracy_computations * 100
        fallback_ratio = fallback_count / total_accuracy_computations * 100
        print(f"  - 使用文件方式计算: {file_metric_count} ({file_metric_ratio:.1f}%)")
        print(f"  - 使用 fallback 字符串匹配: {fallback_count} ({fallback_ratio:.1f}%)")
        if fallback_ratio > 10.0:
            print(f"  ⚠️  警告：fallback 比例较高 ({fallback_ratio:.1f}%)，建议检查 question_id 映射或提供 val_ques_path/val_ann_path")
    
    return rl_data


def main():
    parser = argparse.ArgumentParser(description="生成 RL 数据")
    parser.add_argument("--sft_ckpt", type=str, required=True, help="SFT 模型 checkpoint 路径")
    parser.add_argument("--beam_data", type=str, required=True, help="现有 beam 数据 JSON 路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出 RL 数据 JSON 路径")
    parser.add_argument("--query_emb", type=str, help="Query embedding 路径（.pt 文件）")
    parser.add_argument("--cand_emb", type=str, help="Candidate embedding 路径（.pt 文件）")
    parser.add_argument("--vqa_model", type=str, default="qwen2.5_vl_3B", help="VQA 模型名称")
    parser.add_argument("--dataset", type=str, default="okvqa_local", help="数据集名称")
    parser.add_argument("--config", type=str, help="Hydra 配置文件路径（可选，用于加载数据集和 VQA 模型配置）")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam 数量")
    parser.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.3], help="温度列表")
    parser.add_argument("--num_samples_per_temp", type=int, default=2, help="每个温度的采样数量")
    parser.add_argument("--num_random", type=int, default=1, help="随机组合数量")
    parser.add_argument("--num_retrieval", type=int, default=5, help="Retrieval方法的数量（基于embedding相似度）")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--val_ques_path", type=str, help="验证集问题文件路径（可选，用于准确率计算）")
    parser.add_argument("--val_ann_path", type=str, help="验证集标注文件路径（可选，用于准确率计算）")
    parser.add_argument("--train_ques_path", type=str, help="训练集问题文件路径（可选，用于准确率计算，RL数据通常来自训练集）")
    parser.add_argument("--train_ann_path", type=str, help="训练集标注文件路径（可选，用于准确率计算，RL数据通常来自训练集）")
    parser.add_argument("--train_path", type=str, help="训练集JSON文件路径（用于VQA数据集）")
    parser.add_argument("--val_path", type=str, help="验证集JSON文件路径（用于VQA数据集）")
    parser.add_argument("--train_coco_root", type=str, help="COCO训练集图片根目录")
    parser.add_argument("--val_coco_root", type=str, help="COCO验证集图片根目录")
    parser.add_argument("--strict_eval", action="store_true", default=True, help="严格模式：禁用fallback，必须使用官方VQA评测文件")
    parser.add_argument("--no_strict_eval", dest="strict_eval", action="store_false", help="禁用严格模式（允许fallback）")
    parser.add_argument("--save_prompts", action="store_true", default=False, help="是否保存prompt文本（会增加文件大小）")
    # 分片参数（用于多 GPU 并行）
    parser.add_argument("--start_idx", type=int, default=0, help="起始 query 索引（用于分片）")
    parser.add_argument("--end_idx", type=int, default=-1, help="结束 query 索引（-1 表示全部，用于分片）")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置（如果提供）
    cfg = None
    if args.config:
        if not os.path.exists(args.config):
            print(f"警告：配置文件不存在: {args.config}")
            print("将使用默认配置，请通过命令行参数指定数据集路径")
            args.config = None  # 清除不存在的配置文件路径
        else:
            cfg = OmegaConf.load(args.config)
            # 确保 infer_model.name 格式正确
            if "infer_model" in cfg and "name" in cfg.infer_model:
                model_name_lower = cfg.infer_model.name.lower()
                if "qwen" in model_name_lower and "Qwen2.5-VL" not in cfg.infer_model.name:
                    cfg.infer_model.name = "Qwen2.5-VL"
            # 确保 task.template 不是 None（如果是 None，设置为字符串模板，使用与v0/v1/v2一致的格式）
            if "task" in cfg and cfg.task.get("template") is None:
                cfg.task.template = "Question:<Q> Short answer:<A>"  # 与v0/v1/v2一致（不是Question: {question} Short answer: {answer}）
            # 如果 template 是空字典，也设置为字符串模板（使用与v0/v1/v2一致的格式）
            if "task" in cfg and isinstance(cfg.task.get("template"), dict) and len(cfg.task.template) == 0:
                cfg.task.template = "Question:<Q> Short answer:<A>"  # 与v0/v1/v2一致（不是Question: {question} Short answer: {answer}）
                # 确保 column_token_map 存在（使用与v0/v1/v2一致的格式）
                if "column_token_map" not in cfg.task or not cfg.task.column_token_map:
                    cfg.task.column_token_map = {
                        "question": "<Q>",    # 与v0/v1/v2一致（不是<question>）
                        "answer": "<A>"       # 与v0/v1/v2一致（不是<answer>）
                    }
    else:
        # 映射模型名称到 init_interface 期望的格式
        model_name_lower = args.vqa_model.lower()
        if "qwen" in model_name_lower:
            infer_model_name = "Qwen2.5-VL"
            hf_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif "flamingo" in model_name_lower:
            infer_model_name = "flamingo"
            hf_model_name = None
        elif "idefics" in model_name_lower:
            infer_model_name = "idefics"
            hf_model_name = None
        else:
            infer_model_name = args.vqa_model
            hf_model_name = None
        
        # 创建默认配置
        # 根据数据集名称推断任务类型和配置
        dataset_name = args.dataset.lower()
        if "okvqa" in dataset_name or "vqa" in dataset_name:
            task_name = "vqa"
            # 对于 VQA 任务，需要提供数据集路径配置
            # 优先使用命令行参数，其次使用环境变量，最后使用默认路径
            import os
            # 获取项目根目录（假设脚本在 lever_lm/models/v3/ 目录下）
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            # 根据数据集名称选择正确的路径
            if "vqav2" in dataset_name:
                # VQAv2 数据集路径
                # 根据 configs/dataset/vqav2_local.yaml 中的配置
                vqav2_dir = os.path.join(project_root, "datasets", "vqav2")
                vqav2_hf_dir = os.path.join(vqav2_dir, "vqav2_hf")
                default_train_path = os.path.join(vqav2_hf_dir, "vqav2_mscoco_train2014.json")
                default_val_path = os.path.join(vqav2_hf_dir, "vqav2_mscoco_val2014.json")
                default_train_coco_root = os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "train2014")
                default_val_coco_root = os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "val2014")
                
                train_path = args.train_path or os.getenv("VQAV2_TRAIN_PATH", default_train_path)
                val_path = args.val_path or os.getenv("VQAV2_VAL_PATH", default_val_path)
                train_coco_root = args.train_coco_root or os.getenv("COCO_TRAIN_ROOT", default_train_coco_root)
                val_coco_root = args.val_coco_root or os.getenv("COCO_VAL_ROOT", default_val_coco_root)
            else:
                # OKVQA 数据集路径（默认）
                # 根据 configs/dataset/okvqa_local.yaml 中的配置
                okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
                okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
                default_train_path = os.path.join(okvqa_hf_dir, "vqav2_mscoco_train2014.json")
                default_val_path = os.path.join(okvqa_hf_dir, "vqav2_mscoco_val2014.json")
                default_train_coco_root = os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "train2014")
                default_val_coco_root = os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "val2014")
                
                train_path = args.train_path or os.getenv("OKVQA_TRAIN_PATH", default_train_path)
                val_path = args.val_path or os.getenv("OKVQA_VAL_PATH", default_val_path)
                train_coco_root = args.train_coco_root or os.getenv("COCO_TRAIN_ROOT", default_train_coco_root)
                val_coco_root = args.val_coco_root or os.getenv("COCO_VAL_ROOT", default_val_coco_root)
            
            # 如果默认路径不存在，尝试查找其他可能的文件名和位置
            if not os.path.exists(train_path) and not args.train_path:
                if "vqav2" in dataset_name:
                    # VQAv2 数据集的可能路径
                    vqav2_dir = os.path.join(project_root, "datasets", "vqav2")
                    vqav2_hf_dir = os.path.join(vqav2_dir, "vqav2_hf")
                    possible_train_files = [
                        os.path.join(vqav2_hf_dir, "vqav2_mscoco_train2014.json"),
                        os.path.join(vqav2_dir, "vqav2_hf", "vqav2_mscoco_train2014.json"),
                        os.path.join(vqav2_dir, "train.json"),
                    ]
                else:
                    # OKVQA 数据集的可能路径
                    okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
                    okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
                    possible_train_files = [
                        os.path.join(okvqa_hf_dir, "vqav2_mscoco_train2014.json"),
                        os.path.join(okvqa_dir, "okvqa_hf", "vqav2_mscoco_train2014.json"),
                        os.path.join(okvqa_dir, "train.json"),
                        os.path.join(okvqa_dir, "train_annotations.json"),
                        os.path.join(okvqa_dir, "mscoco_train2014_annotations.json"),
                        os.path.join(okvqa_dir, "OpenEnded_mscoco_train2014_questions.json"),
                    ]
                for possible_path in possible_train_files:
                    if os.path.exists(possible_path):
                        train_path = possible_path
                        print(f"找到训练集文件: {train_path}")
                        break
            
            if not os.path.exists(val_path) and not args.val_path:
                if "vqav2" in dataset_name:
                    # VQAv2 数据集的可能路径
                    vqav2_dir = os.path.join(project_root, "datasets", "vqav2")
                    vqav2_hf_dir = os.path.join(vqav2_dir, "vqav2_hf")
                    possible_val_files = [
                        os.path.join(vqav2_hf_dir, "vqav2_mscoco_val2014.json"),
                        os.path.join(vqav2_dir, "vqav2_hf", "vqav2_mscoco_val2014.json"),
                        os.path.join(vqav2_dir, "val.json"),
                    ]
                else:
                    # OKVQA 数据集的可能路径
                    okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
                    okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
                    possible_val_files = [
                        os.path.join(okvqa_hf_dir, "vqav2_mscoco_val2014.json"),
                        os.path.join(okvqa_dir, "okvqa_hf", "vqav2_mscoco_val2014.json"),
                        os.path.join(okvqa_dir, "val.json"),
                        os.path.join(okvqa_dir, "val_annotations.json"),
                        os.path.join(okvqa_dir, "mscoco_val2014_annotations.json"),
                        os.path.join(okvqa_dir, "OpenEnded_mscoco_val2014_questions.json"),
                    ]
                for possible_path in possible_val_files:
                    if os.path.exists(possible_path):
                        val_path = possible_path
                        print(f"找到验证集文件: {val_path}")
                        break
            
            # 检查路径是否存在，如果不存在则给出提示
            if not os.path.exists(train_path):
                ds_name = "vqav2" if "vqav2" in dataset_name else "okvqa"
                env_var = "VQAV2_TRAIN_PATH" if "vqav2" in dataset_name else "OKVQA_TRAIN_PATH"
                config_file = f"configs/dataset/{ds_name}_local.yaml"
                print(f"错误：训练集文件不存在: {train_path}")
                print(f"请使用 --train_path 参数指定正确的路径，或设置环境变量 {env_var}")
                print(f"根据配置文件 {config_file}，预期路径为：")
                print(f"  {default_train_path}")
                print(f"\n请检查以下目录下的文件：")
                # 检查多个可能的目录
                if "vqav2" in dataset_name:
                    vqav2_dir = os.path.join(project_root, "datasets", "vqav2")
                    vqav2_hf_dir = os.path.join(vqav2_dir, "vqav2_hf")
                    check_dirs = [vqav2_hf_dir, vqav2_dir]
                else:
                    okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
                    okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
                    check_dirs = [okvqa_hf_dir, okvqa_dir]
                for check_dir in check_dirs:
                    if os.path.exists(check_dir):
                        print(f"\n{check_dir} 目录下的文件：")
                        try:
                            json_files = [f for f in os.listdir(check_dir) if f.endswith('.json')]
                            if json_files:
                                for f in json_files:
                                    print(f"  - {f}")
                            else:
                                print("  (无JSON文件)")
                        except Exception as e:
                            print(f"  (无法读取目录: {e})")
                return
            if not os.path.exists(val_path):
                ds_name = "vqav2" if "vqav2" in dataset_name else "okvqa"
                env_var = "VQAV2_VAL_PATH" if "vqav2" in dataset_name else "OKVQA_VAL_PATH"
                config_file = f"configs/dataset/{ds_name}_local.yaml"
                print(f"错误：验证集文件不存在: {val_path}")
                print(f"请使用 --val_path 参数指定正确的路径，或设置环境变量 {env_var}")
                print(f"根据配置文件 {config_file}，预期路径为：")
                print(f"  {default_val_path}")
                print(f"\n请检查以下目录下的文件：")
                # 检查多个可能的目录
                if "vqav2" in dataset_name:
                    vqav2_dir = os.path.join(project_root, "datasets", "vqav2")
                    vqav2_hf_dir = os.path.join(vqav2_dir, "vqav2_hf")
                    check_dirs = [vqav2_hf_dir, vqav2_dir]
                else:
                    okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
                    okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
                    check_dirs = [okvqa_hf_dir, okvqa_dir]
                for check_dir in check_dirs:
                    if os.path.exists(check_dir):
                        print(f"\n{check_dir} 目录下的文件：")
                        try:
                            json_files = [f for f in os.listdir(check_dir) if f.endswith('.json')]
                            if json_files:
                                for f in json_files:
                                    print(f"  - {f}")
                            else:
                                print("  (无JSON文件)")
                        except Exception as e:
                            print(f"  (无法读取目录: {e})")
                return
            
            cfg = OmegaConf.create({
                "dataset": {
                    "name": args.dataset,
                    "version": "local",
                    "train_path": train_path,
                    "val_path": val_path,
                    "train_coco_dataset_root": train_coco_root,
                    "val_coco_dataset_root": val_coco_root,
                },
                "infer_model": {
                    "name": infer_model_name,
                    "model_name": hf_model_name,
                    "load_from_local": False,
                    "precision": "bf16",
                    # 【关键】从config文件加载，确保与v0/v1/v2一致
                    # configs/infer_model/qwen2.5_vl_3B.yaml: icd_join_char: "<|endofchunk|>"
                    "icd_join_char": "<|endofchunk|>",  # 与v0/v1/v2一致
                    # 【关键】从config文件加载system_prompt，确保与v0/v1/v2一致
                    "system_prompt": "In the upcoming task, you will see four sets of dialogues, each containing two roles: user and assistant. The user is the questioner, who provides an image and asks a question based on it; the assistant is the responder, who answers according to the image and question provided by the user. Afterward, you will receive an image and a question from the user. Please act as the assistant and answer based on the four previous dialogue sets and your own knowledge. Strictly follow the answering format: if the examples use only one or two keywords, your reply must also use only one or two keywords; if the examples contain no more than three tokens, your reply must not exceed three tokens either.",
                },
                "task": {
                    "task_name": task_name,
                    # 【关键】使用与v0/v1/v2完全相同的template格式
                    # configs/infer_model/qwen2.5_vl_3B.yaml: vqa_prompt_template: "Question:<Q> Short answer:<A>"
                    "template": "Question:<Q> Short answer:<A>",  # 与v0/v1/v2一致
                    # 【关键】使用与v0/v1/v2完全相同的column_token_map
                    # configs/task/vqa.yaml: column_token_map: {question: "<Q>", answer: "<A>"}
                    "column_token_map": {
                        "question": "<Q>",    # 与v0/v1/v2一致（不是<question>）
                        "answer": "<A>"       # 与v0/v1/v2一致（不是<answer>）
                    },
                    "instruction": "",
                    "image_field": "image",
                    "output_column": "answer",
                },
                "precision": "bf16",
            })
        else:
            # 对于其他任务（如 caption），使用简单配置
            cfg = OmegaConf.create({
                "dataset": {
                    "name": args.dataset,
                },
                "infer_model": {
                    "name": infer_model_name,
                    "model_name": hf_model_name,
                    "load_from_local": False,
                    "precision": "bf16",
                    "icd_join_char": " ",
                },
                "task": {
                    "task_name": "caption",  # 默认
                    "template": {},
                    "column_token_map": {},
                    "instruction": "",
                    "image_field": "image",
                    "output_column": "answer",
                },
                "precision": "bf16",
            })
    
    # 加载 SFT 模型
    print(f"加载 SFT 模型: {args.sft_ckpt}")
    sft_model = load_sft_model(args.sft_ckpt, device)
    
    # 加载 VQA 模型
    print(f"加载 VQA 模型: {args.vqa_model}")
    vqa_model = load_vqa_model(args.vqa_model, device, cfg=cfg)
    
    # 加载 beam 数据
    print(f"加载 beam 数据: {args.beam_data}")
    with open(args.beam_data, "r") as f:
        beam_data = json.load(f)
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    ds = load_ds(cfg)
    train_ds = ds["train"]
    val_ds = ds["validation"]
    
    # 合并训练集和验证集，用于获取 candidate pool
    # 注意：这里假设 candidate_indices 指向训练集
    dataset = train_ds
    
    # 加载 embeddings
    if args.query_emb and args.cand_emb:
        print(f"加载 query embeddings: {args.query_emb}")
        # 优化：加载时直接放到device上，避免后续重复移动
        query_embeddings = torch.load(args.query_emb, map_location=device)
        
        print(f"加载 candidate embeddings: {args.cand_emb}")
        # 优化：加载时直接放到device上，避免后续重复移动
        candidate_embeddings = torch.load(args.cand_emb, map_location=device)
        
        # 假设 candidate_indices 是连续的索引
        candidate_indices = list(range(len(candidate_embeddings)))
    else:
        print("警告：未提供 embedding 路径，将无法生成 pointer 候选")
        print("请使用 --query_emb 和 --cand_emb 参数提供 embedding 路径")
        return
    
    # 生成 RL 数据
    print("开始生成 RL 数据...")
    # 【重要】从config文件加载生成参数，确保与推理完全一致
    # 参考 configs/task/vqa.yaml 中的 gen_args，与 icl_inference.py 中的使用方式一致
    task_config_path = "configs/task/vqa.yaml"
    if os.path.exists(task_config_path):
        task_cfg = OmegaConf.load(task_config_path)
        # 从config文件加载gen_args，确保与推理完全一致
        if hasattr(task_cfg, 'gen_args') and task_cfg.gen_args:
            default_generation_kwargs = OmegaConf.to_container(task_cfg.gen_args)
            print(f"✓ 从 {task_config_path} 加载生成参数:")
            for k, v in default_generation_kwargs.items():
                print(f"  {k}: {v}")
        else:
            # Fallback：如果config中没有gen_args，使用默认值（与configs/task/vqa.yaml一致）
            print(f"⚠️  警告：{task_config_path} 中没有 gen_args，使用默认值")
            default_generation_kwargs = {
                "max_new_tokens": 5,
                "num_beams": 3,
                "length_penalty": 0.0,
                "min_new_tokens": 0,
            }
    else:
        raise FileNotFoundError(f"配置文件不存在: {task_config_path}")
    
    # 确保do_sample参数正确（beam search时应该为False）
    if default_generation_kwargs.get("num_beams", 1) > 1:
        default_generation_kwargs["do_sample"] = False  # beam search时必须是deterministic
    else:
        default_generation_kwargs.setdefault("do_sample", False)
    rl_data = generate_rl_data(
        sft_model=sft_model,
        vqa_model=vqa_model,
        beam_data=beam_data,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        candidate_indices=candidate_indices,
        dataset=dataset,
        num_beams=args.num_beams,
        temps=tuple(args.temps),
        num_samples_per_temp=args.num_samples_per_temp,
        num_random=args.num_random,
        num_retrieval=args.num_retrieval,
        device=device,
        generation_kwargs=default_generation_kwargs,
        val_ques_path=args.val_ques_path,
        val_ann_path=args.val_ann_path,
        train_ques_path=args.train_ques_path,
        train_ann_path=args.train_ann_path,
        strict_eval=args.strict_eval,
        save_prompts=args.save_prompts,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        output_path=args.output_path,
        save_interval=50,  # 每 50 个 query 保存一次
    )
    
    # 【改动F】添加_meta信息
    # 获取VQA模型名称
    vqa_model_name = args.vqa_model
    if hasattr(vqa_model, 'model_name'):
        vqa_model_name = vqa_model.model_name
    elif "Qwen" in str(type(vqa_model)):
        vqa_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # 构建_meta信息
    meta_info = {
        "_meta": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vqa_model": vqa_model_name,
            "task_gen_args": default_generation_kwargs,
            "eval": {
                "train_ques_path": args.train_ques_path,
                "train_ann_path": args.train_ann_path,
                "val_ques_path": args.val_ques_path,
                "val_ann_path": args.val_ann_path,
            },
            "strict_eval": args.strict_eval,
            "save_prompts": args.save_prompts,
            "notes": "RL data v4: save raw response + prompt + gt answers + relevance. pointer_pos vs pointer(global) fixed."
        }
    }
    
    # 合并_meta和rl_data
    output_data = {**meta_info, **rl_data}
    
    # 保存
    print(f"保存 RL 数据到: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("✓ RL 数据生成完成！")
    print(f"  - 总query数: {len(rl_data)}")
    print(f"  - 严格模式: {args.strict_eval}")
    print(f"  - 保存prompt: {args.save_prompts}")


if __name__ == "__main__":
    main()
