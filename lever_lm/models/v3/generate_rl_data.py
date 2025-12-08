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
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from lever_lm.models.v3.rl_data_generation import (
    generate_pointer_candidates_for_query,
    evaluate_pointer_candidate
)
from lever_lm.models.v3 import PointerSelectorV3
from lever_lm.utils import init_interface
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy as compute_vqa_accuracy_metric

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
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def load_vqa_model(model_name: str, device: torch.device, cfg: Optional[DictConfig] = None):
    """
    加载 VQA 模型
    
    Args:
        model_name: 模型名称（如 "qwen2.5_vl_3B" 或 "flamingo_3B"）
        device: 设备
        cfg: 配置对象（可选，如果提供则使用，否则创建默认配置）
    
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
        # 创建默认配置
        cfg = OmegaConf.create({
            "infer_model": {
                "name": infer_model_name,  # 使用映射后的名称
                "model_name": hf_model_name,
                "load_from_local": False,
                "precision": "bf16",
                "icd_join_char": " ",
            },
            "task": {
                "template": "Question: {question} Short answer: {answer}",  # VQA 任务的字符串模板
                "column_token_map": {
                    "question": "<question>",
                    "answer": "<answer>"
                },
                "instruction": "",
                "image_field": "image",
                "output_column": "answer",
            },
            "precision": "bf16",
        })
    
    # 使用 init_interface 加载模型
    interface = init_interface(cfg, device=device)
    return interface


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
    model_name = interface.__class__.__name__.lower()
    if "qwen" in model_name:
        answer = vqa_postprocess(prediction, model_name="qwen2.5_vl_3B")
    elif "flamingo" in model_name:
        answer = vqa_postprocess(prediction, model_name="flamingo_3B")
    else:
        answer = prediction.strip()
    
    return answer


def compute_vqa_accuracy(
    pred_answer: str,
    ground_truth_answers,
    question_id: Optional[str] = None,
    val_ques_path: Optional[str] = None,
    val_ann_path: Optional[str] = None
) -> Tuple[int, float]:
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
    """
    # 处理 ground_truth_answers：如果是字典列表，提取 answer 字段
    if ground_truth_answers and isinstance(ground_truth_answers[0], dict):
        # 字典格式：提取 "answer" 字段
        gt_answers_str = [ans.get("answer", "") if isinstance(ans, dict) else str(ans) 
                         for ans in ground_truth_answers]
    elif ground_truth_answers and isinstance(ground_truth_answers[0], str):
        # 字符串格式：直接使用
        gt_answers_str = ground_truth_answers
    else:
        # 空列表或其他格式：转换为字符串列表
        gt_answers_str = [str(ans) for ans in ground_truth_answers] if ground_truth_answers else []
    
    # 如果提供了文件路径，使用文件方式计算（更准确）
    if val_ques_path and val_ann_path and question_id:
        try:
            # 创建临时结果文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
                
                return correct, acc_score
            finally:
                # 清理临时文件
                if os.path.exists(temp_result_file):
                    os.remove(temp_result_file)
        except Exception as e:
            print(f"警告：使用文件方式计算准确率失败，回退到简单匹配: {e}")
    
    # 简单匹配方式：检查预测答案是否在标准答案列表中（不区分大小写）
    pred_answer_lower = pred_answer.lower().strip()
    gt_answers_lower = [ans.lower().strip() for ans in gt_answers_str]
    
    # 精确匹配
    if pred_answer_lower in gt_answers_lower:
        return 1, 1.0
    
    # 部分匹配（检查预测答案是否包含标准答案，或标准答案是否包含预测答案）
    for gt_ans in gt_answers_lower:
        if pred_answer_lower in gt_ans or gt_ans in pred_answer_lower:
            # 部分匹配，给予较低的分数
            return 1, 0.5
    
    # 不匹配
    return 0, 0.0


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
    device: torch.device = None,
    generation_kwargs: Optional[Dict] = None,
    val_ques_path: Optional[str] = None,
    val_ann_path: Optional[str] = None
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
    
    Returns:
        rl_data: 新的 RL 数据格式
    """
    if device is None:
        device = next(sft_model.parameters()).device
    
    rl_data = {}
    
    # 构建 candidate 索引映射
    cand_idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}
    
    print(f"开始生成 RL 数据...")
    print(f"  - Query 数量: {len(beam_data)}")
    print(f"  - Beam 数量: {num_beams}")
    print(f"  - 温度: {temps}")
    print(f"  - 每个温度采样数: {num_samples_per_temp}")
    print(f"  - 随机组合数: {num_random}")
    
    for query_id_str, query_data in tqdm(beam_data.items(), desc="生成 RL 数据"):
        query_id = int(query_id_str)
        
        # 获取 query embedding
        query_emb = query_embeddings[query_id].to(device)  # [d]
        
        # 获取 candidate embeddings
        cand_emb = candidate_embeddings.to(device)  # [K, d]
        
        # 生成 pointer 候选（beam + 温度采样 + 随机组合）
        pointer_candidates = generate_pointer_candidates_for_query(
            model=sft_model,
            query_emb=query_emb,
            cand_emb=cand_emb,
            num_beams=num_beams,
            temps=temps,
            num_samples_per_temp=num_samples_per_temp,
            num_random=num_random,
            beam_search_fn=None  # TODO: 如果已有 beam search 函数，可以传入
        )
        
        # 获取 query 的原始数据（图像、问题、答案等）
        query_item = dataset[query_id]
        image = query_item.get("image")
        question = query_item.get("question")
        gt_answers = query_item.get("answers", [])
        
        # 构建 candidate_pool（从 dataset 中获取）
        candidate_pool = []
        for idx in candidate_indices:
            candidate_pool.append(dataset[idx])
        
        # 对每个 pointer 候选计算 correctness
        pointer_candidates_with_correctness = []
        for c in pointer_candidates:
            pointer = c["pointer"]
            
            # 将 pointer 中的索引映射回原始 candidate 索引
            original_pointer = [candidate_indices[p] for p in pointer]
            
            try:
                # 获取示例数据
                ex1 = candidate_pool[original_pointer[0]]
                ex2 = candidate_pool[original_pointer[1]]
                
                # 构建 prompt 并生成答案
                pred_answer = build_vqa_prompt_and_generate(
                    interface=vqa_model,
                    image=image,
                    question=question,
                    ex1=ex1,
                    ex2=ex2,
                    generation_kwargs=generation_kwargs or {}
                )
                
                # 计算准确率
                question_id_str = query_item.get("question_id", str(query_id))
                correct, acc_score = compute_vqa_accuracy(
                    pred_answer=pred_answer,
                    ground_truth_answers=gt_answers,
                    question_id=question_id_str,
                    val_ques_path=val_ques_path,
                    val_ann_path=val_ann_path
                )
                
                # 添加 correctness 信息
                c["vqa_pred_answer"] = pred_answer
                c["vqa_correct"] = correct
                c["vqa_acc_score"] = acc_score
                
            except Exception as e:
                print(f"警告：计算 correctness 失败 (query_id={query_id}, pointer={pointer}): {e}")
                import traceback
                traceback.print_exc()
                # 如果失败，设置默认值
                c["vqa_pred_answer"] = ""
                c["vqa_correct"] = 0
                c["vqa_acc_score"] = 0.0
            
            pointer_candidates_with_correctness.append(c)
        
        # 保存到 rl_data
        rl_data[query_id_str] = {
            "pointer_candidates": pointer_candidates_with_correctness
        }
    
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
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--val_ques_path", type=str, help="验证集问题文件路径（可选，用于准确率计算）")
    parser.add_argument("--val_ann_path", type=str, help="验证集标注文件路径（可选，用于准确率计算）")
    parser.add_argument("--train_path", type=str, help="训练集JSON文件路径（用于VQA数据集）")
    parser.add_argument("--val_path", type=str, help="验证集JSON文件路径（用于VQA数据集）")
    parser.add_argument("--train_coco_root", type=str, help="COCO训练集图片根目录")
    parser.add_argument("--val_coco_root", type=str, help="COCO验证集图片根目录")
    
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
            # 确保 task.template 不是 None（如果是 None，设置为字符串模板）
            if "task" in cfg and cfg.task.get("template") is None:
                cfg.task.template = "Question: {question} Short answer: {answer}"
            # 如果 template 是空字典，也设置为字符串模板
            if "task" in cfg and isinstance(cfg.task.get("template"), dict) and len(cfg.task.template) == 0:
                cfg.task.template = "Question: {question} Short answer: {answer}"
                # 确保 column_token_map 存在
                if "column_token_map" not in cfg.task or not cfg.task.column_token_map:
                    cfg.task.column_token_map = {
                        "question": "<question>",
                        "answer": "<answer>"
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
            
            # 设置默认路径（基于项目根目录，使用实际的数据集路径）
            # 根据 configs/dataset/okvqa_local.yaml 中的配置：
            # train_path: ${oc.env:OKVQA_PATH}/okvqa_hf/vqav2_mscoco_train2014.json
            # val_path: ${oc.env:OKVQA_PATH}/okvqa_hf/vqav2_mscoco_val2014.json
            okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
            okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
            # 使用与训练时相同的文件名
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
                # 尝试查找常见的文件名（包括不同目录）
                possible_train_files = [
                    # 优先使用与配置文件一致的路径
                    os.path.join(okvqa_hf_dir, "vqav2_mscoco_train2014.json"),
                    os.path.join(okvqa_dir, "okvqa_hf", "vqav2_mscoco_train2014.json"),
                    # 其他可能的文件名
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
                # 尝试查找常见的文件名（包括不同目录）
                possible_val_files = [
                    # 优先使用与配置文件一致的路径
                    os.path.join(okvqa_hf_dir, "vqav2_mscoco_val2014.json"),
                    os.path.join(okvqa_dir, "okvqa_hf", "vqav2_mscoco_val2014.json"),
                    # 其他可能的文件名
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
                print(f"错误：训练集文件不存在: {train_path}")
                print(f"请使用 --train_path 参数指定正确的路径，或设置环境变量 OKVQA_TRAIN_PATH")
                print(f"根据配置文件 configs/dataset/okvqa_local.yaml，预期路径为：")
                print(f"  {default_train_path}")
                print(f"\n请检查以下目录下的文件：")
                # 检查多个可能的目录
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
                print(f"错误：验证集文件不存在: {val_path}")
                print(f"请使用 --val_path 参数指定正确的路径，或设置环境变量 OKVQA_VAL_PATH")
                print(f"根据配置文件 configs/dataset/okvqa_local.yaml，预期路径为：")
                print(f"  {default_val_path}")
                print(f"\n请检查以下目录下的文件：")
                # 检查多个可能的目录
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
                    "icd_join_char": " ",
                },
                "task": {
                    "task_name": task_name,
                    "template": "Question: {question} Short answer: {answer}",  # VQA 任务的字符串模板
                    "column_token_map": {
                        "question": "<question>",
                        "answer": "<answer>"
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
        query_embeddings = torch.load(args.query_emb, map_location=device)
        
        print(f"加载 candidate embeddings: {args.cand_emb}")
        candidate_embeddings = torch.load(args.cand_emb, map_location=device)
        
        # 假设 candidate_indices 是连续的索引
        candidate_indices = list(range(len(candidate_embeddings)))
    else:
        print("警告：未提供 embedding 路径，将无法生成 pointer 候选")
        print("请使用 --query_emb 和 --cand_emb 参数提供 embedding 路径")
        return
    
    # 生成 RL 数据
    print("开始生成 RL 数据...")
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
        device=device,
        generation_kwargs={},
        val_ques_path=args.val_ques_path,
        val_ann_path=args.val_ann_path
    )
    
    # 保存
    print(f"保存 RL 数据到: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(rl_data, f, indent=2)
    
    print("✓ RL 数据生成完成！")


if __name__ == "__main__":
    main()
