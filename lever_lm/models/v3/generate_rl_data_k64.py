"""
RL 数据生成脚本（K=64版本）：确保每个query使用固定的64个候选池

关键修复：
1. 加载sampler缓存，获取每个query的64个候选池索引
2. 为每个query构建独立的候选池embedding [64, d]
3. 生成的pointer使用局部索引 [0, 63]
4. 保存候选池信息到输出数据中

使用方法：
    python -m lever_lm.models.v3.generate_rl_data_k64 \
        --sft_ckpt <path_to_checkpoint> \
        --sampler_cache <sampler_cache.json> \
        --output_path <output_rl_data.json> \
        --vqa_model qwen2.5_vl_3B \
        --dataset okvqa_local \
        --num_beams 5 \
        --temps 1.0 1.3 \
        --num_samples_per_temp 2 \
        --num_random 2

作者: Lever-Plus Team
日期: 2025-12-20
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

# 临时文件目录
TEMP_DIR = "/mnt/share/yiyun/Projects/Lever-Plus/tmp" if os.path.exists("/mnt/share") else None
if TEMP_DIR:
    os.makedirs(TEMP_DIR, exist_ok=True)

from lever_lm.models.v3.rl_data_generation import (
    generate_pointer_candidates_for_query,
)
from lever_lm.models.v3 import PointerSelectorV3
from lever_lm.utils import init_interface
from open_mmicl.metrics.vqa_metrics import VQA, VQAEval

# 导入根目录的utils.py
import importlib.util
import os as _os
_utils_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
vqa_postprocess = _root_utils.vqa_postprocess
load_ds = _root_utils.load_ds


def load_sft_model(checkpoint_path: str, device: torch.device, d_model: int = 512) -> PointerSelectorV3:
    """加载 SFT 模型"""
    model = PointerSelectorV3(
        d_model=d_model,
        K=64,
        shot_num=2
    )
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        state_dict = {k.replace('lever_lm.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if not missing and not unexpected:
        print("✓ Checkpoint 参数完全匹配")
    else:
        if missing:
            print(f"⚠️ 缺失参数: {len(missing)}")
        if unexpected:
            print(f"⚠️ 多余参数: {len(unexpected)}")
    
    model.to(device)
    model.eval()
    return model


def load_vqa_model(model_name: str, device: torch.device, cfg: Optional[DictConfig] = None):
    """加载 VQA 模型"""
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        infer_model_name = "Qwen2.5-VL"
        hf_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    else:
        infer_model_name = model_name
        hf_model_name = None
    
    if cfg is None:
        infer_model_config_path = "configs/infer_model/qwen2.5_vl_3B.yaml"
        task_config_path = "configs/task/vqa.yaml"
        
        infer_model_cfg = OmegaConf.load(infer_model_config_path)
        task_cfg = OmegaConf.load(task_config_path)
        
        cfg = OmegaConf.create({
            "infer_model": {
                "name": infer_model_name,
                "model_name": hf_model_name,
                "load_from_local": False,
                "precision": "bf16",
                "icd_join_char": infer_model_cfg.get("icd_join_char", "<|endofchunk|>"),
                "system_prompt": infer_model_cfg.get("system_prompt", ""),
            },
            "task": {
                "template": infer_model_cfg.get("vqa_prompt_template", "Question:<Q> Short answer:<A>"),
                "column_token_map": OmegaConf.to_container(task_cfg.get("column_token_map", {"question": "<Q>", "answer": "<A>"})),
                "instruction": infer_model_cfg.get("vqa_instruction", ""),
                "image_field": "image",
                "output_column": "answer",
            },
            "precision": "bf16",
        })
    
    interface = init_interface(cfg, device=device)
    return interface


def build_vqa_prompt_and_generate(
    interface,
    image,
    question: str,
    ex1: Dict,
    ex2: Dict,
    generation_kwargs: Optional[Dict] = None
) -> Dict:
    """构建 VQA prompt 并生成答案"""
    if generation_kwargs is None:
        generation_kwargs = {}
    
    query_sample = {"image": image, "question": question}
    data_sample_list = [ex1, ex2, query_sample]
    
    prompts = interface.transfer_prompts([data_sample_list], is_last_for_generation=True)
    input_dict = interface.prepare_input(prompts, is_last_for_generation=True)
    
    if hasattr(input_dict, 'data'):
        input_dict = dict(input_dict.data)
    elif not isinstance(input_dict, dict):
        input_dict = dict(input_dict)
    
    data = {k: v.to(interface.device) if isinstance(v, torch.Tensor) else v 
           for k, v in input_dict.items()}
    
    # 处理 Qwen2.5-VL 的特殊情况
    if 'image_grid_thw' in data:
        if data['image_grid_thw'].dim() == 3 and data['image_grid_thw'].shape[0] == 1:
            data['image_grid_thw'] = data['image_grid_thw'].squeeze(0)
        if 'image_nums' not in data and data['image_grid_thw'].dim() == 2:
            num_images = data['image_grid_thw'].shape[0]
            data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
    
    prompt_len = int(data["attention_mask"].shape[1])
    
    with torch.inference_mode():
        outputs = interface.generate(
            **data,
            eos_token_id=interface.tokenizer.eos_token_id,
            pad_token_id=interface.tokenizer.pad_token_id,
            **generation_kwargs,
        )
    
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.tolist()
    if not isinstance(outputs, list):
        outputs = [outputs]
    if len(outputs) > 0 and not isinstance(outputs[0], list):
        outputs = [outputs]
    
    generated = interface.tokenizer.batch_decode(
        [output[prompt_len:] for output in outputs],
        skip_special_tokens=True,
    )
    
    prediction = generated[0] if generated else ""
    if prediction is None or not isinstance(prediction, str):
        prediction = ""
    
    model_name = interface.__class__.__name__.lower()
    try:
        if "qwen" in model_name:
            answer = vqa_postprocess(prediction, model_name="qwen2.5_vl_3B")
        else:
            answer = prediction.strip()
        if answer is None:
            answer = ""
    except Exception:
        answer = prediction.strip() if prediction else ""
    
    return {
        "pred_answer": answer,
        "raw_generation": prediction,
        "prompt_len": prompt_len,
    }


def compute_vqa_accuracy(
    pred_answer: str,
    ground_truth_answers: List,
    question_id: Optional[str] = None,
    vqa_cache: Optional[VQA] = None,
    ques_path: Optional[str] = None
) -> Tuple[int, float, bool]:
    """计算 VQA 准确率"""
    # 处理 ground_truth_answers
    if not ground_truth_answers:
        gt_answers_str = []
    elif isinstance(ground_truth_answers[0], dict):
        gt_answers_str = [ans.get("answer", "") for ans in ground_truth_answers]
    else:
        gt_answers_str = [str(ans) for ans in ground_truth_answers]
    
    # 使用官方VQA评测
    if vqa_cache is not None and question_id:
        try:
            temp_kwargs = {'mode': 'w', 'suffix': '.json', 'delete': False}
            if TEMP_DIR:
                temp_kwargs['dir'] = TEMP_DIR
            with tempfile.NamedTemporaryFile(**temp_kwargs) as f:
                temp_result_file = f.name
                json.dump([{"answer": pred_answer, "question_id": question_id}], f, indent=4)
            
            try:
                with contextlib.redirect_stdout(StringIO()):
                    vqaRes = vqa_cache.loadRes(temp_result_file, ques_path)
                    vqaEval = VQAEval(vqa_cache, vqaRes, n=2)
                    vqaEval.params = {"question_id": [int(question_id)]}
                    vqaEval.evaluate()
                
                if int(question_id) in vqaEval.evalQA:
                    accuracy = vqaEval.evalQA[int(question_id)]
                else:
                    accuracy = 0.0
                
                acc_score = accuracy / 100.0 if accuracy > 1 else accuracy
                correct = 1 if acc_score > 0.0 else 0
                return correct, acc_score, True
            finally:
                if os.path.exists(temp_result_file):
                    os.remove(temp_result_file)
        except Exception as e:
            print(f"警告：VQA评测失败: {e}")
    
    # Fallback: 简单匹配
    if pred_answer is None:
        pred_answer = ""
    pred_lower = pred_answer.lower().strip()
    gt_lower = [ans.lower().strip() for ans in gt_answers_str]
    
    if pred_lower in gt_lower:
        return 1, 1.0, False
    return 0, 0.0, False


def compute_relevance(pred: str, gt_list: List[str]) -> Dict[str, float]:
    """计算pred与gt_list的相关性指标"""
    def _tok(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", s.lower())
    
    def token_f1(a: str, b: str) -> float:
        A, B = _tok(a), _tok(b)
        if len(A) == 0 or len(B) == 0:
            return 0.0
        ca, cb = Counter(A), Counter(B)
        common = sum((ca & cb).values())
        prec = common / max(1, len(A))
        rec = common / max(1, len(B))
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
    
    def edit_sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
    
    if pred is None:
        pred = ""
    pred = pred.strip().lower()
    gt_list = [(g or "").strip().lower() for g in gt_list]
    
    if pred == "" or len(gt_list) == 0:
        return {"vqa_rel_token_f1": 0.0, "vqa_rel_edit_sim": 0.0, "vqa_rel_score": 0.0}
    
    f1_max = max(token_f1(pred, g) for g in gt_list)
    ed_max = max(edit_sim(pred, g) for g in gt_list)
    rel_score = max(f1_max, ed_max)
    
    return {"vqa_rel_token_f1": f1_max, "vqa_rel_edit_sim": ed_max, "vqa_rel_score": rel_score}


def generate_rl_data_k64(
    sft_model,
    vqa_model,
    sampler_cache: Dict[str, List[int]],
    all_embeddings: torch.Tensor,
    dataset,
    num_beams: int = 5,
    temps: tuple = (1.0, 1.3),
    num_samples_per_temp: int = 2,
    num_random: int = 2,
    num_retrieval: int = 3,
    device: torch.device = None,
    generation_kwargs: Optional[Dict] = None,
    train_ques_path: Optional[str] = None,
    train_ann_path: Optional[str] = None,
    strict_eval: bool = True,
) -> Dict:
    """
    生成 RL 数据（K=64版本）
    
    关键修改：
    1. 每个query使用sampler_cache中的64个候选池索引
    2. 为每个query构建独立的候选池embedding [64, d]
    3. 生成的pointer使用局部索引 [0, 63]
    4. 输出数据包含 candidate_pool_ids 字段
    
    Args:
        sft_model: SFT 模型
        vqa_model: VQA 模型 interface
        sampler_cache: {query_id_str: [64个候选索引]}
        all_embeddings: [N, d] 所有样本的 embeddings
        dataset: 数据集对象
        其他参数同 generate_rl_data
    
    Returns:
        rl_data: 新的 RL 数据格式，每个query包含 candidate_pool_ids
    """
    if device is None:
        device = next(sft_model.parameters()).device
    
    rl_data = {}
    
    # 预加载 VQA 对象
    vqa_cache = None
    if train_ques_path and train_ann_path:
        try:
            print("预加载 VQA 标注文件...")
            vqa_cache = VQA(train_ann_path, train_ques_path)
            print("✓ VQA 对象已缓存")
        except Exception as e:
            print(f"警告：预加载 VQA 对象失败: {e}")
    
    # 确保 embeddings 在 device 上
    if all_embeddings.device != device:
        all_embeddings = all_embeddings.to(device)
    
    print(f"开始生成 RL 数据（K=64版本）...")
    print(f"  - Query 数量: {len(sampler_cache)}")
    print(f"  - Beam 数量: {num_beams}")
    print(f"  - 温度: {temps}")
    print(f"  - 每个温度采样数: {num_samples_per_temp}")
    print(f"  - 随机组合数: {num_random}")
    print(f"  - Retrieval数量: {num_retrieval}")
    print(f"  - Strict eval: {strict_eval}")
    
    # 统计变量
    total_accuracy_computations = 0
    file_metric_count = 0
    fallback_count = 0
    
    for query_id_str, candidate_pool_ids in tqdm(sampler_cache.items(), desc="生成 RL 数据"):
        query_id = int(query_id_str)
        
        # 验证候选池大小
        if len(candidate_pool_ids) != 64:
            print(f"警告：Query {query_id} 的候选池大小为 {len(candidate_pool_ids)}，跳过")
            continue
        
        # 【关键】为该query构建独立的候选池embedding [64, d]
        cand_emb = all_embeddings[candidate_pool_ids]  # [64, d]
        query_emb = all_embeddings[query_id]  # [d]
        
        # 构建局部索引到全局索引的映射
        local_to_global = {pos: idx for pos, idx in enumerate(candidate_pool_ids)}
        global_to_local = {idx: pos for pos, idx in enumerate(candidate_pool_ids)}
        
        # 检查 query_id 是否在候选池中（需要排除）
        exclude_local_indices = []
        if query_id in global_to_local:
            exclude_local_indices.append(global_to_local[query_id])
        
        # 生成 pointer 候选（使用局部索引 0~63）
        pointer_candidates = generate_pointer_candidates_for_query(
            model=sft_model,
            query_emb=query_emb,
            cand_emb=cand_emb,
            num_beams=num_beams,
            temps=temps,
            num_samples_per_temp=num_samples_per_temp,
            num_random=num_random,
            num_retrieval=num_retrieval,
            beam_search_fn=None,
            exclude_indices=exclude_local_indices if exclude_local_indices else None
        )
        
        # 获取 query 的原始数据
        query_item = dataset[query_id]
        image = query_item.get("image")
        question = query_item.get("question")
        gt_answers_raw = query_item.get("answers", [])
        
        # 处理 gt_answers_raw
        if gt_answers_raw and isinstance(gt_answers_raw[0], dict):
            gt_answers_raw = [ans.get("answer", "") for ans in gt_answers_raw]
        elif gt_answers_raw and not isinstance(gt_answers_raw[0], str):
            gt_answers_raw = [str(ans) for ans in gt_answers_raw]
        
        gt_answers_norm = [vqa_postprocess(ans, model_name="qwen2.5_vl_3B") for ans in gt_answers_raw]
        
        # 预构建该query的候选池数据
        candidate_pool = [dataset[idx] for idx in candidate_pool_ids]
        
        # 对每个 pointer 候选计算 correctness
        pointer_candidates_with_correctness = []
        for c in pointer_candidates:
            # pointer 是局部索引 [0, 63]
            pointer_local = c["pointer"]
            
            # 验证索引范围
            if any(p < 0 or p >= 64 for p in pointer_local):
                print(f"警告：Query {query_id} 的 pointer {pointer_local} 超出范围，跳过")
                continue
            
            # 转换为全局索引（用于记录）
            pointer_global = [local_to_global[p] for p in pointer_local]
            
            try:
                # 使用局部索引获取示例
                ex1 = candidate_pool[pointer_local[0]]
                ex2 = candidate_pool[pointer_local[1]]
                
                # 构建 prompt 并生成答案
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
                
                # 计算准确率
                question_id_str = query_item.get("question_id", str(query_id))
                
                correct, acc_score, used_file_metric = compute_vqa_accuracy(
                    pred_answer=pred_answer,
                    ground_truth_answers=gt_answers_raw,
                    question_id=question_id_str,
                    vqa_cache=vqa_cache,
                    ques_path=train_ques_path
                )
                
                # 严格模式检查
                if strict_eval and not used_file_metric:
                    continue
                
                total_accuracy_computations += 1
                if used_file_metric:
                    file_metric_count += 1
                else:
                    fallback_count += 1
                
                # 计算 relevance
                rel = compute_relevance(pred_answer, gt_answers_norm)
                
                # 保存结果（使用局部索引）
                c["pointer"] = pointer_local  # 局部索引 [0, 63]
                c["pointer_global"] = pointer_global  # 全局索引（用于调试）
                c["vqa_raw_generation"] = raw_generation
                c["vqa_pred_answer"] = pred_answer
                c["vqa_correct"] = correct
                c["vqa_acc_score"] = acc_score
                c["vqa_eval_mode"] = "vqaEval" if used_file_metric else "fallback"
                c.update(rel)
                
                pointer_candidates_with_correctness.append(c)
                
            except Exception as e:
                print(f"警告：计算 correctness 失败 (query_id={query_id}, pointer={pointer_local}): {e}")
                if not strict_eval:
                    c["pointer"] = pointer_local
                    c["pointer_global"] = pointer_global
                    c["vqa_raw_generation"] = ""
                    c["vqa_pred_answer"] = ""
                    c["vqa_correct"] = 0
                    c["vqa_acc_score"] = 0.0
                    c["vqa_eval_mode"] = "error"
                    c["eval_failed"] = True
                    pointer_candidates_with_correctness.append(c)
        
        # 保存到 rl_data
        rl_data[query_id_str] = {
            "query": {
                "query_id": query_id,
                "question_id": query_item.get("question_id", str(query_id)),
                "image_id": query_item.get("image_id", None),
                "question": question,
                "gt_answers_raw": gt_answers_raw,
                "gt_answers_norm": gt_answers_norm,
            },
            "candidate_pool_ids": candidate_pool_ids,  # 【关键】保存64个候选池索引
            "pointer_candidates": pointer_candidates_with_correctness
        }
    
    # 打印统计信息
    print(f"\n✓ RL 数据生成完成！")
    print(f"  - 总准确率计算次数: {total_accuracy_computations}")
    if total_accuracy_computations > 0:
        file_metric_ratio = file_metric_count / total_accuracy_computations * 100
        fallback_ratio = fallback_count / total_accuracy_computations * 100
        print(f"  - 使用文件方式计算: {file_metric_count} ({file_metric_ratio:.1f}%)")
        print(f"  - 使用 fallback: {fallback_count} ({fallback_ratio:.1f}%)")
    
    return rl_data


def main():
    parser = argparse.ArgumentParser(description="生成 RL 数据（K=64版本）")
    parser.add_argument("--sft_ckpt", type=str, required=True, help="SFT 模型 checkpoint 路径")
    parser.add_argument("--sampler_cache", type=str, required=True, help="Sampler 缓存 JSON 路径（包含每个query的64个候选池）")
    parser.add_argument("--output_path", type=str, required=True, help="输出 RL 数据 JSON 路径")
    parser.add_argument("--embeddings_path", type=str, required=True, help="所有样本的 embeddings 路径（.pt 文件）")
    parser.add_argument("--vqa_model", type=str, default="qwen2.5_vl_3B", help="VQA 模型名称")
    parser.add_argument("--dataset", type=str, default="okvqa_local", help="数据集名称")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam 数量")
    parser.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.3], help="温度列表")
    parser.add_argument("--num_samples_per_temp", type=int, default=2, help="每个温度的采样数量")
    parser.add_argument("--num_random", type=int, default=2, help="随机组合数量")
    parser.add_argument("--num_retrieval", type=int, default=3, help="Retrieval方法的数量")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--train_ques_path", type=str, help="训练集问题文件路径")
    parser.add_argument("--train_ann_path", type=str, help="训练集标注文件路径")
    parser.add_argument("--train_path", type=str, help="训练集JSON文件路径")
    parser.add_argument("--train_coco_root", type=str, help="COCO训练集图片根目录")
    parser.add_argument("--strict_eval", action="store_true", default=True, help="严格模式")
    parser.add_argument("--no_strict_eval", dest="strict_eval", action="store_false")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载 sampler 缓存
    print(f"加载 sampler 缓存: {args.sampler_cache}")
    with open(args.sampler_cache, "r") as f:
        sampler_cache = json.load(f)
    print(f"  - 包含 {len(sampler_cache)} 个 query 的候选池")
    
    # 验证候选池大小
    pool_sizes = [len(v) for v in sampler_cache.values()]
    print(f"  - 候选池大小: min={min(pool_sizes)}, max={max(pool_sizes)}, avg={sum(pool_sizes)/len(pool_sizes):.1f}")
    if min(pool_sizes) != 64 or max(pool_sizes) != 64:
        print(f"⚠️ 警告：候选池大小不全为64，请检查 sampler 缓存")
    
    # 加载 embeddings
    print(f"加载 embeddings: {args.embeddings_path}")
    all_embeddings = torch.load(args.embeddings_path, map_location=device)
    print(f"  - Embeddings shape: {all_embeddings.shape}")
    
    # 加载 SFT 模型
    print(f"加载 SFT 模型: {args.sft_ckpt}")
    sft_model = load_sft_model(args.sft_ckpt, device, d_model=args.d_model)
    
    # 构建配置
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
    okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
    
    train_path = args.train_path or os.path.join(okvqa_hf_dir, "vqav2_mscoco_train2014.json")
    train_coco_root = args.train_coco_root or os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "train2014")
    
    cfg = OmegaConf.create({
        "dataset": {
            "name": args.dataset,
            "version": "local",
            "train_path": train_path,
            "val_path": train_path,  # 使用训练集
            "train_coco_dataset_root": train_coco_root,
            "val_coco_dataset_root": train_coco_root,
        },
        "infer_model": {
            "name": "Qwen2.5-VL",
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "load_from_local": False,
            "precision": "bf16",
            "icd_join_char": "<|endofchunk|>",
            "system_prompt": "In the upcoming task, you will see four sets of dialogues...",
        },
        "task": {
            "task_name": "vqa",
            "template": "Question:<Q> Short answer:<A>",
            "column_token_map": {"question": "<Q>", "answer": "<A>"},
            "instruction": "",
            "image_field": "image",
            "output_column": "answer",
        },
        "precision": "bf16",
    })
    
    # 加载 VQA 模型
    print(f"加载 VQA 模型: {args.vqa_model}")
    vqa_model = load_vqa_model(args.vqa_model, device, cfg=cfg)
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    ds = load_ds(cfg)
    train_ds = ds["train"]
    
    # 加载生成参数
    task_config_path = "configs/task/vqa.yaml"
    if os.path.exists(task_config_path):
        task_cfg = OmegaConf.load(task_config_path)
        if hasattr(task_cfg, 'gen_args') and task_cfg.gen_args:
            generation_kwargs = OmegaConf.to_container(task_cfg.gen_args)
        else:
            generation_kwargs = {"max_new_tokens": 5, "num_beams": 3, "length_penalty": 0.0, "min_new_tokens": 0}
    else:
        generation_kwargs = {"max_new_tokens": 5, "num_beams": 3, "length_penalty": 0.0, "min_new_tokens": 0}
    
    if generation_kwargs.get("num_beams", 1) > 1:
        generation_kwargs["do_sample"] = False
    
    print(f"生成参数: {generation_kwargs}")
    
    # 生成 RL 数据
    rl_data = generate_rl_data_k64(
        sft_model=sft_model,
        vqa_model=vqa_model,
        sampler_cache=sampler_cache,
        all_embeddings=all_embeddings,
        dataset=train_ds,
        num_beams=args.num_beams,
        temps=tuple(args.temps),
        num_samples_per_temp=args.num_samples_per_temp,
        num_random=args.num_random,
        num_retrieval=args.num_retrieval,
        device=device,
        generation_kwargs=generation_kwargs,
        train_ques_path=args.train_ques_path,
        train_ann_path=args.train_ann_path,
        strict_eval=args.strict_eval,
    )
    
    # 添加 _meta 信息
    meta_info = {
        "_meta": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "k64",
            "description": "RL data with fixed K=64 candidate pool per query",
            "vqa_model": args.vqa_model,
            "task_gen_args": generation_kwargs,
            "sampler_cache": args.sampler_cache,
            "strict_eval": args.strict_eval,
            "params": {
                "num_beams": args.num_beams,
                "temps": args.temps,
                "num_samples_per_temp": args.num_samples_per_temp,
                "num_random": args.num_random,
                "num_retrieval": args.num_retrieval,
            }
        }
    }
    
    output_data = {**meta_info, **rl_data}
    
    # 保存
    print(f"保存 RL 数据到: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # 验证输出
    print(f"\n验证输出数据...")
    queries = {k: v for k, v in output_data.items() if k != '_meta'}
    
    pool_sizes = []
    trajectory_counts = []
    positive_counts = []
    pointer_range_ok = True
    
    for qid, qdata in queries.items():
        pool = qdata.get('candidate_pool_ids', [])
        trajectories = qdata.get('pointer_candidates', [])
        positives = sum(1 for t in trajectories if t.get('vqa_correct', 0) == 1)
        
        pool_sizes.append(len(pool))
        trajectory_counts.append(len(trajectories))
        positive_counts.append(positives)
        
        # 验证 pointer 范围
        for traj in trajectories:
            pointer = traj.get('pointer', [])
            for pos in pointer:
                if pos < 0 or pos >= 64:
                    print(f"⚠️ Query {qid}: pointer {pos} 超出范围 [0, 63]")
                    pointer_range_ok = False
    
    print(f"\n输出统计:")
    print(f"  - 总 query 数: {len(queries)}")
    print(f"  - 候选池大小: min={min(pool_sizes)}, max={max(pool_sizes)}, avg={sum(pool_sizes)/len(pool_sizes):.1f}")
    print(f"  - 轨迹数量: min={min(trajectory_counts)}, max={max(trajectory_counts)}, avg={sum(trajectory_counts)/len(trajectory_counts):.1f}")
    print(f"  - 正样本数量: min={min(positive_counts)}, max={max(positive_counts)}, avg={sum(positive_counts)/len(positive_counts):.1f}")
    print(f"  - All-Zero query: {sum(1 for p in positive_counts if p == 0)} ({sum(1 for p in positive_counts if p == 0)/len(positive_counts)*100:.1f}%)")
    
    # 断言验证
    assert all(s == 64 for s in pool_sizes), f"候选池大小不全为64"
    assert pointer_range_ok, "存在超出范围的 pointer"
    
    print(f"\n✓ 所有验证通过！")
    print(f"✓ RL 数据生成完成！")


if __name__ == "__main__":
    main()
