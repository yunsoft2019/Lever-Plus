"""
V3模型VQA评估脚本

评估流程：
1. 加载GRPO训练后的V3模型
2. 加载真实CLIP embedding
3. 使用V3模型进行范例选择
4. 调用VQA推理接口（Qwen2.5-VL）
5. 计算VQA准确率

作者: Lever-Plus Team
日期: 2025-12-02
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from loguru import logger

from lever_lm.models.v3 import (
    PointerSelectorV3,
    load_v3_from_grpo_checkpoint,
    load_v3_from_sft_checkpoint,
    predict_with_v3
)
from transformers import CLIPVisionModelWithProjection, AutoProcessor


def init_simple_vlm_interface(model_name: str, dataset: str, device: str):
    """
    简化的VLM接口初始化函数
    
    Args:
        model_name: 模型名称（如 "Qwen2.5-VL-3B-Instruct"）
        dataset: 数据集名称（如 "okvqa" 或 "vqav2"）
        device: 设备（如 "cuda:0"）
    
    Returns:
        interface: VLM接口实例
    """
    from open_mmicl.interface import Qwen2VLInterface
    
    # VQA任务的默认配置（与configs/infer_model/qwen2.5_vl_3B.yaml一致）
    prompt_template = "Question:<Q> Short answer:<A>"
    column_token_map = {"question": "<Q>", "answer": "<A>"}
    instruction = ""  # Qwen2.5-VL不需要额外指令
    
    # 根据模型名称确定模型路径
    if "Qwen2.5-VL" in model_name or "Qwen2VL" in model_name:
        # 检查是否包含完整路径
        if "/" in model_name:
            model_path = model_name
        else:
            # 使用默认路径
            model_path = f"Qwen/{model_name}"
        
        interface = Qwen2VLInterface(
            model_name=model_path,
            load_from_local=False,  # 从HuggingFace加载
            precision="bf16",
            device=device,
            prompt_template=prompt_template,
            column_token_map=column_token_map,
            instruction=instruction,
            image_field="image",
            label_field="answer",
            icd_join_char="<|endofchunk|>",
            system_prompt=None,
            use_lora=False,
            lora_checkpoint_path=None,
        )
        return interface
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def evaluate_v3_model(
    model: PointerSelectorV3,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    test_ids: List[int],
    device: torch.device,
    batch_size: int = 32
) -> Dict[int, List[int]]:
    """
    使用V3模型进行范例选择
    
    Args:
        model: V3模型
        query_embeddings: [N, d] 所有query的embedding
        candidate_embeddings: [K, d] 候选池embedding
        test_ids: 测试样本ID列表
        device: 设备
        batch_size: 批次大小
    
    Returns:
        predictions: {query_id: [icd1, icd2, ...]}
    """
    model.eval()
    predictions = {}
    
    num_batches = (len(test_ids) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Predicting"):
        batch_ids = test_ids[i * batch_size : (i + 1) * batch_size]
        
        # 获取query embedding
        query_emb = torch.stack([query_embeddings[qid] for qid in batch_ids]).to(device)
        
        # 扩展candidate embedding
        cand_emb = candidate_embeddings.unsqueeze(0).expand(len(batch_ids), -1, -1).to(device)
        
        # 预测
        with torch.no_grad():
            preds, scores = predict_with_v3(model, query_emb, cand_emb)
        
        # 保存结果
        for j, qid in enumerate(batch_ids):
            predictions[qid] = preds[j].cpu().tolist()
    
    return predictions


def compare_with_baseline(
    v3_predictions: Dict[int, List[int]],
    baseline_predictions: Dict[int, List[int]]
) -> Dict[str, float]:
    """
    比较V3预测与基线预测的差异
    
    Args:
        v3_predictions: V3模型预测
        baseline_predictions: 基线预测（如V2或beam search top-1）
    
    Returns:
        metrics: 比较指标
    """
    common_ids = set(v3_predictions.keys()) & set(baseline_predictions.keys())
    
    exact_match = 0
    partial_match = 0
    
    for qid in common_ids:
        v3_pred = v3_predictions[qid]
        base_pred = baseline_predictions[qid]
        
        if v3_pred == base_pred:
            exact_match += 1
        elif set(v3_pred) & set(base_pred):
            partial_match += 1
    
    total = len(common_ids)
    
    return {
        "exact_match_rate": exact_match / total if total > 0 else 0,
        "partial_match_rate": partial_match / total if total > 0 else 0,
        "total_samples": total
    }


def run_vqa_inference(
    interface,
    train_ds,
    test_ds,
    icd_idx_list: List[List[int]],
    val_ques_path: str,
    val_ann_path: str,
    model_name: str,
    generation_kwargs: dict,
) -> float:
    """
    运行VQA推理并计算准确率
    
    复用icl_inference.py中的inference_vqa_direct逻辑
    """
    import uuid
    from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy, postprocess_vqa_generation
    
    # 定义vqa_postprocess函数（从根目录utils.py复制）
    def vqa_postprocess(pred, model_name=None):
        """VQA答案后处理"""
        return postprocess_vqa_generation(pred)
    
    preds = []
    
    for idx, sample in enumerate(tqdm(test_ds, desc="VQA推理中", ncols=100)):
        if idx < len(icd_idx_list):
            example_indices = icd_idx_list[idx]
            
            # 获取范例
            ice_sample_list = []
            for ex_idx in example_indices:
                if ex_idx < len(train_ds):
                    ice_sample_list.append(train_ds[ex_idx])
            
            # 组合范例和测试样本
            data_sample_list = ice_sample_list + [sample]
            
            # 转换为prompt
            prompts = interface.transfer_prompts(
                [data_sample_list], is_last_for_generation=True
            )
            
            # 准备输入
            input_dict = interface.prepare_input(
                prompts, is_last_for_generation=True
            )
            
            if hasattr(input_dict, 'data'):
                input_dict = dict(input_dict.data)
            elif not isinstance(input_dict, dict):
                input_dict = dict(input_dict)
            
            data = {k: v.to(interface.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in input_dict.items()}
            
            # 处理Qwen2.5-VL特殊情况
            if 'image_grid_thw' in data:
                if data['image_grid_thw'].dim() == 3 and data['image_grid_thw'].shape[0] == 1:
                    data['image_grid_thw'] = data['image_grid_thw'].squeeze(0)
                    if 'image_nums' in data:
                        if isinstance(data['image_nums'], torch.Tensor) and data['image_nums'].numel() > 1:
                            data['image_nums'] = data['image_nums'][0:1]
                elif data['image_grid_thw'].dim() == 2:
                    if 'image_nums' not in data:
                        num_images = data['image_grid_thw'].shape[0]
                        data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
            
            # 推理
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
            answer = vqa_postprocess(prediction, model_name=model_name)
            
            question_id = sample.get('question_id', None)
            if question_id is not None:
                preds.append({
                    "answer": answer,
                    "question_id": question_id,
                })
    
    # 计算准确率
    if len(preds) > 0:
        random_uuid = str(uuid.uuid4())
        temp_result_file = f"{random_uuid}.json"
        
        with open(temp_result_file, "w") as f:
            json.dump(preds, f, indent=4)
        
        try:
            accuracy = compute_vqa_accuracy(temp_result_file, val_ques_path, val_ann_path)
            if accuracy > 1:
                accuracy = accuracy / 100
            return accuracy
        finally:
            if os.path.exists(temp_result_file):
                os.remove(temp_result_file)
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="V3模型VQA评估")
    parser.add_argument("--grpo_ckpt", type=str, required=True, help="GRPO检查点路径")
    parser.add_argument("--img_emb", type=str, required=True, help="图像embedding路径")
    parser.add_argument("--beam_data", type=str, required=True, help="束搜索数据JSON路径")
    parser.add_argument("--dataset", type=str, default="okvqa", help="数据集名称")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct", help="VLM模型名称")
    parser.add_argument("--shot_num", type=int, default=2, help="范例数量")
    parser.add_argument("--output_dir", type=str, default="results/v3_eval", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--test_num", type=int, default=None, help="测试样本数（None表示全部）")
    parser.add_argument("--skip_vqa", action="store_true", help="跳过VQA推理，只做范例选择")
    parser.add_argument("--full_train_set", action="store_true", help="使用完整训练集作为候选池（而非仅beam_data中的候选）")
    parser.add_argument("--candidate_size", type=int, default=None, help="限制候选池大小（用于测试泛化问题）")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载beam数据
    print(f"\n加载束搜索数据: {args.beam_data}")
    with open(args.beam_data, "r") as f:
        beam_data = json.load(f)
    
    # 提取候选池索引
    if args.full_train_set:
        # 使用完整训练集作为候选池（需要先加载数据集获取大小）
        print("[模式] 使用完整训练集作为候选池")
        candidate_indices = None  # 稍后根据img_emb确定
    else:
        all_icd_indices = set()
        # 检测数据格式：新格式（pointer_candidates）或旧格式（id_list）
        first_key = list(beam_data.keys())[0]
        first_data = beam_data[first_key]
        
        if "pointer_candidates" in first_data:
            # 新格式：RL数据
            print("✓ 检测到新格式数据（RL数据，包含pointer_candidates）")
            for qid, data in beam_data.items():
                for candidate in data.get("pointer_candidates", []):
                    pointer = candidate.get("pointer", [])
                    for idx in pointer:
                        all_icd_indices.add(idx)
        elif "id_list" in first_data:
            # 旧格式：传统beam数据
            print("✓ 检测到旧格式数据（传统beam数据）")
            for qid, data in beam_data.items():
                for beam in data["id_list"]:
                    for idx in beam[:-1]:
                        all_icd_indices.add(idx)
        else:
            raise ValueError(f"未知的数据格式！数据应包含 'pointer_candidates' 或 'id_list'")
        
        candidate_indices = sorted(list(all_icd_indices))
        K = len(candidate_indices)
        print(f"候选池大小: {K}")
    
    # 2. 加载embedding
    print(f"\n加载图像embedding: {args.img_emb}")
    img_emb_data = torch.load(args.img_emb, weights_only=False)
    if not isinstance(img_emb_data, torch.Tensor):
        img_emb_data = torch.from_numpy(img_emb_data).float()
    
    d_model = img_emb_data.shape[1]
    print(f"Embedding维度: {d_model}")
    
    # 候选池embedding
    if args.full_train_set:
        candidate_indices = list(range(img_emb_data.shape[0]))
        candidate_embeddings = img_emb_data  # 使用全部训练集
        K = img_emb_data.shape[0]
        print(f"候选池大小: {K} (完整训练集)")
    else:
        candidate_embeddings = img_emb_data[candidate_indices]  # [K, d]
    
    # 可选：限制候选池大小（用于测试泛化问题）
    if args.candidate_size is not None and args.candidate_size < K:
        print(f"\n[限制候选池] 从 {K} 减少到 {args.candidate_size}")
        candidate_indices = candidate_indices[:args.candidate_size]
        candidate_embeddings = candidate_embeddings[:args.candidate_size]
        K = args.candidate_size
    
    # 3. 加载V3模型
    print(f"\n加载检查点: {args.grpo_ckpt}")
    ckpt = torch.load(args.grpo_ckpt, map_location='cpu', weights_only=False)
    
    # 处理不同格式的checkpoint
    if 'model_state_dict' in ckpt:
        model_state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        # PyTorch Lightning格式，需要去掉前缀
        model_state = {}
        for k, v in ckpt['state_dict'].items():
            # 去掉 "lever_lm." 前缀
            if k.startswith('lever_lm.'):
                new_key = k[len('lever_lm.'):]
            else:
                new_key = k
            model_state[new_key] = v
    else:
        model_state = ckpt
    
    # 【自动检测架构】检查checkpoint的d_model和num_layers
    detected_d_model = None
    detected_num_layers = 1  # 默认单层
    is_adapter_format = False
    
    # 检测是否是PointerSelectorAdapter格式
    if 'pointer_selector.input_proj.weight' in model_state:
        is_adapter_format = True
        detected_d_model = model_state['pointer_selector.input_proj.weight'].shape[1]
        # 检测是否有多层cross_attn
        if 'pointer_selector.cross_attn_layers.0.in_proj_weight' in model_state:
            detected_num_layers = sum(1 for k in model_state if 'cross_attn_layers' in k and 'in_proj_weight' in k)
        print(f"  检测到Adapter格式: d_model={detected_d_model}, num_layers={detected_num_layers}")
    elif 'input_proj.weight' in model_state:
        detected_d_model = model_state['input_proj.weight'].shape[1]
        if 'cross_attn_layers.0.in_proj_weight' in model_state:
            detected_num_layers = sum(1 for k in model_state if 'cross_attn_layers' in k and 'in_proj_weight' in k)
        print(f"  检测到直接格式: d_model={detected_d_model}, num_layers={detected_num_layers}")
    
    # 检查d_model是否与embeddings匹配
    if detected_d_model is not None and detected_d_model != d_model:
        print(f"\n⚠️  警告: checkpoint的d_model({detected_d_model})与embeddings的d_model({d_model})不匹配！")
        print(f"   checkpoint使用的CLIP模型与embeddings不同，需要重新生成embeddings或使用匹配的checkpoint")
        print(f"   当前将使用d_model={detected_d_model}，但评估结果可能不正确")
        d_model = detected_d_model
    
    # 处理Adapter格式的权重映射
    if is_adapter_format:
        v3_model_state = {}
        for old_key, value in model_state.items():
            if old_key.startswith('pointer_selector.'):
                new_key = old_key[len('pointer_selector.'):]
                # 单层cross_attn -> 多层cross_attn_layers
                if 'cross_attn.' in new_key and 'cross_attn_layers' not in new_key:
                    new_key = new_key.replace('cross_attn.', 'cross_attn_layers.0.')
                if 'attn_norm.' in new_key and 'attn_norms' not in new_key:
                    new_key = new_key.replace('attn_norm.', 'attn_norms.0.')
                v3_model_state[new_key] = value
        
        # 将第一层权重复制到其他层
        keys_to_copy = list(v3_model_state.keys())
        for key in keys_to_copy:
            if 'cross_attn_layers.0' in key:
                for layer_idx in [1, 2]:
                    new_key = key.replace('.0.', f'.{layer_idx}.')
                    if new_key not in v3_model_state:
                        v3_model_state[new_key] = v3_model_state[key].clone()
            if 'attn_norms.0' in key:
                for layer_idx in [1, 2]:
                    new_key = key.replace('.0.', f'.{layer_idx}.')
                    if new_key not in v3_model_state:
                        v3_model_state[new_key] = v3_model_state[key].clone()
        print(f"  Adapter->V3权重映射完成: {len(v3_model_state)} 个权重")
    else:
        v3_model_state = model_state
    
    model = PointerSelectorV3(
        d_model=d_model,
        K=K,
        shot_num=args.shot_num,
        num_layers=max(detected_num_layers, 1)  # 至少1层
    )
    model.load_state_dict(v3_model_state, strict=False)
    model = model.to(device)
    model.eval()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 加载数据集
    print(f"\n加载数据集: {args.dataset}")
    from lever_lm.load_ds_utils import load_vqav2_ds
    
    # 数据集配置 - 使用相对路径
    dataset_configs = {
        "okvqa": {
            "version": "local",
            "train_path": "datasets/okvqa/okvqa_hf/vqav2_mscoco_train2014.json",
            "val_path": "datasets/okvqa/okvqa_hf/vqav2_mscoco_val2014.json",
            "train_coco_root": "datasets/mscoco/mscoco2014/train2014",
            "val_coco_root": "datasets/mscoco/mscoco2014/val2014",
            "val_ques_path": "datasets/okvqa/OpenEnded_mscoco_val2014_questions.json",
            "val_ann_path": "datasets/okvqa/mscoco_val2014_annotations.json",
        },
        "vqav2": {
            "version": "local",
            "train_path": "datasets/vqav2/vqav2_hf/vqav2_mscoco_train2014.json",
            "val_path": "datasets/vqav2/vqav2_hf/vqav2_mscoco_val2014.json",
            "train_coco_root": "datasets/mscoco/mscoco2014/train2014",
            "val_coco_root": "datasets/mscoco/mscoco2014/val2014",
            "val_ques_path": "datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
            "val_ann_path": "datasets/vqav2/v2_mscoco_val2014_annotations.json",
        }
    }
    
    if args.dataset not in dataset_configs:
        print(f"错误：不支持的数据集 {args.dataset}")
        return
    
    ds_cfg = dataset_configs[args.dataset]
    ds = load_vqav2_ds(
        version=ds_cfg["version"],
        train_path=ds_cfg["train_path"],
        val_path=ds_cfg["val_path"],
        train_coco_dataset_root=ds_cfg["train_coco_root"],
        val_coco_dataset_root=ds_cfg["val_coco_root"],
    )
    train_ds = ds["train"]
    test_ds = ds["validation"]
    
    if args.test_num:
        test_ds = test_ds.select(range(min(args.test_num, len(test_ds))))
    print(f"训练集: {len(train_ds)}, 测试集: {len(test_ds)}")
    
    # 5. 使用V3模型进行范例选择
    print(f"\n使用V3模型选择范例...")
    
    # 构建位置到原始索引的映射
    pos_to_idx = {pos: idx for pos, idx in enumerate(candidate_indices)}
    
    # 【修复】根据d_model选择正确的CLIP模型
    if d_model == 512:
        clip_model_name = "openai/clip-vit-base-patch32"
    else:
        clip_model_name = "openai/clip-vit-large-patch14"
    print(f"\n加载CLIP模型: {clip_model_name} (d_model={d_model})")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device)
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    print(f"✓ CLIP模型加载完成")
    
    icd_idx_list = []
    num_batches = (len(test_ds) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="V3范例选择"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(test_ds))
        batch_size = end_idx - start_idx
        
        # 【修复】使用CLIP实时编码验证集样本的图像
        batch_images = [test_ds[i]["image"] for i in range(start_idx, end_idx)]
        clip_inputs = clip_processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_query_emb = clip_model(**clip_inputs).image_embeds  # [B, d]
            batch_query_emb = batch_query_emb / batch_query_emb.norm(dim=-1, keepdim=True)  # L2归一化
        
        # 扩展候选embedding
        batch_cand_emb = candidate_embeddings.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        # 预测
        with torch.no_grad():
            preds, _ = predict_with_v3(model, batch_query_emb, batch_cand_emb)
        
        # 转换为原始索引
        for pred in preds:
            original_indices = [pos_to_idx[p.item()] for p in pred]
            icd_idx_list.append(original_indices)
    
    print(f"✓ 范例选择完成: {len(icd_idx_list)}个样本")
    print(f"  示例 (样本0): {icd_idx_list[0]}")
    
    # 6. 保存范例选择结果
    os.makedirs(args.output_dir, exist_ok=True)
    icd_output_path = os.path.join(args.output_dir, "v3_icd_predictions.json")
    with open(icd_output_path, "w") as f:
        json.dump(icd_idx_list, f)
    print(f"✓ 范例选择结果保存至: {icd_output_path}")
    
    # 7. VQA推理（可选）
    if not args.skip_vqa:
        print(f"\n开始VQA推理...")
        
        # 使用之前加载的数据集配置中的路径
        val_ques_path = ds_cfg["val_ques_path"]
        val_ann_path = ds_cfg["val_ann_path"]
        
        # 初始化VLM
        interface = init_simple_vlm_interface(args.model_name, args.dataset, args.device)
        
        generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": False,
        }
        
        # 运行VQA推理
        accuracy = run_vqa_inference(
            interface=interface,
            train_ds=train_ds,
            test_ds=test_ds,
            icd_idx_list=icd_idx_list,
            val_ques_path=val_ques_path,
            val_ann_path=val_ann_path,
            model_name=args.model_name,
            generation_kwargs=generation_kwargs,
        )
        
        print(f"\n{'='*70}")
        print(f"VQA准确率: {accuracy*100:.2f}%")
        print(f"{'='*70}")
        
        # 保存结果
        result_path = os.path.join(args.output_dir, "v3_vqa_result.json")
        with open(result_path, "w") as f:
            json.dump({
                "accuracy": accuracy,
                "model": args.model_name,
                "dataset": args.dataset,
                "shot_num": args.shot_num,
                "grpo_ckpt": args.grpo_ckpt,
            }, f, indent=2)
        print(f"✓ 结果保存至: {result_path}")
    else:
        print("\n跳过VQA推理（使用 --skip_vqa 参数）")


if __name__ == "__main__":
    # 如果有命令行参数，运行main()
    if len(sys.argv) > 1:
        main()
    else:
        # 否则运行简单测试
        print("="*70)
        print("V3模型VQA评估 - 使用方法")
        print("="*70)
        print("""
用法:
  python -m lever_lm.workflows.evaluate_v3 \\
    --grpo_ckpt <GRPO检查点路径> \\
    --img_emb <图像embedding路径> \\
    --beam_data <束搜索数据路径> \\
    --dataset okvqa \\
    --output_dir results/v3_eval

参数:
  --grpo_ckpt   GRPO训练后的检查点 (必需)
  --img_emb     CLIP图像embedding缓存 (必需)
  --beam_data   束搜索数据JSON (必需)
  --dataset     数据集名称 (默认: okvqa)
  --model_name  VLM模型名称 (默认: Qwen2.5-VL-3B-Instruct)
  --shot_num    范例数量 (默认: 2)
  --test_num    测试样本数 (默认: 全部)
  --skip_vqa    跳过VQA推理，只做范例选择
  --batch_size  批次大小 (默认: 32)

示例:
  python -m lever_lm.workflows.evaluate_v3 \\
    --grpo_ckpt results/okvqa/grpo_v3/grpo_epoch3.pt \\
    --img_emb results/okvqa/cache/vqa-okvqa-clip-vit-large-patch14-ImgFeatures.pth \\
    --beam_data "results/okvqa/generated_data/sub_proc_data/vqa-okvqa-Qwen2_5-VL-3B-Instruct-ImgSimSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:800_rank:0_(0, 800).json" \\
    --dataset okvqa \\
    --test_num 100 \\
    --skip_vqa
        """)
