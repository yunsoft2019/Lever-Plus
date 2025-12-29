#!/usr/bin/env python
"""
V4-7 原生推理脚本

直接使用完整的 V4-7 模型进行推理，保留 DPP 多样性增益机制。
不转换为 V2 格式，确保训练和推理的一致性。

使用方法:
    python scripts/inference_v4_7_native.py \
        --checkpoint results/okvqa/model_cpk/v3_plan_v4_7_rank32/grpo_epoch4.pt \
        --dpp_rank 32 \
        --test_data_num 800 \
        --shot_num_list 1 2 3 4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# 添加项目根目录到 path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lever_lm.models.v3.pointer_selector_v4_7_rl import PointerSelectorV4_7_RL


def load_v4_7_model(checkpoint_path: str, dpp_rank: int, device: torch.device):
    """加载完整的 V4-7 模型"""
    print(f"加载 V4-7 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    # 从 state_dict 推断模型配置
    hidden_dim = 256
    d_model = 512
    
    for key in state_dict.keys():
        if "query_proj.weight" in key:
            hidden_dim = state_dict[key].shape[0]
        if "input_proj.weight" in key:
            d_model = state_dict[key].shape[1]
    
    # 检测是否有 step_emb 和 gru
    use_step_emb = any("step_emb" in k for k in state_dict.keys())
    use_gru = any("decoder_gru" in k for k in state_dict.keys())
    
    # 从 state_dict 检测 dpp_rank
    for key in state_dict.keys():
        if "dpp_proj.weight" in key:
            dpp_rank = state_dict[key].shape[0]
            break
    
    print(f"模型配置: d_model={d_model}, hidden_dim={hidden_dim}")
    print(f"  use_step_emb={use_step_emb}, use_gru={use_gru}, dpp_rank={dpp_rank}")
    
    # 创建模型
    model = PointerSelectorV4_7_RL(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=1,
        use_step_emb=use_step_emb,
        use_gru=use_gru,
        dpp_rank=dpp_rank,
        dpp_lambda_init=-2.0,
        label_smoothing=0.0,
        dropout=0.0  # 推理时关闭 dropout
    )

    # 加载权重
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"缺失参数: {len(missing)} 个")
        for k in missing[:5]:
            print(f"  - {k}")
    if unexpected:
        print(f"多余参数: {len(unexpected)} 个")
    
    model = model.to(device)
    model.eval()
    
    # 打印 DPP lambda 值
    dpp_lambda_effective = torch.nn.functional.softplus(model.dpp_lambda).item()
    print(f"DPP Lambda (effective): {dpp_lambda_effective:.4f}")
    
    print(f"✓ V4-7 模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_clip_encoder(device: torch.device):
    """加载 CLIP 编码器"""
    from transformers import CLIPProcessor, CLIPModel
    
    clip_name = "openai/clip-vit-base-patch32"
    print(f"加载 CLIP 编码器: {clip_name}")
    
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_model.eval()
    
    return clip_model, processor


def encode_with_clip(clip_model, processor, images, texts, device):
    """使用 CLIP 编码图像和文本"""
    with torch.no_grad():
        # 编码图像
        image_inputs = processor(images=images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_features = clip_model.get_image_features(**image_inputs)
        
        # 编码文本
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = clip_model.get_text_features(**text_inputs)
        
        # 拼接图像和文本特征
        combined = torch.cat([image_features, text_features], dim=-1)
        
        # L2 归一化
        combined = torch.nn.functional.normalize(combined, p=2, dim=-1)
        
    return combined


def run_inference(
    model: PointerSelectorV4_7_RL,
    clip_model,
    processor,
    val_dataset,
    train_dataset,
    shot_num: int,
    device: torch.device,
    test_data_num: int = 800
):
    """运行推理"""
    from PIL import Image
    
    # 预编码所有训练样本（候选池）
    print(f"预编码训练集 ({len(train_dataset)} 个样本)...")
    train_embeddings = []
    train_batch_size = 32
    
    for i in tqdm(range(0, len(train_dataset), train_batch_size), desc="编码训练集"):
        batch_end = min(i + train_batch_size, len(train_dataset))
        batch_images = []
        batch_texts = []
        
        for j in range(i, batch_end):
            sample = train_dataset[j]
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            batch_images.append(img)
            batch_texts.append(sample["question"])
        
        batch_emb = encode_with_clip(clip_model, processor, batch_images, batch_texts, device)
        train_embeddings.append(batch_emb.cpu())
    
    train_embeddings = torch.cat(train_embeddings, dim=0)  # [N_train, d]
    print(f"训练集编码完成: {train_embeddings.shape}")
    
    # 推理
    results = []
    test_samples = min(test_data_num, len(val_dataset))
    
    print(f"\n开始推理 (shot_num={shot_num}, test_samples={test_samples})...")
    
    for idx in tqdm(range(test_samples), desc=f"推理 shot={shot_num}"):
        sample = val_dataset[idx]
        
        # 编码 query
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        
        query_emb = encode_with_clip(
            clip_model, processor, [img], [sample["question"]], device
        )  # [1, d]
        
        # 候选池（使用全部训练集）
        cand_emb = train_embeddings.unsqueeze(0).to(device)  # [1, N_train, d]
        
        # 使用 V4-7 模型选择
        with torch.no_grad():
            predictions, scores = model.predict(
                query_emb, cand_emb, shot_num=shot_num
            )
        
        selected_indices = predictions[0].cpu().tolist()
        
        results.append({
            "query_idx": idx,
            "question_id": sample.get("question_id", idx),
            "selected_indices": selected_indices,
            "question": sample["question"],
            "answer": sample.get("answer", sample.get("answers", [])),
        })
    
    return results, train_dataset


def run_vqa_inference(
    results: List[Dict],
    train_dataset,
    val_dataset,
    shot_num: int,
    device: torch.device
):
    """使用 Qwen2.5-VL 进行 VQA 推理"""
    from open_mmicl.interface.qwen2vl_interface import Qwen2VLInterface
    from PIL import Image
    
    print(f"\n加载 Qwen2.5-VL 模型...")
    qwen = Qwen2VLInterface(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        init_device=str(device),
        load_from_local=False
    )
    
    predictions = []
    
    system_prompt = (
        "In the upcoming task, you will see four sets of dialogues, each containing two roles: "
        "user and assistant. The user is the questioner, who provides an image and asks a question "
        "based on it; the assistant is the responder, who answers according to the image and question "
        "provided by the user. Afterward, you will receive an image and a question from the user. "
        "Please act as the assistant and answer based on the four previous dialogue sets and your own "
        "knowledge. Strictly follow the answering format: if the examples use only one or two keywords, "
        "your reply must also use only one or two keywords; if the examples contain no more than three "
        "tokens, your reply must not exceed three tokens either."
    )
    
    print(f"VQA 推理 (shot_num={shot_num})...")
    
    for result in tqdm(results, desc="VQA 推理"):
        query_idx = result["query_idx"]
        selected_indices = result["selected_indices"][:shot_num]
        
        # 构建 ICL prompt
        icl_examples = []
        for icd_idx in selected_indices:
            icd_sample = train_dataset[icd_idx]
            icd_img = icd_sample["image"]
            if not isinstance(icd_img, Image.Image):
                icd_img = Image.open(icd_img).convert("RGB")
            
            # 获取答案
            icd_answer = icd_sample.get("answer", "")
            if not icd_answer and "answers" in icd_sample:
                answers = icd_sample["answers"]
                if isinstance(answers, list) and len(answers) > 0:
                    if isinstance(answers[0], dict):
                        icd_answer = answers[0].get("answer", "")
                    else:
                        icd_answer = answers[0]
            
            icl_examples.append({
                "image": icd_img,
                "question": icd_sample["question"],
                "answer": icd_answer
            })
        
        # Query
        query_sample = val_dataset[query_idx]
        query_img = query_sample["image"]
        if not isinstance(query_img, Image.Image):
            query_img = Image.open(query_img).convert("RGB")
        
        # 生成答案
        try:
            pred_answer = qwen.generate_with_icl(
                query_image=query_img,
                query_question=query_sample["question"],
                icl_examples=icl_examples,
                system_prompt=system_prompt,
                max_new_tokens=5
            )
        except Exception as e:
            print(f"生成失败 (idx={query_idx}): {e}")
            pred_answer = ""
        
        predictions.append({
            "question_id": result["question_id"],
            "answer": pred_answer
        })
    
    return predictions


def evaluate_vqa(predictions: List[Dict], val_dataset, test_data_num: int):
    """评估 VQA 结果"""
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
    import tempfile
    
    # 准备 ground truth
    gt_data = {
        "annotations": [],
        "questions": []
    }
    
    for idx in range(min(test_data_num, len(val_dataset))):
        sample = val_dataset[idx]
        qid = sample.get("question_id", idx)
        
        answers = sample.get("answers", [])
        if isinstance(answers, str):
            answers = [{"answer": answers}]
        elif isinstance(answers, list):
            if len(answers) > 0 and not isinstance(answers[0], dict):
                answers = [{"answer": a} for a in answers]
        
        gt_data["annotations"].append({
            "question_id": qid,
            "answers": answers
        })
        gt_data["questions"].append({
            "question_id": qid,
            "question": sample["question"]
        })
    
    # 计算准确率（简单匹配）
    correct = 0
    total = len(predictions)
    
    for pred in predictions:
        qid = pred["question_id"]
        pred_answer = pred["answer"].strip().lower()
        
        # 找到对应的 ground truth
        for ann in gt_data["annotations"]:
            if ann["question_id"] == qid:
                gt_answers = [a["answer"].strip().lower() for a in ann["answers"]]
                if pred_answer in gt_answers:
                    correct += 1
                break
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="V4-7 Native Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="V4-7 checkpoint 路径")
    parser.add_argument("--dpp_rank", type=int, default=32, help="DPP 低秩投影维度")
    parser.add_argument("--test_data_num", type=int, default=800, help="测试样本数")
    parser.add_argument("--shot_num_list", type=int, nargs="+", default=[1, 2, 3, 4], help="Shot 数量列表")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--skip_vqa", action="store_true", help="跳过 VQA 推理，只输出选择结果")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_v4_7_model(args.checkpoint, args.dpp_rank, device)
    clip_model, processor = load_clip_encoder(device)
    
    # 加载数据集
    print("\n加载数据集...")
    okvqa_path = os.environ.get("OKVQA_PATH", "/mnt/share/yiyun/Datasets/okvqa")
    coco_path = os.environ.get("COCO_PATH", "/mnt/share/yiyun/Datasets/coco")
    
    # 使用 HuggingFace datasets 加载
    train_dataset = load_dataset(
        "json",
        data_files=f"{okvqa_path}/okvqa_hf/vqav2_mscoco_train2014.json",
        split="train"
    )
    val_dataset = load_dataset(
        "json",
        data_files=f"{okvqa_path}/okvqa_hf/vqav2_mscoco_val2014.json",
        split="train"
    )
    
    # 添加图像路径
    def add_image_path(example, idx, split="train"):
        img_dir = f"{coco_path}/mscoco2014/train2014" if split == "train" else f"{coco_path}/mscoco2014/val2014"
        image_id = example.get("image_id", idx)
        example["image"] = f"{img_dir}/COCO_{split}2014_{str(image_id).zfill(12)}.jpg"
        return example
    
    train_dataset = train_dataset.map(lambda x, i: add_image_path(x, i, "train"), with_indices=True)
    val_dataset = val_dataset.map(lambda x, i: add_image_path(x, i, "val"), with_indices=True)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 对每个 shot_num 进行推理
    all_results = {}
    
    for shot_num in args.shot_num_list:
        print(f"\n{'='*60}")
        print(f"Shot {shot_num}")
        print("="*60)
        
        # 选择样本
        results, train_ds = run_inference(
            model, clip_model, processor,
            val_dataset, train_dataset,
            shot_num, device, args.test_data_num
        )
        
        if not args.skip_vqa:
            # VQA 推理
            predictions = run_vqa_inference(
                results, train_ds, val_dataset,
                shot_num, device
            )
            
            # 评估
            accuracy = evaluate_vqa(predictions, val_dataset, args.test_data_num)
            print(f"\nShot {shot_num} 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            all_results[shot_num] = {
                "accuracy": accuracy,
                "predictions": predictions
            }
        else:
            all_results[shot_num] = {
                "selections": results
            }
    
    # 打印汇总
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    for shot_num in args.shot_num_list:
        if "accuracy" in all_results[shot_num]:
            acc = all_results[shot_num]["accuracy"]
            print(f"Shot {shot_num}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n✓ V4-7 原生推理完成！")


if __name__ == "__main__":
    main()
