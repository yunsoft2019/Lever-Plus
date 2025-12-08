#!/usr/bin/env python3
"""
V3模型推理脚本 - 与inference.sh使用相同的评估逻辑
用于与V0/V1/V2进行公平对比
"""
import argparse
import json
import os
import sys
import uuid

import torch
from loguru import logger
from tqdm import tqdm

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lever_lm.models.v3 import PointerSelectorV3
from open_mmicl.interface import Qwen2VLInterface
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy
from transformers import CLIPVisionModelWithProjection, AutoProcessor

# 导入根目录的utils.py
import importlib.util
_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
vqa_postprocess = _root_utils.vqa_postprocess

from lever_lm.load_ds_utils import load_vqav2_ds


def load_v3_model(ckpt_path: str, img_emb_path: str, device: torch.device):
    """加载V3模型"""
    # 加载图像embedding获取维度
    img_emb = torch.load(img_emb_path, map_location='cpu', weights_only=True)
    if isinstance(img_emb, dict):
        sample_emb = list(img_emb.values())[0]
    else:
        sample_emb = img_emb[0]
    d_model = sample_emb.shape[-1]
    
    # 加载检查点
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    
    # 检测num_layers
    num_layers = 1
    for key in ckpt.keys():
        if 'cross_attn_layers' in key:
            layer_idx = int(key.split('cross_attn_layers.')[1].split('.')[0])
            num_layers = max(num_layers, layer_idx + 1)
    
    # 初始化模型
    model = PointerSelectorV3(
        d_model=d_model,
        hidden_dim=256,
        num_layers=num_layers
    )
    
    # 加载权重
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, img_emb


def select_examples_v3(model, query_emb, cand_emb, shot_num: int):
    """使用V3模型选择范例"""
    with torch.no_grad():
        result = model.forward(query_emb, cand_emb, return_loss=False)
        logits = result['logits']  # [1, S, K]
        
        # 贪婪解码选择top-1
        selected = []
        for s in range(min(shot_num, logits.shape[1])):
            scores = logits[0, s, :]  # [K]
            # 排除已选择的
            for idx in selected:
                scores[idx] = float('-inf')
            best_idx = scores.argmax().item()
            selected.append(best_idx)
        
        return selected


def inference_vqa_v3(
    model,
    img_emb,
    clip_model,
    clip_processor,
    interface,
    train_ds,
    test_ds,
    shot_num: int,
    device: torch.device,
    model_name: str,
    val_ques_path: str,
    val_ann_path: str,
    test_num: int = None,
):
    """V3模型VQA推理"""
    # img_emb是tensor [N, D]，按索引访问
    # 候选embedding就是训练集的embedding
    cand_emb = img_emb.unsqueeze(0).to(device)  # [1, K, D]
    
    # 生成配置
    generation_kwargs = {
        "max_new_tokens": 32,
        "do_sample": False,
        "num_beams": 1,
    }
    
    preds = []
    test_samples = list(test_ds)
    if test_num is not None:
        test_samples = test_samples[:test_num]
    
    for idx, sample in enumerate(tqdm(test_samples, desc="VQA推理中")):
        # 使用CLIP编码query图像
        clip_inputs = clip_processor(images=[sample["image"]], return_tensors="pt").to(device)
        with torch.no_grad():
            query_emb = clip_model(**clip_inputs).image_embeds  # [1, d]
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)  # L2归一化
        
        # V3模型选择范例
        selected_indices = select_examples_v3(model, query_emb, cand_emb, shot_num)
        
        # 获取ICDs
        icds = [train_ds[i] for i in selected_indices]
        
        # 构建prompt并推理 (ICDs + query)
        data_sample_list = icds + [sample]
        prompts = interface.transfer_prompts([data_sample_list], is_last_for_generation=True)
        input_dict = interface.prepare_input(prompts, is_last_for_generation=True)
        
        # 处理 BatchFeature 对象
        if hasattr(input_dict, 'data'):
            input_dict = dict(input_dict.data)
        elif not isinstance(input_dict, dict):
            input_dict = dict(input_dict)
        
        # 将数据移动到设备
        data = {k: v.to(interface.device) if isinstance(v, torch.Tensor) else v 
               for k, v in input_dict.items()}
        
        # 处理image_grid_thw
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
                accuracy_percent = accuracy
            else:
                accuracy_percent = accuracy * 100
            return accuracy_percent
        finally:
            if os.path.exists(temp_result_file):
                os.remove(temp_result_file)
    else:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="V3模型推理（与V0/V1/V2公平对比）")
    parser.add_argument("--v3_ckpt", type=str, required=True, help="V3检查点路径(.pt)")
    parser.add_argument("--img_emb", type=str, required=True, help="图像embedding路径")
    parser.add_argument("--dataset", type=str, default="okvqa", help="数据集名称")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct", help="VLM模型名称")
    parser.add_argument("--shot_num", type=int, default=1, help="范例数量")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--test_num", type=int, default=None, help="测试样本数")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载V3模型
    print(f"加载V3模型: {args.v3_ckpt}")
    model, img_emb = load_v3_model(args.v3_ckpt, args.img_emb, device)
    print(f"✓ V3模型加载完成")
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    dataset_configs = {
        "okvqa": {
            "version": "local",
            "train_path": "datasets/okvqa/okvqa_hf/vqav2_mscoco_train2014.json",
            "val_path": "datasets/okvqa/okvqa_hf/vqav2_mscoco_val2014.json",
            "train_coco_root": "datasets/mscoco/mscoco2014/train2014",
            "val_coco_root": "datasets/mscoco/mscoco2014/val2014",
        },
        "vqav2": {
            "version": "local",
            "train_path": "datasets/vqav2/vqav2_hf/vqav2_mscoco_train2014.json",
            "val_path": "datasets/vqav2/vqav2_hf/vqav2_mscoco_val2014.json",
            "train_coco_root": "datasets/mscoco/mscoco2014/train2014",
            "val_coco_root": "datasets/mscoco/mscoco2014/val2014",
        }
    }
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
    print(f"✓ 数据集加载完成: 训练集 {len(train_ds)}, 测试集 {len(test_ds)}")
    
    # 加载CLIP模型（用于编码query图像）
    d_model = model.d_model
    if d_model == 512:
        clip_model_name = "openai/clip-vit-base-patch32"
    else:
        clip_model_name = "openai/clip-vit-large-patch14"
    print(f"加载CLIP模型: {clip_model_name}")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device)
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    print(f"✓ CLIP模型加载完成")
    
    # 初始化VLM接口
    print(f"初始化VLM: {args.model_name}")
    interface = Qwen2VLInterface(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        load_from_local=False,
        precision="bf16",
        device=args.device,
        prompt_template="Question:<Q> Short answer:<A>",
        column_token_map={"question": "<Q>", "answer": "<A>"},
        instruction="",
        icd_join_char="\n",
        image_field="image",
        label_field="answer",
    )
    print(f"✓ VLM加载完成")
    
    # VQA路径
    if args.dataset == "okvqa":
        val_ques_path = "datasets/okvqa/OpenEnded_mscoco_val2014_questions.json"
        val_ann_path = "datasets/okvqa/mscoco_val2014_annotations.json"
    else:
        val_ques_path = "datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
        val_ann_path = "datasets/vqav2/v2_mscoco_val2014_annotations.json"
    
    # 推理
    print(f"\n开始VQA推理 (shot_num={args.shot_num}, test_num={args.test_num})...")
    accuracy = inference_vqa_v3(
        model=model,
        img_emb=img_emb,
        clip_model=clip_model,
        clip_processor=clip_processor,
        interface=interface,
        train_ds=train_ds,
        test_ds=test_ds,
        shot_num=args.shot_num,
        device=device,
        model_name=args.model_name,
        val_ques_path=val_ques_path,
        val_ann_path=val_ann_path,
        test_num=args.test_num,
    )
    
    print(f"\n======================================================================")
    print(f"VQA准确率: {accuracy:.2f}%")
    print(f"======================================================================")


if __name__ == "__main__":
    main()
