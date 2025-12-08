#!/usr/bin/env python3
"""
用V2的CLIP+LoRA模型来评估V3的pointer_selector

原理：
1. 加载V2的完整模型（包含CLIP+LoRA + pointer_selector）
2. 替换pointer_selector的权重为V3训练的权重
3. 用V2的推理方式（实时编码图像）来评估

这样确保V3用的embedding和V2完全一致，实现公平对比。
"""
import argparse
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy

# 导入根目录的utils.py
import importlib.util
_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
vqa_postprocess = _root_utils.vqa_postprocess
init_lever_lm = _root_utils.init_lever_lm
load_ds = _root_utils.load_ds

from lever_lm.utils import init_interface


def load_v2_model_with_v3_pointer(v2_ckpt_path, v3_ckpt_path, cfg, device):
    """
    加载V2模型，然后替换pointer_selector为V3的权重
    """
    print(f"加载V2模型: {v2_ckpt_path}")
    
    # 手动加载检查点，避免CUDA设备映射问题
    checkpoint = torch.load(v2_ckpt_path, map_location='cpu', weights_only=False)
    
    # 初始化模型
    lever_lm = hydra.utils.instantiate(cfg.train.lever_lm)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("lever_lm.", ""): v for k, v in state_dict.items()}
    lever_lm.load_state_dict(state_dict, strict=False)
    
    # 获取processor
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained(cfg.train.lever_lm.clip_name)
    
    lever_lm = lever_lm.to(device)
    
    if v3_ckpt_path and v3_ckpt_path != "none":
        print(f"加载V3 pointer_selector权重: {v3_ckpt_path}")
        v3_ckpt = torch.load(v3_ckpt_path, map_location='cpu', weights_only=True)
        
        # 处理不同格式的checkpoint
        if 'model_state_dict' in v3_ckpt:
            v3_ckpt = v3_ckpt['model_state_dict']
            print("  (从model_state_dict中提取权重)")
    else:
        print("使用V2原版pointer_selector（不替换）")
        return lever_lm, processor
    
    # 获取V2模型中pointer_selector的当前状态
    v2_ps_state = lever_lm.pointer_selector.state_dict()
    
    # 检查V3权重是否与V2 pointer_selector兼容
    v3_keys = set(v3_ckpt.keys())
    v2_keys = set(v2_ps_state.keys())
    
    common_keys = v3_keys & v2_keys
    missing_in_v3 = v2_keys - v3_keys
    extra_in_v3 = v3_keys - v2_keys
    
    print(f"  共同的键: {len(common_keys)}")
    if missing_in_v3:
        print(f"  V3中缺少的键: {missing_in_v3}")
    if extra_in_v3:
        print(f"  V3中多余的键: {extra_in_v3}")
    
    # 只加载共同的键
    filtered_v3_ckpt = {k: v for k, v in v3_ckpt.items() if k in common_keys}
    lever_lm.pointer_selector.load_state_dict(filtered_v3_ckpt, strict=False)
    
    print("✓ V3 pointer_selector权重加载完成")
    
    return lever_lm, processor


def main():
    parser = argparse.ArgumentParser(description="用V2的CLIP+LoRA评估V3的pointer_selector")
    parser.add_argument("--v2_ckpt", type=str, required=True, help="V2检查点路径")
    parser.add_argument("--v3_ckpt", type=str, required=True, help="V3 pointer_selector检查点路径")
    parser.add_argument("--dataset", type=str, default="okvqa", choices=["okvqa", "vqav2"])
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 构建配置（模拟hydra配置）
    with hydra.initialize(config_path="../configs", version_base=None):
        if args.dataset == "okvqa":
            cfg = hydra.compose(
                config_name="inference",
                overrides=[
                    "train=query_img_text_icd_img_text_v2",
                    "dataset=okvqa_local",
                    "task=vqa",
                    f"device={args.device}",
                    "infer_model=qwen2.5_vl_3B",
                ]
            )
        else:
            cfg = hydra.compose(
                config_name="inference",
                overrides=[
                    "train=query_img_text_icd_img_text_v2",
                    "dataset=vqav2_local",
                    "task=vqa",
                    f"device={args.device}",
                    "infer_model=qwen2.5_vl_3B",
                ]
            )
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    ds = load_ds(cfg)
    train_ds = ds["train"]
    test_ds = ds["validation"]
    print(f"✓ 数据集加载完成: 训练集 {len(train_ds)}, 测试集 {len(test_ds)}")
    
    # 加载模型
    lever_lm, processor = load_v2_model_with_v3_pointer(
        args.v2_ckpt, args.v3_ckpt, cfg, device
    )
    lever_lm.eval()
    
    # 初始化VLM接口
    print(f"初始化VLM: {args.model_name}")
    interface = init_interface(cfg, device=args.device)
    print(f"✓ VLM加载完成")
    
    # 构建retriever
    from open_mmicl.retriever import LeverLMRetriever
    retriever = LeverLMRetriever(
        train_ds,
        test_ds.select(range(min(args.test_num, len(test_ds)))),
        lever_lm=lever_lm,
        processor=processor,
        query_image_field=cfg.train.lever_lm_ds.query_image_field,
        query_text_field=cfg.train.lever_lm_ds.query_text_field,
        icd_image_field=cfg.train.lever_lm_ds.icd_image_field,
        icd_text_field=cfg.train.lever_lm_ds.icd_text_field,
        device=args.device,
        infer_batch_size=1,
        infer_num_workers=0,
        reverse_seq=False,
    )
    
    # 检索ICD
    print(f"\n开始检索ICD (shot_num={args.shot_num}, test_num={args.test_num})...")
    icd_idx_list = retriever.retrieve(args.shot_num)
    print(f"✓ 检索完成，共{len(icd_idx_list)}个样本")
    
    # VQA推理
    print("\n开始VQA推理...")
    test_subset = test_ds.select(range(min(args.test_num, len(test_ds))))
    
    results = []
    for i, (sample, ice_idx) in enumerate(tqdm(zip(test_subset, icd_idx_list), 
                                                  total=len(icd_idx_list),
                                                  desc="VQA推理中")):
        # 准备ICD样本
        ice_sample_list = [train_ds[idx] for idx in ice_idx]
        
        # 构建prompt
        data_sample_list = ice_sample_list + [sample]
        prompts = interface.transfer_prompts([data_sample_list], is_last_for_generation=True)
        input_dict = interface.prepare_input(prompts, is_last_for_generation=True)
        
        # 处理 BatchFeature 对象
        if hasattr(input_dict, 'data'):
            input_dict = dict(input_dict.data)
        elif not isinstance(input_dict, dict):
            input_dict = dict(input_dict)
        
        # 将数据移动到设备
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
               for k, v in input_dict.items()}
        
        # 处理 Qwen2.5-VL 的特殊情况
        if 'image_grid_thw' in data:
            if data['image_grid_thw'].dim() == 3 and data['image_grid_thw'].shape[0] == 1:
                data['image_grid_thw'] = data['image_grid_thw'].squeeze(0)
        
        # 生成答案
        with torch.inference_mode():
            outputs = interface.generate(
                **data,
                eos_token_id=interface.tokenizer.eos_token_id,
                pad_token_id=interface.tokenizer.pad_token_id,
                max_new_tokens=10,
                do_sample=False,
            )
        
        # 解码
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.tolist()
        prompt_len = int(data["attention_mask"].shape[1])
        output_text = interface.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        pred = vqa_postprocess(output_text, args.model_name)
        
        results.append({
            "question_id": sample.get("question_id", i),
            "answer": pred,
        })
    
    # 保存结果到临时文件
    import tempfile
    import json
    result_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(results, result_file)
    result_file.close()
    
    # 计算VQA准确率
    if args.dataset == "okvqa":
        val_ques_path = "datasets/okvqa/OpenEnded_mscoco_val2014_questions.json"
        val_ann_path = "datasets/okvqa/mscoco_val2014_annotations.json"
    else:
        val_ques_path = "datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
        val_ann_path = "datasets/vqav2/v2_mscoco_val2014_annotations.json"
    
    accuracy = compute_vqa_accuracy(
        result_json_path=result_file.name,
        question_json_path=val_ques_path,
        annotation_json_path=val_ann_path,
    )
    
    # 清理临时文件
    os.unlink(result_file.name)
    
    print("\n" + "=" * 70)
    print(f"V3 (使用V2的CLIP+LoRA) VQA准确率: {accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
