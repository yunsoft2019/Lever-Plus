#!/usr/bin/env python3
"""
对比两个 checkpoint 选择的范例索引（ICD indices）
用于检查 KL_BETA=0.12 和 0.15 的模型是否选择了相同的范例
"""

import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lever_lm.utils import init_interface
from open_mmicl.retriever.lever_lm_retriever import LeverLMRetriever
from omegaconf import DictConfig, OmegaConf

# 导入根目录的utils.py（避免与lever_lm/utils/冲突）
import importlib.util
import os as _os
_utils_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
init_lever_lm = _root_utils.init_lever_lm
get_lever_lm_path = _root_utils.get_lever_lm_path
load_ds = _root_utils.load_ds


def load_checkpoint_and_get_indices(
    ckpt_path: str,
    dataset_name: str = "okvqa_local",
    test_num: int = 800,
    shot_num: int = 1,
    device: str = "cuda:0",
    seed: int = 42,
):
    """加载 checkpoint 并获取选择的范例索引"""
    
    # 设置随机种子
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 加载数据集
    print(f"加载数据集: {dataset_name}")
    # 从环境变量读取路径
    from dotenv import load_dotenv
    load_dotenv()
    
    # 创建临时配置用于加载数据集
    temp_cfg = DictConfig({
        'task': {'task_name': 'vqa'},
        'dataset': {
            'name': dataset_name,
            'version': 'local',
            'train_path': os.path.join(os.environ.get('OKVQA_PATH', ''), 'okvqa_hf/vqav2_mscoco_train2014.json'),
            'val_path': os.path.join(os.environ.get('OKVQA_PATH', ''), 'okvqa_hf/vqav2_mscoco_val2014.json'),
            'train_coco_dataset_root': os.path.join(os.environ.get('COCO_PATH', ''), 'mscoco2014/train2014'),
            'val_coco_dataset_root': os.path.join(os.environ.get('COCO_PATH', ''), 'mscoco2014/val2014'),
        },
    })
    ds = load_ds(temp_cfg)
    
    # 限制测试集大小
    if test_num > 0 and test_num < len(ds["validation"]):
        test_indices = list(range(test_num))
        test_ds = ds["validation"].select(test_indices)
    else:
        test_ds = ds["validation"]
    
    print(f"测试集大小: {len(test_ds)}")
    
    # 创建配置
    cfg = DictConfig({
        'train': {
            'lever_lm_ds': {
                'query_image_field': 'image',
                'query_text_field': 'question',
                'icd_image_field': 'image',
                'icd_text_field': 'question',
            },
            'lever_lm': {
                '_target_': 'lever_lm.models.v2.adapter_builder.build_model_v2_with_adapter',
                'config': {
                    '_target_': 'lever_lm.models.v2.pointer_selector_v2.PointerSelectorV2Config',
                    'd_model': 512,
                    'K': 2,
                    'shot_num': 2,
                    'label_smoothing': 0.0,
                    'dropout': 0.5,
                    'hidden_dim': 256,
                    'num_heads': 1,
                    'attn_dropout': 0.1,
                    'num_layers': 1,
                },
                'clip_name': 'openai/clip-vit-base-patch32',
                'query_encoding_flag': ['image', 'text'],
                'icd_encoding_flag': ['image', 'text'],
                'adapter': False,
                'norm': True,
            },
        },
        'device': device,
        'lever_lm_bs': 1,
        'lever_lm_num_workers': 0,
        'reverse_seq': False,
    })
    
    # 设置 checkpoint 路径
    os.environ['LEVER_LM_CHECKPOINT_PATH'] = ckpt_path
    
    # 加载模型
    print(f"加载 checkpoint: {ckpt_path}")
    lever_lm, processor = init_lever_lm(cfg, lever_lm_path=ckpt_path)
    
    # 创建 retriever
    retriever = LeverLMRetriever(
        index_ds=ds["train"],
        test_ds=test_ds,
        lever_lm=lever_lm,
        processor=processor,
        query_image_field=cfg.train.lever_lm_ds.query_image_field,
        query_text_field=cfg.train.lever_lm_ds.query_text_field,
        icd_image_field=cfg.train.lever_lm_ds.icd_image_field,
        icd_text_field=cfg.train.lever_lm_ds.icd_text_field,
        device=device,
        infer_batch_size=cfg.lever_lm_bs,
        infer_num_workers=cfg.lever_lm_num_workers,
        reverse_seq=cfg.reverse_seq,
    )
    
    # 获取范例索引
    print(f"获取范例索引 (shot_num={shot_num})...")
    icd_idx_list = retriever.retrieve(shot_num)
    
    return icd_idx_list


def compare_indices(indices1, indices2, name1="Model 1", name2="Model 2"):
    """对比两个索引列表"""
    
    if len(indices1) != len(indices2):
        print(f"⚠️  警告: 索引列表长度不同 ({len(indices1)} vs {len(indices2)})")
        min_len = min(len(indices1), len(indices2))
        indices1 = indices1[:min_len]
        indices2 = indices2[:min_len]
    
    total_samples = len(indices1)
    identical_samples = 0
    different_samples = 0
    
    # 统计每个 shot_num 的情况
    shot_num = len(indices1[0]) if indices1 else 0
    identical_by_position = [0] * shot_num
    
    different_examples = []
    
    for idx, (icd1, icd2) in enumerate(zip(indices1, indices2)):
        if icd1 == icd2:
            identical_samples += 1
            for pos in range(len(icd1)):
                if pos < len(icd2) and icd1[pos] == icd2[pos]:
                    identical_by_position[pos] += 1
        else:
            different_samples += 1
            if len(different_examples) < 10:  # 只保存前10个不同的例子
                different_examples.append({
                    'sample_idx': idx,
                    f'{name1}': icd1,
                    f'{name2}': icd2,
                })
    
    print("\n" + "=" * 60)
    print("范例索引对比结果")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"完全相同的样本: {identical_samples} ({identical_samples/total_samples*100:.2f}%)")
    print(f"不同的样本: {different_samples} ({different_samples/total_samples*100:.2f}%)")
    
    if shot_num > 0:
        print(f"\n按位置统计 (shot_num={shot_num}):")
        for pos in range(shot_num):
            identical_rate = identical_by_position[pos] / total_samples * 100
            print(f"  位置 {pos+1}: {identical_by_position[pos]}/{total_samples} ({identical_rate:.2f}%)")
    
    if different_examples:
        print(f"\n前 {len(different_examples)} 个不同的例子:")
        for ex in different_examples:
            print(f"  样本 {ex['sample_idx']}:")
            print(f"    {name1}: {ex[name1]}")
            print(f"    {name2}: {ex[name2]}")
    
    print("=" * 60)
    
    return {
        'total_samples': total_samples,
        'identical_samples': identical_samples,
        'different_samples': different_samples,
        'identical_rate': identical_samples / total_samples if total_samples > 0 else 0,
        'identical_by_position': identical_by_position,
        'different_examples': different_examples,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="对比两个 checkpoint 选择的范例索引")
    parser.add_argument("--ckpt1", type=str, required=True, help="第一个 checkpoint 路径")
    parser.add_argument("--ckpt2", type=str, required=True, help="第二个 checkpoint 路径")
    parser.add_argument("--name1", type=str, default="KL_BETA=0.12", help="第一个模型名称")
    parser.add_argument("--name2", type=str, default="KL_BETA=0.15", help="第二个模型名称")
    parser.add_argument("--test_num", type=int, default=800, help="测试样本数")
    parser.add_argument("--shot_num", type=int, default=1, help="范例数量")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("范例索引对比工具")
    print("=" * 60)
    print(f"模型1 ({args.name1}): {args.ckpt1}")
    print(f"模型2 ({args.name2}): {args.ckpt2}")
    print(f"测试样本数: {args.test_num}")
    print(f"范例数量: {args.shot_num}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 获取第一个模型的索引
    print(f"\n[1/2] 获取 {args.name1} 的范例索引...")
    indices1 = load_checkpoint_and_get_indices(
        args.ckpt1,
        test_num=args.test_num,
        shot_num=args.shot_num,
        device=args.device,
        seed=args.seed,
    )
    
    # 获取第二个模型的索引
    print(f"\n[2/2] 获取 {args.name2} 的范例索引...")
    indices2 = load_checkpoint_and_get_indices(
        args.ckpt2,
        test_num=args.test_num,
        shot_num=args.shot_num,
        device=args.device,
        seed=args.seed,
    )
    
    # 对比索引
    print(f"\n对比索引...")
    result = compare_indices(indices1, indices2, args.name1, args.name2)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {args.output}")
    
    return result


if __name__ == "__main__":
    main()

