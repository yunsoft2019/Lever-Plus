"""
导出 query 和 candidate embeddings 用于 RL 数据生成

从 SFT 模型（v2 checkpoint）中导出所有 query 和 candidate 的 embedding，
保存为 query_embeddings.pt 和 candidate_embeddings.pt。

使用方法：
    python -m lever_lm.models.v3.export_embeddings \
        --sft_ckpt <path_to_v2_checkpoint> \
        --dataset okvqa_local \
        --output_dir results/okvqa/cache \
        --device cuda:0

作者: Lever-Plus Team
日期: 2025-12-06
参考: 强化学习.md §4.1
"""

import argparse
import os
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPProcessor

from lever_lm.models.v3.adapter_builder import build_model_v3_with_adapter

# 导入根目录的utils.py
import importlib.util
import os as _os
_utils_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
load_ds = _root_utils.load_ds
init_lever_lm = _root_utils.init_lever_lm


def load_sft_model_from_checkpoint(checkpoint_path: str, device: torch.device, config_path: str = None):
    """
    从 checkpoint 加载 SFT 模型（包含 adapter）
    
    Args:
        checkpoint_path: checkpoint 路径
        device: 设备
        config_path: Hydra 配置文件路径（可选，如果 checkpoint 中没有配置则使用此配置）
    
    Returns:
        model: PointerSelectorAdapter 模型
        processor: CLIPProcessor
    """
    print(f"加载 SFT 模型: {checkpoint_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 尝试从 checkpoint 中获取配置
    cfg = None
    if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
        cfg = checkpoint['hyper_parameters']['cfg']
        # 转换为 DictConfig（如果还不是）
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        print("✓ 从 checkpoint 中读取配置")
    
    # 如果 checkpoint 中没有配置，使用配置文件或默认配置
    if cfg is None:
        if config_path and os.path.exists(config_path):
            print(f"使用配置文件: {config_path}")
            loaded_cfg = OmegaConf.load(config_path)
            # 配置文件可能只包含 lever_lm 部分，需要包装在 train 键下
            if "train" in loaded_cfg:
                cfg = loaded_cfg
            else:
                # 如果配置文件中没有 train 键，将其包装在 train.lever_lm 下
                cfg = OmegaConf.create({
                    "train": {
                        "lever_lm": loaded_cfg
                    }
                })
        else:
            # 使用默认配置（v2 配置）
            print("使用默认 v2 配置")
            cfg = OmegaConf.create({
                "train": {
                    "lever_lm": {
                        "_target_": "lever_lm.models.v2.adapter_builder.build_model_v2_with_adapter",
                        "config": {
                            "_target_": "lever_lm.models.v2.pointer_selector_v2.PointerSelectorV2Config",
                            "d_model": 512,
                            "K": 2,
                            "shot_num": 2,
                            "label_smoothing": 0.0,
                            "dropout": 0.5,
                            "hidden_dim": 256,
                            "num_heads": 1,
                            "attn_dropout": 0.1,
                            "num_layers": 3,
                        },
                        "clip_name": "openai/clip-vit-base-patch32",
                        "query_encoding_flag": ["image", "text"],
                        "icd_encoding_flag": ["image", "text"],
                        "adapter": False,
                        "norm": True,
                    }
                }
            })
    
    # 设置 device（用于模型初始化）
    cfg.device = str(device)
    
    # 构建模型（使用配置中的参数）
    model, processor = init_lever_lm(cfg, checkpoint_path)
    model.eval()
    
    print(f"✓ 模型加载完成")
    return model, processor


def export_embeddings(
    model,
    processor: CLIPProcessor,
    dataset,
    device: torch.device,
    batch_size: int = 32
):
    """
    导出所有 query 和 candidate 的 embedding
    
    Args:
        model: PointerSelectorAdapter 模型
        processor: CLIPProcessor
        dataset: 数据集（包含所有训练样本）
        device: 设备
        batch_size: 批处理大小
    
    Returns:
        query_embeddings: [N, d] 所有 query 的 embedding
        candidate_embeddings: [N, d] 所有 candidate 的 embedding（与 query 相同，因为候选池就是训练集）
    """
    print(f"开始导出 embeddings（数据集大小: {len(dataset)}）...")
    
    all_query_embeddings = []
    all_candidate_embeddings = []
    
    # 批量处理
    for i in tqdm(range(0, len(dataset), batch_size), desc="导出 embeddings"):
        batch_indices = list(range(i, min(i + batch_size, len(dataset))))
        batch_data = [dataset[idx] for idx in batch_indices]
        
        # 准备 query_input（使用 processor 预处理）
        query_inputs = []
        for item in batch_data:
            query_input_dict = {}
            
            # 图像
            images = None
            if "image" in model.query_encoding_flag:
                images = [item.get("image")]
            
            # 文本（问题）
            text = None
            if "text" in model.query_encoding_flag:
                question = item.get("question", "")
                text = [question]
            
            # 使用 processor 预处理
            if images is not None and text is not None:
                processed = processor(images=images, text=text, return_tensors="pt", padding=True)
            elif images is not None:
                processed = processor(images=images, return_tensors="pt")
            elif text is not None:
                processed = processor(text=text, return_tensors="pt", padding=True)
            else:
                raise ValueError("query_encoding_flag 必须包含 'image' 或 'text'")
            
            # 移动到设备
            query_input_dict = {k: v.to(device) for k, v in processed.items()}
            query_inputs.append(query_input_dict)
        
        # 批量编码 query
        batch_query_embeddings = []
        with torch.no_grad():
            for query_input_dict in query_inputs:
                # 使用 adapter 的编码方法
                query_emb = model._extract_query_emb(query_input_dict)  # [1, d]
                batch_query_embeddings.append(query_emb.squeeze(0).cpu())  # [d]
        
        # 对于 candidate，我们使用相同的样本（候选池就是训练集）
        # 所以 candidate_embeddings 与 query_embeddings 相同
        batch_candidate_embeddings = batch_query_embeddings.copy()
        
        all_query_embeddings.extend(batch_query_embeddings)
        all_candidate_embeddings.extend(batch_candidate_embeddings)
    
    # 转换为 tensor
    query_embeddings = torch.stack(all_query_embeddings, dim=0)  # [N, d]
    candidate_embeddings = torch.stack(all_candidate_embeddings, dim=0)  # [N, d]
    
    print(f"✓ Embeddings 导出完成")
    print(f"  - Query embeddings 形状: {query_embeddings.shape}")
    print(f"  - Candidate embeddings 形状: {candidate_embeddings.shape}")
    
    return query_embeddings, candidate_embeddings


def main():
    parser = argparse.ArgumentParser(description="导出 query 和 candidate embeddings")
    parser.add_argument("--sft_ckpt", type=str, required=True, help="SFT 模型 checkpoint 路径")
    parser.add_argument("--dataset", type=str, default="okvqa_local", help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="results/okvqa/cache", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--config", type=str, help="Hydra 配置文件路径（可选，用于数据集和模型配置）")
    parser.add_argument("--train_config", type=str, help="训练配置文件路径（可选，用于模型配置，如 configs/train/lever_lm/v2/query_img_text_icd_img_text_lever_lm.yaml）")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载数据集配置
    if args.config and os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        # 创建默认配置（用于加载数据集）
        # 根据数据集名称推断任务类型和配置
        dataset_name = args.dataset.lower()
        if "okvqa" in dataset_name or "vqa" in dataset_name:
            task_name = "vqa"
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            # 设置默认路径（与 generate_rl_data.py 一致）
            okvqa_dir = os.path.join(project_root, "datasets", "okvqa")
            okvqa_hf_dir = os.path.join(okvqa_dir, "okvqa_hf")
            default_train_path = os.path.join(okvqa_hf_dir, "vqav2_mscoco_train2014.json")
            default_val_path = os.path.join(okvqa_hf_dir, "vqav2_mscoco_val2014.json")
            default_train_coco_root = os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "train2014")
            default_val_coco_root = os.path.join(project_root, "datasets", "mscoco", "mscoco2014", "val2014")
            
            train_path = os.getenv("OKVQA_TRAIN_PATH", default_train_path)
            val_path = os.getenv("OKVQA_VAL_PATH", default_val_path)
            train_coco_root = os.getenv("COCO_TRAIN_ROOT", default_train_coco_root)
            val_coco_root = os.getenv("COCO_VAL_ROOT", default_val_coco_root)
            
            cfg = OmegaConf.create({
                "dataset": {
                    "name": args.dataset,
                    "version": "local",
                    "train_path": train_path,
                    "val_path": val_path,
                    "train_coco_dataset_root": train_coco_root,
                    "val_coco_dataset_root": val_coco_root,
                },
                "task": {
                    "task_name": task_name,
                },
            })
        else:
            # 对于其他任务，使用简单配置
            cfg = OmegaConf.create({
                "dataset": {
                    "name": args.dataset,
                },
                "task": {
                    "task_name": "caption",  # 默认
                },
            })
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    ds = load_ds(cfg)
    train_ds = ds["train"]
    print(f"✓ 训练集大小: {len(train_ds)}")
    
    # 加载 SFT 模型
    model, processor = load_sft_model_from_checkpoint(
        args.sft_ckpt, 
        device, 
        config_path=args.train_config
    )
    
    # 导出 embeddings
    query_embeddings, candidate_embeddings = export_embeddings(
        model=model,
        processor=processor,
        dataset=train_ds,
        device=device,
        batch_size=args.batch_size
    )
    
    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    query_emb_path = os.path.join(args.output_dir, "query_embeddings.pt")
    cand_emb_path = os.path.join(args.output_dir, "candidate_embeddings.pt")
    
    print(f"保存 query embeddings 到: {query_emb_path}")
    torch.save(query_embeddings.cpu(), query_emb_path)
    
    print(f"保存 candidate embeddings 到: {cand_emb_path}")
    torch.save(candidate_embeddings.cpu(), cand_emb_path)
    
    print("✓ 完成！")


if __name__ == "__main__":
    main()
