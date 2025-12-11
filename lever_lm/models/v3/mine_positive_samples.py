"""
正样本挖掘工具

从旧 RL 数据和 SFT 数据中挖掘对 Qwen2.5-VL 仍然有效的正样本

作者: Lever-Plus Team
日期: 2025-12-12
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch

# 延迟导入，避免循环依赖
def get_vqa_interface(model_name: str, device: str = "cuda:0"):
    """获取 VQA 推理接口"""
    from open_mmicl.interface.qwen2vl_interface import Qwen2VLInterface
    return Qwen2VLInterface(
        model_name=model_name,
        device=device,
    )


def load_okvqa_dataset(cfg_or_path: str = None):
    """加载 OKVQA 数据集"""
    import datasets
    from PIL import Image
    import os
    
    # 从环境变量获取路径
    okvqa_path = os.environ.get("OKVQA_PATH", "/mnt/share/yiyun/Datasets/okvqa")
    coco_path = os.environ.get("COCO_PATH", "/mnt/share/yiyun/Datasets/coco")
    
    train_json = f"{okvqa_path}/okvqa_hf/vqav2_mscoco_train2014.json"
    train_img_dir = f"{coco_path}/mscoco2014/train2014"
    
    with open(train_json, "r") as f:
        data = json.load(f)
    
    # 构建数据集
    ds_list = []
    for item in data:
        img_path = os.path.join(train_img_dir, item["image"])
        ds_list.append({
            "question": item["question"],
            "answer": item["answer"],
            "answers": item.get("answers", [item["answer"]]),
            "image_path": img_path,
            "image_id": item.get("image_id", 0),
            "question_id": item.get("question_id", 0),
        })
    
    return ds_list


def build_vqa_prompt_2shot(
    interface,
    query_image_path: str,
    query_question: str,
    ex1: Dict,
    ex2: Optional[Dict] = None,
) -> str:
    """
    构建 2-shot VQA prompt 并生成答案
    
    Args:
        interface: VQA 推理接口
        query_image_path: query 图像路径
        query_question: query 问题
        ex1: 第一个 ICD 样本
        ex2: 第二个 ICD 样本（可选）
        
    Returns:
        生成的答案
    """
    from PIL import Image
    
    # 构建 few-shot examples
    examples = []
    
    if ex1:
        ex1_img = Image.open(ex1["image_path"]).convert("RGB")
        examples.append({
            "image": ex1_img,
            "question": ex1["question"],
            "answer": ex1["answer"],
        })
    
    if ex2:
        ex2_img = Image.open(ex2["image_path"]).convert("RGB")
        examples.append({
            "image": ex2_img,
            "question": ex2["question"],
            "answer": ex2["answer"],
        })
    
    # 加载 query 图像
    query_img = Image.open(query_image_path).convert("RGB")
    
    # 调用接口生成答案
    pred_answer = interface.generate_with_icl(
        query_image=query_img,
        query_question=query_question,
        examples=examples,
        max_new_tokens=10,
    )
    
    return pred_answer


def compute_vqa_accuracy_simple(
    pred_answer: str,
    ground_truth_answers: List[str],
) -> Tuple[int, float]:
    """
    简单的 VQA 准确率计算（不依赖官方评测文件）
    
    Args:
        pred_answer: 预测答案
        ground_truth_answers: 真实答案列表
        
    Returns:
        (vqa_correct, vqa_acc_score)
    """
    pred = pred_answer.strip().lower()
    
    # 计算匹配数
    match_count = 0
    for gt in ground_truth_answers:
        gt_clean = gt.strip().lower()
        if pred == gt_clean or pred in gt_clean or gt_clean in pred:
            match_count += 1
    
    # VQA 官方公式：min(match_count / 3, 1)
    acc_score = min(match_count / 3.0, 1.0)
    correct = 1 if acc_score >= 0.5 else 0  # 通常 acc >= 0.5 视为正确
    
    return correct, acc_score


def mine_positive_from_old_rl(
    old_rl_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cuda:0",
    max_queries: int = -1,
    skip_existing: bool = True,
):
    """
    从旧 RL 数据中挖掘对 Qwen 仍然有效的正样本
    
    Args:
        old_rl_path: 旧 RL 数据路径
        output_path: 输出路径
        model_name: VQA 模型名称
        device: 设备
        max_queries: 最大处理 query 数（-1 表示全部）
        skip_existing: 是否跳过已存在的输出文件
    """
    if skip_existing and os.path.exists(output_path):
        print(f"输出文件已存在，跳过: {output_path}")
        return
    
    print(f"加载旧 RL 数据: {old_rl_path}")
    with open(old_rl_path, "r") as f:
        old_rl = json.load(f)
    
    print("加载 OKVQA 数据集...")
    ds = load_okvqa_dataset()
    
    print(f"加载 VQA 模型: {model_name}")
    interface = get_vqa_interface(model_name, device)
    
    new_rl = {}
    total_candidates = 0
    kept_pos = 0
    
    query_items = list(old_rl.items())
    if max_queries > 0:
        query_items = query_items[:max_queries]
    
    for qid_str, qinfo in tqdm(query_items, desc="Mining positives from old RL"):
        qid = int(qid_str)
        pcs = qinfo.get("pointer_candidates", [])
        if not pcs:
            # 旧格式：id_list + score_list
            id_list = qinfo.get("id_list", [])
            score_list = qinfo.get("score_list", [])
            if id_list:
                pcs = []
                for i, beam in enumerate(id_list):
                    # beam 格式: [icd1, icd2, query_id]
                    pointer = beam[:-1] if len(beam) > 2 else beam[:2]
                    pcs.append({
                        "pointer": pointer,
                        "beam_score": score_list[i] if i < len(score_list) else 0.0,
                        "gen_method": "beam",
                    })
        
        if not pcs:
            continue
        
        if qid >= len(ds):
            continue
        
        query_item = ds[qid]
        query_image_path = query_item["image_path"]
        query_question = query_item["question"]
        gt_answers = query_item["answers"]
        
        new_pcs = []
        
        for c in pcs:
            ptr = c.get("pointer", [])
            if len(ptr) < 2:
                continue
            
            if ptr[0] >= len(ds) or ptr[1] >= len(ds):
                continue
            
            ex1 = ds[ptr[0]]
            ex2 = ds[ptr[1]]
            
            try:
                pred_answer = build_vqa_prompt_2shot(
                    interface=interface,
                    query_image_path=query_image_path,
                    query_question=query_question,
                    ex1=ex1,
                    ex2=ex2,
                )
                
                correct, acc = compute_vqa_accuracy_simple(
                    pred_answer=pred_answer,
                    ground_truth_answers=gt_answers,
                )
            except Exception as e:
                print(f"Error processing qid={qid}, ptr={ptr}: {e}")
                continue
            
            total_candidates += 1
            
            if correct == 1:
                kept_pos += 1
                new_pcs.append({
                    "pointer": ptr,
                    "beam_score": c.get("beam_score"),
                    "logprob_score": c.get("logprob_score"),
                    "gen_method": c.get("gen_method", "mined_from_old"),
                    "vqa_pred_answer": pred_answer,
                    "vqa_correct": correct,
                    "vqa_acc_score": acc,
                    "vqa_eval_mode": "simple_match",
                })
        
        if new_pcs:
            new_rl[qid_str] = {"pointer_candidates": new_pcs}
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(new_rl, f, indent=2, ensure_ascii=False)
    
    print(f"\n从旧 RL 数据中挖掘到的 Qwen 正样本:")
    print(f"  - 处理候选数: {total_candidates}")
    print(f"  - 保留正样本数: {kept_pos}")
    print(f"  - 正样本比例: {kept_pos / max(total_candidates, 1) * 100:.3f}%")
    print(f"  - 有正样本的 Query 数: {len(new_rl)}")
    print(f"  - 输出文件: {output_path}")


def merge_rl_data(
    base_rl_path: str,
    mined_paths: List[str],
    output_path: str,
):
    """
    合并多个 RL 数据文件
    
    Args:
        base_rl_path: 基础 RL 数据路径
        mined_paths: 挖掘的正样本数据路径列表
        output_path: 输出路径
    """
    print(f"加载基础 RL 数据: {base_rl_path}")
    with open(base_rl_path, "r") as f:
        base_rl = json.load(f)
    
    mined_data_list = []
    for path in mined_paths:
        if os.path.exists(path):
            print(f"加载挖掘数据: {path}")
            with open(path, "r") as f:
                mined_data_list.append(json.load(f))
        else:
            print(f"警告: 文件不存在，跳过: {path}")
    
    merged = {}
    
    # 收集所有 query ID
    all_qids = set(base_rl.keys())
    for mined in mined_data_list:
        all_qids.update(mined.keys())
    
    total_added = 0
    
    for qid in all_qids:
        pcs = []
        ptr_set = set()
        
        # 1) 基础 RL 数据的候选
        if qid in base_rl:
            for c in base_rl[qid].get("pointer_candidates", []):
                ptr_tuple = tuple(c.get("pointer", []))
                if ptr_tuple and ptr_tuple not in ptr_set:
                    pcs.append(c)
                    ptr_set.add(ptr_tuple)
        
        # 2) 挖掘的正样本（避免重复）
        for mined in mined_data_list:
            if qid in mined:
                for c in mined[qid].get("pointer_candidates", []):
                    ptr_tuple = tuple(c.get("pointer", []))
                    if ptr_tuple and ptr_tuple not in ptr_set:
                        pcs.append(c)
                        ptr_set.add(ptr_tuple)
                        total_added += 1
        
        if pcs:
            merged[qid] = {"pointer_candidates": pcs}
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f"\n合并结果:")
    print(f"  - 合并后 Query 数: {len(merged)}")
    print(f"  - 新增候选数: {total_added}")
    print(f"  - 输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="正样本挖掘工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 子命令: mine_old
    mine_old_parser = subparsers.add_parser("mine_old", help="从旧 RL 数据挖掘正样本")
    mine_old_parser.add_argument("--old_rl", type=str, required=True, help="旧 RL 数据路径")
    mine_old_parser.add_argument("--output", type=str, required=True, help="输出路径")
    mine_old_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    mine_old_parser.add_argument("--device", type=str, default="cuda:0")
    mine_old_parser.add_argument("--max_queries", type=int, default=-1)
    
    # 子命令: merge
    merge_parser = subparsers.add_parser("merge", help="合并 RL 数据")
    merge_parser.add_argument("--base_rl", type=str, required=True, help="基础 RL 数据路径")
    merge_parser.add_argument("--mined", type=str, nargs="+", required=True, help="挖掘的正样本数据路径")
    merge_parser.add_argument("--output", type=str, required=True, help="输出路径")
    
    args = parser.parse_args()
    
    if args.command == "mine_old":
        mine_positive_from_old_rl(
            old_rl_path=args.old_rl,
            output_path=args.output,
            model_name=args.model,
            device=args.device,
            max_queries=args.max_queries,
        )
    elif args.command == "merge":
        merge_rl_data(
            base_rl_path=args.base_rl,
            mined_paths=args.mined,
            output_path=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
