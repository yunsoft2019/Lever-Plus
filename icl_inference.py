import datetime
import json
import os
import random
import sys
import uuid

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoProcessor

from lever_lm.utils import init_interface
from open_mmicl.metrics.cider_calculator import compute_cider
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy
from open_mmicl.retriever import *

# 导入根目录的utils.py（避免与lever_lm/utils/冲突）
import importlib.util
import os as _os
_utils_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'utils.py')
_spec = importlib.util.spec_from_file_location("root_utils", _utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
caption_postprocess = _root_utils.caption_postprocess
get_lever_lm_path = _root_utils.get_lever_lm_path
init_lever_lm = _root_utils.init_lever_lm
load_ds = _root_utils.load_ds
parse_checkpoint_filename = _root_utils.parse_checkpoint_filename
vqa_postprocess = _root_utils.vqa_postprocess


def record(result_json_path: str, new_data: dict):
    recorded_data = {}
    if os.path.exists(result_json_path):
        with open(result_json_path, "r") as f:
            recorded_data = json.load(f)

    with open(result_json_path, "w") as f:
        recorded_data.update(new_data)
        json.dump(recorded_data, f, indent=4)


def inference_cls(
    inferencer,
    ds,
    icd_idx_list,
    output_json_filename,
):
    output_dict = inferencer.ppl_inference(
        ds["train"],
        ds["validation"],
        icd_idx_list,
        output_json_filename=output_json_filename,
    )
    predictions = [v["prediction"] for k, v in output_dict.items()]
    targets = ds["validation"]["label"]
    metrics = {}

    # 计算并存储准确率
    metrics["accuracy"] = accuracy_score(targets, predictions)

    # 计算并存储宏平均和加权平均精确率
    metrics["precision_macro"] = precision_score(targets, predictions, average="macro")

    # 计算并存储宏平均和加权平均召回率
    metrics["recall_macro"] = recall_score(targets, predictions, average="macro")

    # 计算并存储宏平均和加权平均F1分数
    metrics["f1_macro"] = f1_score(targets, predictions, average="macro")
    return metrics["accuracy"]


def init_retriever(retriever_name, ds, cfg):
    if retriever_name == "ZeroShot":
        return ZeroRetriever(ds["train"], ds["validation"])
    elif retriever_name == "RandomRetriever":
        return RandRetriever(
            ds["train"],
            ds["validation"],
            seed=cfg.seed,
            fixed=cfg.random_retrieval_fixed,
        )
    elif retriever_name.startswith("MMTopKRetriever"):
        mode = retriever_name.split("-")[-1]
        index_field = (
            cfg.task.icd_text_feature_field
            if mode.endswith("t")
            else cfg.task.image_field
        )
        test_field = (
            cfg.task.image_field
            if mode.startswith("i")
            else cfg.task.icd_text_feature_field
        )

        cache_file = os.path.join(
            cfg.result_dir,
            "cache",
            f'{cfg.task.task_name}-{cfg.dataset.name}-{cfg.mmtopk_clip_name.split("/")[-1]}-{mode}-'
            f"index_field:{index_field}-test_data_num:{cfg.test_data_num}-"
            f"test_field:{test_field}-emb_cache.pth",
        )
        return MMTopkRetriever(
            ds["train"],
            ds["validation"],
            mode=mode,
            index_field=index_field,
            test_field=test_field,
            clip_model_name=cfg.mmtopk_clip_name,
            cache_file=cache_file,
            reversed_order=cfg.mmtopk_reversed_order,
            batch_size=32,
            num_workers=8,
        )
    elif retriever_name == "LeverLMRetriever":
        lever_lm_path = get_lever_lm_path(cfg)
        lever_lm, processor = init_lever_lm(cfg, lever_lm_path=lever_lm_path)
        return LeverLMRetriever(
            ds["train"],
            ds["validation"],
            lever_lm=lever_lm,
            processor=processor,
            query_image_field=cfg.train.lever_lm_ds.query_image_field,
            query_text_field=cfg.train.lever_lm_ds.query_text_field,
            icd_image_field=cfg.train.lever_lm_ds.icd_image_field,
            icd_text_field=cfg.train.lever_lm_ds.icd_text_field,
            device=cfg.device,
            infer_batch_size=cfg.lever_lm_bs,
            infer_num_workers=cfg.lever_lm_num_workers,
            reverse_seq=cfg.reverse_seq,
        )

    return None


def inference_vqa_direct(
    interface,
    train_ds,
    test_ds,
    icd_idx_list,
    val_ques_path,
    val_ann_path,
    model_name,
    generation_kwargs,
):
    """
    直接推理VQA任务，按照步骤：
    1. 遍历测试集
    2. 根据范例id找到范例数据
    3. 包装messages（区分Flamingo和Qwen2.5-VL）
    4. 输入模型得到答案
    5. 计算准确率
    """
    preds = []
    
    # 遍历测试集
    for idx, sample in enumerate(tqdm(test_ds, desc="推理中", ncols=100)):
        if icd_idx_list is not None and idx < len(icd_idx_list):
            example_indices = icd_idx_list[idx]
            
            # 步骤4：根据范例id，找到范例的图片，问题，答案
            ice_sample_list = []
            for ex_idx in example_indices:
                if ex_idx < len(train_ds):
                    ice_sample_list.append(train_ds[ex_idx])
                else:
                    logger.warning(f"警告：范例索引 {ex_idx} 超出训练集范围（训练集大小: {len(train_ds)}）")
            
            # 将范例和测试样本组合
            data_sample_list = ice_sample_list + [sample]
            
            # 步骤5：包装messages（区分Flamingo和Qwen2.5-VL）
            # 使用transfer_prompts转换为prompt格式
            prompts = interface.transfer_prompts(
                [data_sample_list], is_last_for_generation=True
            )
            
            # 使用prepare_input转换为messages格式（tensor）
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
                    # 移除batch维度
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
            
            # 步骤6：把messages输入推理模型，得到推理答案
            prompt_len = int(data["attention_mask"].shape[1])
            
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
            
            # 确保outputs是列表格式
            if not isinstance(outputs, list):
                outputs = [outputs]
            if len(outputs) > 0 and not isinstance(outputs[0], list):
                outputs = [outputs]
            
            # 解码：只取prompt之后的部分
            generated = interface.tokenizer.batch_decode(
                [output[prompt_len:] for output in outputs],
                skip_special_tokens=True,
            )
            
            # 后处理得到answer
            prediction = generated[0] if generated else ""
            answer = vqa_postprocess(prediction, model_name=model_name)
            
            # 保存预测结果
            question_id = sample.get('question_id', None)
            if question_id is not None:
                preds.append({
                    "answer": answer,
                    "question_id": question_id,
                })
            else:
                logger.warning(f"样本 {idx}: 缺少 question_id，无法用于计算准确率")
        else:
            logger.warning(f"样本 {idx}: 无法获取 ICDs 列表（icd_idx_list 为空或索引超出范围）")
    
    # 步骤7：根据推理答案计算准确率
    if len(preds) > 0:
        random_uuid = str(uuid.uuid4())
        temp_result_file = f"{random_uuid}.json"
        
        with open(temp_result_file, "w") as f:
            json.dump(preds, f, indent=4)
        
        try:
            accuracy = compute_vqa_accuracy(temp_result_file, val_ques_path, val_ann_path)
            # 处理准确率格式
            if accuracy > 1:
                accuracy_percent = accuracy
                accuracy_decimal = accuracy / 100
            else:
                accuracy_decimal = accuracy
                accuracy_percent = accuracy * 100
            return accuracy_decimal
        finally:
            if os.path.exists(temp_result_file):
                os.remove(temp_result_file)
    else:
        logger.warning("没有有效的预测结果，无法计算准确率")
        return 0.0


def inference_caption_direct(
    interface,
    train_ds,
    test_ds,
    icd_idx_list,
    val_ann_path,
    model_name,
    generation_kwargs,
):
    """
    直接推理Caption任务
    """
    pred_coco = []
    
    # 遍历测试集
    for idx, sample in enumerate(tqdm(test_ds, desc="推理中", ncols=100)):
        if icd_idx_list is not None and idx < len(icd_idx_list):
            example_indices = icd_idx_list[idx]
            
            # 根据范例id找到范例数据
            ice_sample_list = []
            for ex_idx in example_indices:
                if ex_idx < len(train_ds):
                    ice_sample_list.append(train_ds[ex_idx])
            
            # 将范例和测试样本组合
            data_sample_list = ice_sample_list + [sample]
            
            # 包装messages
            prompts = interface.transfer_prompts(
                [data_sample_list], is_last_for_generation=True
            )
            
            input_dict = interface.prepare_input(
                prompts, is_last_for_generation=True
            )
            
            # 处理 BatchFeature 对象
            if hasattr(input_dict, 'data'):
                input_dict = dict(input_dict.data)
            elif not isinstance(input_dict, dict):
                input_dict = dict(input_dict)
            
            # 将数据移动到设备
            data = {k: v.to(interface.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in input_dict.items()}
            
            # 处理 Qwen2.5-VL 的特殊情况
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
            
            # 模型生成
            prompt_len = int(data["attention_mask"].shape[1])
            
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
            
            if not isinstance(outputs, list):
                outputs = [outputs]
            if len(outputs) > 0 and not isinstance(outputs[0], list):
                outputs = [outputs]
            
            generated = interface.tokenizer.batch_decode(
                [output[prompt_len:] for output in outputs],
                skip_special_tokens=True,
            )
            
            # 后处理得到caption
            prediction = generated[0] if generated else ""
            caption = caption_postprocess(prediction, model_name)
            
            image_id = sample.get('image_id', None)
            if image_id is not None:
                pred_coco.append({
                    "image_id": image_id,
                    "caption": caption,
                })
    
    # 计算CIDEr分数
    if len(pred_coco) > 0:
        cider_score = compute_cider(pred_coco, val_ann_path)
        return cider_score * 100
    else:
        return 0.0


def evaluate_retriever(
    retriever_name,
    interface,
    retriever,
    ds,
    base_info,
    shot_num_list,
    result_json_path,
    cfg,
):
    retriever_res = {}
    info = base_info + retriever_name
    
    for shot_num in shot_num_list:
        logger.info(
            f"Now begin test {cfg.task.task_name}: {retriever_name} with {shot_num=}"
        )
        output_files = info + f"-bs:{cfg.inference_bs}-{shot_num=}"
        
        # 步骤3：根据测试集的图片和问题，通过SFT进行预测范例id
        icd_idx_list = retriever.retrieve(shot_num)
        
        # 获取生成参数
        generation_kwargs = cfg.task.gen_args if hasattr(cfg.task, 'gen_args') else {}
        
        if cfg.task.task_name == "caption":
            metric = inference_caption_direct(
                interface=interface,
                train_ds=ds["train"],
                test_ds=ds["validation"],
                icd_idx_list=icd_idx_list,
                val_ann_path=cfg.dataset.val_coco_annotation_file,
                model_name=cfg.infer_model.name,
                generation_kwargs=generation_kwargs,
            )
        elif cfg.task.task_name == "vqa":
            metric = inference_vqa_direct(
                interface=interface,
                train_ds=ds["train"],
                test_ds=ds["validation"],
                icd_idx_list=icd_idx_list,
                val_ques_path=cfg.dataset.val_ques_path,
                val_ann_path=cfg.dataset.val_ann_path,
                model_name=cfg.infer_model.name,
                generation_kwargs=generation_kwargs,
            )
        elif cfg.task.task_name == "sst2":
            # SST2任务仍使用原来的inferencer方式
            from open_mmicl.icl_inferencer import ICLInferecer
            inferencer = ICLInferecer(
                interface=interface,
                train_ds=ds["train"],
                test_ds=ds["validation"],
                generation_kwargs=generation_kwargs,
                other_save_field=cfg.task.other_save_field,
                num_workers=cfg.num_workers,
                num_proc=cfg.num_proc,
                batch_size=cfg.inference_bs,
                output_json_filepath=None,
            )
            metric = inference_cls(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                output_json_filename=output_files,
            )
        else:
            logger.error(f"不支持的任务类型: {cfg.task.task_name}")
            continue
        
        retriever_res[f"{shot_num=}"] = metric
        logger.info(f"{output_files}: {metric=}")
        record(result_json_path, {info: retriever_res})


def set_seed(seed: int):
    """设置全局随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置PyTorch的确定性模式（可能会降低性能，但确保可复现性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置为: {seed}")


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    # 设置日志级别为 INFO，过滤掉 DEBUG 日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}")
    
    # 设置全局随机种子（从配置中读取，默认为42）
    seed = cfg.get('seed', 42)
    set_seed(seed)
    
    logger.info(f"{cfg=}")
    
    # 构建结果路径
    if cfg.test_lever_lm:
        try:
            lever_lm_path = get_lever_lm_path(cfg)
            checkpoint_info = parse_checkpoint_filename(lever_lm_path)
            
            # 从检查点路径中提取版本信息
            # 路径格式: ./results/{dataset}/model_cpk/{version}/...
            version = "v0"  # 默认版本
            if lever_lm_path:
                path_parts = lever_lm_path.split('/')
                if 'model_cpk' in path_parts:
                    idx = path_parts.index('model_cpk')
                    if idx + 1 < len(path_parts):
                        potential_version = path_parts[idx + 1]
                        # 检查是否是版本格式（v0, v1, v2, v3, v4）
                        if potential_version.startswith('v') and len(potential_version) <= 3:
                            try:
                                # 验证版本号是否有效（v0-v4）
                                version_num = int(potential_version[1:])
                                if 0 <= version_num <= 4:
                                    version = potential_version
                            except ValueError:
                                pass
            
            dataset_name = cfg.dataset.name.replace('_local', '')
            infer_model_name = cfg.infer_model.name
            sampler_name = checkpoint_info.get('sampler_name')
            training_params = checkpoint_info.get('training_params')
            
            if sampler_name and training_params:
                result_filename = f"{infer_model_name}_{sampler_name}_{training_params}_metrics.json"
                # 添加版本目录：results/{dataset}/icl_inference/{version}/
                result_dir = os.path.join(
                    cfg.result_dir,
                    dataset_name,
                    "icl_inference",
                    version,  # 添加版本目录（v0, v1, v2, v3, v4）
                )
                result_json_path = os.path.join(result_dir, result_filename)
                logger.info(f"使用新路径格式（版本: {version}）: {result_json_path}")
            else:
                logger.info(f"检查点已加载: {lever_lm_path}")
                logger.info(f"检查点文件名格式不符合新规范（v2格式），结果文件将使用旧路径格式保存")
                result_dir = os.path.join(
                    cfg.result_dir,
                    "icl_inference",
                    cfg.infer_model.name,
                    cfg.task.task_name,
                    cfg.ex_name,
                )
                result_json_path = os.path.join(result_dir, "metrics.json")
        except Exception as e:
            logger.warning(f"获取检查点路径失败: {e}，使用旧路径格式")
            result_dir = os.path.join(
                cfg.result_dir,
                "icl_inference",
                cfg.infer_model.name,
                cfg.task.task_name,
                cfg.ex_name,
            )
            result_json_path = os.path.join(result_dir, "metrics.json")
    else:
        result_dir = os.path.join(
            cfg.result_dir,
            "icl_inference",
            cfg.infer_model.name,
            cfg.task.task_name,
            cfg.ex_name,
        )
        result_json_path = os.path.join(result_dir, "metrics.json")
    
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    test_data_num = cfg.test_data_num
    index_data_num = cfg.index_data_num

    # 步骤1：加载SFT模型及模型参数，加载推理模型
    logger.info("=" * 60)
    logger.info("步骤1：加载模型")
    logger.info("=" * 60)
    
    # 加载数据集
    ds = load_ds(cfg)

    if index_data_num != -1:
        ds["train"] = ds["train"].select(
            random.sample(range(len(ds["train"])), index_data_num)
        )
    if test_data_num != -1:
        ds["validation"] = ds["validation"].select(range(test_data_num))

    # 加载推理模型（Flamingo-3B或Qwen2.5-VL）
    logger.info(f"加载推理模型: {cfg.infer_model.name}")
    interface = init_interface(cfg, device=cfg.device)
    logger.info("推理模型加载完成")

    base_info = f"{str(datetime.datetime.now())}-{test_data_num=}-"

    # 步骤2：遍历测试集（在evaluate_retriever中完成）
    retriever_list = [
        ("ZeroShot", [0] if cfg.test_zero_shot else []),
        ("RandomRetriever", cfg.shot_num_list if cfg.test_random else []),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2t',
            cfg.shot_num_list if cfg.test_i2t else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2i',
            cfg.shot_num_list if cfg.test_i2i else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-t2t',
            cfg.shot_num_list if cfg.test_t2t else [],
        ),
        (
            "LeverLMRetriever",
            cfg.shot_num_list if cfg.test_lever_lm else [],
        ),
    ]

    # 测试不同的retriever
    for retriever_name, shot_nums in retriever_list:
        if shot_nums:  # Only initialize and evaluate if shot_nums is not empty
            retriever_instance = init_retriever(retriever_name, ds, cfg)
            evaluate_retriever(
                retriever_name,
                interface,
                retriever_instance,
                ds,
                base_info,
                shot_nums,
                result_json_path,
                cfg,
            )


def shuffle_2d_list(matrix):
    new_matrix = [row.copy() for row in matrix]
    if len(new_matrix[0]) == 1:
        return new_matrix
    for i, row in enumerate(tqdm(new_matrix)):
        while row == matrix[i]:
            random.shuffle(row)
    return new_matrix


if __name__ == "__main__":
    load_dotenv()
    main()
