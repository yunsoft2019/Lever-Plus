import datetime
import json
import os
import random
import uuid

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoProcessor

from lever_lm.utils import init_interface
from open_mmicl.icl_inferencer import ICLInferecer
from open_mmicl.metrics.cider_calculator import compute_cider
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy
from open_mmicl.retriever import *
from utils import (
    caption_postprocess,
    get_lever_lm_path,
    init_lever_lm,
    load_ds,
    parse_checkpoint_filename,
    vqa_postprocess,
)


def record(result_json_path: str, new_data: dict):
    recorded_data = {}
    if os.path.exists(result_json_path):
        with open(result_json_path, "r") as f:
            recorded_data = json.load(f)

    with open(result_json_path, "w") as f:
        recorded_data.update(new_data)
        json.dump(recorded_data, f, indent=4)


def evaluate_retriever(
    retriever_name,
    inferencer,
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
        icd_idx_list = retriever.retrieve(shot_num)

        if cfg.task.task_name == "caption":
            metric = inference_caption(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                val_ann_path=cfg.dataset.val_coco_annotation_file,
                output_json_filename=output_files,
                model_name=cfg.infer_model.name,
            )
        elif cfg.task.task_name == "vqa":
            metric = inference_vqa(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                val_ques_path=cfg.dataset.val_ques_path,
                val_ann_path=cfg.dataset.val_ann_path,
                output_json_filename=output_files,
                model_name=cfg.infer_model.name,
            )
        elif cfg.task.task_name == "sst2":
            metric = inference_cls(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                output_json_filename=output_files,
            )
        retriever_res[f"{shot_num=}"] = metric
        logger.info(f"{output_files}: {metric=}")
        record(result_json_path, {info: retriever_res})


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


def inference_caption(
    inferencer,
    ds,
    icd_idx_list,
    val_ann_path,
    output_json_filename,
    model_name,
):
    output_dict = inferencer.inference(
        train_ds=ds["train"],
        test_ds=ds["validation"],
        ice_idx_list=icd_idx_list,
        output_json_filename=output_json_filename,
    )
    pred_coco = []
    for idx in output_dict:
        pred_coco.append(
            {
                "image_id": output_dict[idx]["image_id"],
                "caption": caption_postprocess(
                    output_dict[idx]["prediction"], model_name
                ),
            }
        )
    cider_score = compute_cider(pred_coco, val_ann_path)
    return cider_score * 100


def inference_vqa(
    inferencer,
    ds,
    icd_idx_list,
    val_ques_path,
    val_ann_path,
    output_json_filename,
    model_name,
):
    output_dict = inferencer.inference(
        train_ds=ds["train"],
        test_ds=ds["validation"],
        ice_idx_list=icd_idx_list,
        output_json_filename=output_json_filename,
    )
    preds = []
    for idx in output_dict:
        preds.append(
            {
                "answer": vqa_postprocess(
                    output_dict[idx]["prediction"], model_name=model_name
                ),
                "question_id": output_dict[idx]["question_id"],
            }
        )
    random_uuid = str(uuid.uuid4())

    with open(f"{random_uuid}.json", "w") as f:
        f.write(json.dumps(preds, indent=4))
    acc = compute_vqa_accuracy(f"{random_uuid}.json", val_ques_path, val_ann_path)
    # delete the temporary file
    os.remove(f"{random_uuid}.json")
    return acc


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    logger.info(f"{cfg=}")
    
    # 构建新的输出路径格式: results/{数据集名}/icl_inference/{束搜索模型}_{选择器名}_infoscore_left_beam5_shot2_cand64_sample800_metrics.json
    # 如果使用 LeverLMRetriever，从检查点文件名中提取信息
    if cfg.test_lever_lm:
        try:
            lever_lm_path = get_lever_lm_path(cfg)
            checkpoint_info = parse_checkpoint_filename(lever_lm_path)
            
            # 获取数据集名（去掉 _local 后缀）
            dataset_name = cfg.dataset.name.replace('_local', '')
            
            # 获取束搜索模型名
            infer_model_name = cfg.infer_model.name
            
            # 获取选择器名和训练参数
            sampler_name = checkpoint_info.get('sampler_name')
            training_params = checkpoint_info.get('training_params')
            
            if sampler_name and training_params:
                # 构建新格式的路径
                result_filename = f"{infer_model_name}_{sampler_name}_{training_params}_metrics.json"
                result_dir = os.path.join(
                    cfg.result_dir,
                    dataset_name,
                    "icl_inference",
                )
                result_json_path = os.path.join(result_dir, result_filename)
                logger.info(f"使用新路径格式: {result_json_path}")
            else:
                # 如果无法解析检查点文件名，使用旧格式
                logger.warning(f"无法从检查点文件名解析信息，使用旧路径格式")
                result_dir = os.path.join(
                    cfg.result_dir,
                    "icl_inference",
                    cfg.infer_model.name,
                    cfg.task.task_name,
                    cfg.ex_name,
                )
                result_json_path = os.path.join(result_dir, "metrics.json")
        except Exception as e:
            # 如果获取检查点路径失败，使用旧格式
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
        # 非 LeverLMRetriever 使用旧格式
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

    ds = load_ds(cfg)

    if index_data_num != -1:
        ds["train"] = ds["train"].select(
            random.sample(range(len(ds["train"])), index_data_num)
        )
    if test_data_num != -1:
        ds["validation"] = ds["validation"].select(range(test_data_num))

    interface = init_interface(cfg, device=cfg.device)

    inferencer = ICLInferecer(
        interface=interface,
        train_ds=ds["train"],
        test_ds=ds["validation"],
        generation_kwargs=cfg.task.gen_args,
        other_save_field=cfg.task.other_save_field,
        num_workers=cfg.num_workers,
        num_proc=cfg.num_proc,
        batch_size=cfg.inference_bs,
        output_json_filepath=os.path.join(result_dir, "generation_metainfo"),
    )

    base_info = f"{str(datetime.datetime.now())}-{test_data_num=}-"

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

    # Test for other
    for retriever_name, shot_nums in retriever_list:
        if shot_nums:  # Only initialize and evaluate if shot_nums is not empty
            retriever_instance = init_retriever(retriever_name, ds, cfg)
            evaluate_retriever(
                retriever_name,
                inferencer,
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
