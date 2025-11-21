import json
import os
import uuid

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from lever_lm.utils import init_interface
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy
from open_mmicl.retriever import LeverLMRetriever
from utils import get_lever_lm_path, init_lever_lm, load_ds, vqa_postprocess


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    # 设置日志级别为 INFO，过滤掉 DEBUG 日志，避免干扰进度条显示
    import sys
    logger.remove()  # 移除默认的 handler
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}")  # 只显示 INFO 及以上级别
    
    # 打印关键配置信息
    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info(f"  任务类型 (cfg.task.task_name): {cfg.task.task_name}")
    logger.info(f"  数据集名称 (cfg.dataset.name): {cfg.dataset.name}")
    logger.info(f"  数据集版本 (cfg.dataset.version): {cfg.dataset.get('version', 'N/A')}")
    logger.info(f"  验证集路径 (cfg.dataset.val_path): {cfg.dataset.get('val_path', 'N/A')}")
    logger.info("=" * 60)
    
    # 第一步：遍历测试集
    # 加载数据集
    logger.info("开始加载数据集...")
    logger.info(f"load_ds 将根据 cfg.task.task_name='{cfg.task.task_name}' 来选择加载函数")
    
    ds = load_ds(cfg)
    logger.info(f"数据集加载完成，数据集键: {list(ds.keys())}")
    
    # 获取测试集
    test_ds = ds["validation"]
    logger.info(f"测试集字段: {test_ds.column_names}")
    
    # 检查是否有不应该存在的字段
    if "captions" in test_ds.column_names or "single_caption" in test_ds.column_names:
        logger.warning("⚠️  警告：测试集中包含 caption 相关字段！")
        logger.warning(f"   这不应该出现在 VQA 数据集中。")
        logger.warning(f"   当前 task_name: {cfg.task.task_name}")
        logger.warning(f"   可能加载了错误的数据集类型！")
    
    # 推理全部测试样本
    logger.info(f"测试集总样本数: {len(test_ds)}")
    
    # 加载推理模型（用于生成预测）
    logger.info("=" * 60)
    logger.info("开始加载推理模型...")
    logger.info(f"  模型名称: {cfg.infer_model.name}")
    logger.info(f"  设备: {cfg.device}")
    logger.info(f"  精度: {cfg.precision}")
    interface = init_interface(cfg, device=cfg.device)
    logger.info("推理模型加载完成")
    logger.info("=" * 60)
    
    # 加载 SFT 模型（用于选择范例）
    lever_lm = None
    processor = None
    retriever = None
    if cfg.test_lever_lm or (hasattr(cfg, 'lever_lm_path') and cfg.lever_lm_path is not None):
        logger.info("=" * 60)
        logger.info("开始加载 SFT 模型（LeverLM）...")
        lever_lm_path = get_lever_lm_path(cfg)
        logger.info(f"  检查点路径: {lever_lm_path}")
        lever_lm, processor = init_lever_lm(cfg, lever_lm_path=lever_lm_path)
        logger.info("SFT 模型加载完成")
        
        # 初始化 LeverLMRetriever
        logger.info("初始化 LeverLMRetriever...")
        retriever = LeverLMRetriever(
            index_ds=ds["train"],
            test_ds=test_ds,
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
        logger.info("LeverLMRetriever 初始化完成")
        logger.info("=" * 60)
    else:
        logger.warning("⚠️  未配置 SFT 模型，无法检索范例！")
        logger.warning("   请设置 test_lever_lm=true 或 lever_lm_path 来启用范例检索")
    
    # 设置范例数量（shot_num）
    shot_num = cfg.shot_num_list[0] if cfg.shot_num_list else 2
    logger.info(f"使用范例数量: {shot_num}")
    
    # 获取所有测试样本的范例列表（一次性检索）
    icd_idx_list = None
    if retriever is not None:
        logger.info("=" * 60)
        logger.info(f"开始使用 SFT 模型检索范例（shot_num={shot_num}）...")
        icd_idx_list = retriever.retrieve(shot_num)
        logger.info(f"范例检索完成，共 {len(icd_idx_list)} 个测试样本的范例列表")
        logger.info("=" * 60)
    
    # 遍历测试集，将messages输入模型并获取answer
    logger.info(f"开始推理测试集，共 {len(test_ds)} 个样本")
    logger.info("=" * 60)
    
    # 获取生成参数
    generation_kwargs = cfg.task.gen_args if hasattr(cfg.task, 'gen_args') else {}
    
    # 存储所有预测结果
    preds = []
    
    train_ds = ds["train"]
    # 使用进度条显示推理进度
    for idx, sample in enumerate(tqdm(test_ds, desc="推理中", ncols=100)):
        # 使用 SFT 模型获取范例列表
        if icd_idx_list is not None and idx < len(icd_idx_list):
            example_indices = icd_idx_list[idx]
            
            # 从训练集中获取范例数据
            ice_sample_list = []
            for ex_idx in example_indices:
                if ex_idx < len(train_ds):
                    ice_sample_list.append(train_ds[ex_idx])
                else:
                    logger.warning(f"    警告：范例索引 {ex_idx} 超出训练集范围（训练集大小: {len(train_ds)}）")
            
            # 将范例和测试样本组合成数据样本列表
            data_sample_list = ice_sample_list + [sample]
            
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
            
            # 获取prompt长度
            prompt_len = int(data["attention_mask"].shape[1])
            
            # 模型生成（image_nums 会在 base_interface.generate 中处理）
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
            
            # 调试：只在预测为空时警告（不打印每条记录）
            if not prediction or prediction.strip() == "":
                logger.debug(f"样本 {idx} 原始预测为空")
            
            # 后处理得到answer
            answer = vqa_postprocess(prediction, model_name=cfg.infer_model.name)
            
            # 保存预测结果（用于计算准确率）
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
    
    # 计算准确率
    if len(preds) > 0 and cfg.task.task_name == "vqa":
        logger.info("=" * 60)
        logger.info("开始计算准确率...")
        
        # 检查配置文件中的路径
        val_ques_path = cfg.dataset.get('val_ques_path', None)
        val_ann_path = cfg.dataset.get('val_ann_path', None)
        
        if val_ques_path and val_ann_path:
            # 保存预测结果为临时JSON文件
            random_uuid = str(uuid.uuid4())
            temp_result_file = f"{random_uuid}.json"
            
            with open(temp_result_file, "w") as f:
                json.dump(preds, f, indent=4)
            
            logger.info(f"预测结果已保存到临时文件: {temp_result_file}")
            logger.info(f"问题文件路径: {val_ques_path}")
            logger.info(f"标注文件路径: {val_ann_path}")
            
            # 计算准确率
            try:
                accuracy = compute_vqa_accuracy(temp_result_file, val_ques_path, val_ann_path)
                logger.info("=" * 60)
                # compute_vqa_accuracy 返回的是小数形式（0-1之间），需要乘以100转换为百分比
                # 但如果值大于1，说明已经是百分比形式了
                if accuracy > 1:
                    accuracy_percent = accuracy
                    accuracy_decimal = accuracy / 100
                else:
                    accuracy_decimal = accuracy
                    accuracy_percent = accuracy * 100
                logger.info(f"✅ VQA 准确率: {accuracy_decimal:.4f} ({accuracy_percent:.2f}%)")
                logger.info(f"   预测样本数: {len(preds)}")
                logger.info("=" * 60)
            except Exception as e:
                logger.error(f"计算准确率时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                # 删除临时文件
                if os.path.exists(temp_result_file):
                    os.remove(temp_result_file)
                    logger.debug(f"已删除临时文件: {temp_result_file}")
        else:
            logger.warning("⚠️  缺少 val_ques_path 或 val_ann_path，无法计算准确率")
            logger.warning(f"   val_ques_path: {val_ques_path}")
            logger.warning(f"   val_ann_path: {val_ann_path}")
    elif len(preds) == 0:
        logger.warning("⚠️  没有有效的预测结果，无法计算准确率")
    elif cfg.task.task_name != "vqa":
        logger.info(f"任务类型为 {cfg.task.task_name}，跳过 VQA 准确率计算")


if __name__ == "__main__":
    load_dotenv()
    main()

