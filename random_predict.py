import json
import os
import sys
import uuid

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from lever_lm.utils import init_interface
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy
from open_mmicl.retriever import RandRetriever
from utils import load_ds, vqa_postprocess


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


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    # 设置日志级别为 INFO，过滤掉 DEBUG 日志，避免干扰进度条显示
    logger.remove()  # 移除默认的 handler
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}")  # 只显示 INFO 及以上级别
    
    # 打印关键配置信息
    logger.info("=" * 60)
    logger.info("随机范例推理配置信息:")
    logger.info(f"  任务类型 (cfg.task.task_name): {cfg.task.task_name}")
    logger.info(f"  数据集名称 (cfg.dataset.name): {cfg.dataset.name}")
    logger.info(f"  数据集版本 (cfg.dataset.version): {cfg.dataset.get('version', 'N/A')}")
    logger.info(f"  验证集路径 (cfg.dataset.val_path): {cfg.dataset.get('val_path', 'N/A')}")
    logger.info("=" * 60)
    
    # 加载数据集
    logger.info("开始加载数据集...")
    logger.info(f"load_ds 将根据 cfg.task.task_name='{cfg.task.task_name}' 来选择加载函数")
    
    ds = load_ds(cfg)
    logger.info(f"数据集加载完成，数据集键: {list(ds.keys())}")
    
    # 获取测试集和训练集
    test_ds = ds["validation"]
    train_ds = ds["train"]
    logger.info(f"测试集字段: {test_ds.column_names}")
    logger.info(f"测试集总样本数: {len(test_ds)}")
    logger.info(f"训练集总样本数: {len(train_ds)}")
    
    # 检查是否有不应该存在的字段
    if "captions" in test_ds.column_names or "single_caption" in test_ds.column_names:
        logger.warning("⚠️  警告：测试集中包含 caption 相关字段！")
        logger.warning(f"   这不应该出现在 VQA 数据集中。")
        logger.warning(f"   当前 task_name: {cfg.task.task_name}")
        logger.warning(f"   可能加载了错误的数据集类型！")
    
    # 定义要测试的模型和shot数量
    # 注意：只测试当前指定的单个模型
    
    # 从命令行参数中获取 infer_model 的值（配置文件名）
    # sys 已在文件顶部导入，这里直接使用
    infer_model_config = None
    for arg in sys.argv:
        if arg.startswith("infer_model="):
            infer_model_config = arg.split("=", 1)[1]
            break
    
    if not infer_model_config:
        logger.error("错误: 必须指定 infer_model 参数")
        logger.error("用法: python random_predict.py task=vqa dataset=okvqa_local infer_model=flamingo_3B")
        return
    
    # 只测试指定的单个模型
    models_to_test = [infer_model_config]
    logger.info(f"测试模型: {infer_model_config}")
    
    shot_num_list = [1, 2, 3, 4, 6, 8]
    
    # 检查配置文件中的路径
    val_ques_path = cfg.dataset.get('val_ques_path', None)
    val_ann_path = cfg.dataset.get('val_ann_path', None)
    
    if not val_ques_path or not val_ann_path:
        logger.error("⚠️  缺少 val_ques_path 或 val_ann_path，无法计算准确率")
        logger.error(f"   val_ques_path: {val_ques_path}")
        logger.error(f"   val_ann_path: {val_ann_path}")
        return
    
    # 获取生成参数
    generation_kwargs = cfg.task.gen_args if hasattr(cfg.task, 'gen_args') else {}
    
    # 初始化随机检索器
    logger.info("=" * 60)
    logger.info("初始化随机检索器（RandomRetriever）...")
    retriever = RandRetriever(
        train_ds,
        test_ds,
        seed=cfg.get('seed', 42),
        fixed=cfg.get('random_retrieval_fixed', True),
    )
    logger.info("随机检索器初始化完成")
    logger.info("=" * 60)
    
    # 存储所有结果
    results = {}
    
    # 遍历每个模型
    for model_config_name in models_to_test:
        logger.info("=" * 60)
        logger.info(f"开始测试模型配置: {model_config_name}")
        logger.info("=" * 60)
        
        # 保存原始配置
        from omegaconf import OmegaConf
        import os
        original_infer_model = OmegaConf.create(OmegaConf.to_container(cfg.infer_model))
        
        # 加载新的模型配置文件
        config_file = os.path.join("configs", "infer_model", f"{model_config_name}.yaml")
        if not os.path.exists(config_file):
            logger.error(f"配置文件不存在: {config_file}")
            continue
        
        try:
            # 加载新的模型配置
            new_model_config = OmegaConf.load(config_file)
            # 更新当前cfg的infer_model部分
            cfg.infer_model = new_model_config
        except Exception as e:
            logger.error(f"加载模型配置失败: {e}")
            logger.error(f"配置文件: {config_file}")
            continue
        
        # 获取实际的模型名称（用于日志和结果保存）
        actual_model_name = cfg.infer_model.name
        
        # 加载推理模型
        logger.info(f"加载推理模型: {actual_model_name} (配置: {model_config_name})")
        logger.info(f"  设备: {cfg.device}")
        logger.info(f"  精度: {cfg.precision}")
        interface = init_interface(cfg, device=cfg.device)
        logger.info("推理模型加载完成")
        
        # 存储该模型的结果
        model_results = {}
        
        # 提前准备结果文件路径（用于增量保存）
        dataset_name = cfg.dataset.name.replace('_local', '')
        result_dir = os.path.join(
            cfg.result_dir,
            dataset_name,
            "icl_inference",
        )
        os.makedirs(result_dir, exist_ok=True)
        model_name_safe = actual_model_name.replace('.', '_').replace('-', '_').replace('/', '_')
        result_filename = f"{model_name_safe}_RandomRetriever_baseline_metrics.json"
        result_json_path = os.path.join(result_dir, result_filename)
        
        # 如果文件已存在，尝试加载已有结果（支持断点续跑）
        if os.path.exists(result_json_path):
            try:
                with open(result_json_path, "r") as f:
                    existing_results = json.load(f)
                    logger.info(f"发现已有结果文件，加载已有结果: {result_json_path}")
                    model_results = existing_results
            except Exception as e:
                logger.warning(f"加载已有结果失败，将重新开始: {e}")
                model_results = {}
        
        # 遍历每个shot数量
        for shot_num in shot_num_list:
            # 检查该shot_num是否已完成（支持断点续跑）
            shot_key = f"shot_num_{shot_num}"
            if shot_key in model_results:
                logger.info("=" * 60)
                logger.info(f"⏭️  跳过已完成: {actual_model_name} with shot_num={shot_num}")
                logger.info(f"   已有结果: {model_results[shot_key]:.4f} ({model_results[shot_key]*100:.2f}%)")
                logger.info("=" * 60)
                continue
            
            logger.info("=" * 60)
            logger.info(f"开始测试: {actual_model_name} with shot_num={shot_num}")
            logger.info("=" * 60)
            
            # 使用随机检索器获取范例列表
            logger.info(f"使用随机检索器检索范例（shot_num={shot_num}）...")
            icd_idx_list = retriever.retrieve(shot_num)
            logger.info(f"范例检索完成，共 {len(icd_idx_list)} 个测试样本的范例列表")
            
            # 进行推理并计算准确率
            logger.info("开始推理...")
            accuracy = inference_vqa_direct(
                interface=interface,
                train_ds=train_ds,
                test_ds=test_ds,
                icd_idx_list=icd_idx_list,
                val_ques_path=val_ques_path,
                val_ann_path=val_ann_path,
                model_name=actual_model_name,
                generation_kwargs=generation_kwargs,
            )
            
            # 保存结果
            model_results[f"shot_num_{shot_num}"] = accuracy
            accuracy_percent = accuracy * 100 if accuracy <= 1 else accuracy
            
            # 立即保存到文件（增量保存，防止中途崩溃丢失结果）
            logger.info(f"💾 保存结果到文件: {result_json_path}")
            with open(result_json_path, "w") as f:
                json.dump(model_results, f, indent=4)
            logger.info("✅ 结果已保存")
            
            logger.info("=" * 60)
            logger.info(f"✅ {actual_model_name} - shot_num={shot_num}: {accuracy:.4f} ({accuracy_percent:.2f}%)")
            logger.info("=" * 60)
        
        # 恢复原始配置
        cfg.infer_model = original_infer_model
        
        # 保存该模型的所有结果（使用实际模型名称作为key，更易读）
        results[actual_model_name] = model_results
    
    # 收集所有结果文件路径（用于最终汇总显示）
    result_files = {}
    dataset_name = cfg.dataset.name.replace('_local', '')
    result_dir = os.path.join(
        cfg.result_dir,
        dataset_name,
        "icl_inference",
    )
    for model_name, model_results in results.items():
        model_name_safe = model_name.replace('.', '_').replace('-', '_').replace('/', '_')
        result_filename = f"{model_name_safe}_RandomRetriever_baseline_metrics.json"
        result_json_path = os.path.join(result_dir, result_filename)
        result_files[model_name] = result_json_path
    
    logger.info("=" * 60)
    logger.info("结果文件保存位置:")
    for model_name, file_path in result_files.items():
        logger.info(f"  {model_name}: {file_path}")
    logger.info("=" * 60)
    
    # 打印最终结果汇总
    logger.info("=" * 60)
    logger.info("最终结果汇总:")
    logger.info("=" * 60)
    for model_name, model_results in results.items():
        logger.info(f"\n{model_name}:")
        for shot_key, accuracy in sorted(model_results.items()):
            accuracy_percent = accuracy * 100 if accuracy <= 1 else accuracy
            logger.info(f"  {shot_key}: {accuracy:.4f} ({accuracy_percent:.2f}%)")
    logger.info("=" * 60)
    logger.info("✅ 所有结果已保存到 icl_inference 目录")
    logger.info("=" * 60)


if __name__ == "__main__":
    load_dotenv()
    main()

