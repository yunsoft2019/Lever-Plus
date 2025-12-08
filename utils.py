import os
from typing import Dict, List, Optional, Union

import hydra
import more_itertools
import torch
from loguru import logger
from transformers import AutoProcessor

from lever_lm.load_ds_utils import load_coco_ds, load_hf_ds, load_vqav2_ds
from open_mmicl.interface import FlamingoInterface, IDEFICSInterface, LLMInterface, Qwen2VLInterface
from open_mmicl.metrics.cider_calculator import compute_cider
from open_mmicl.metrics.vqa_metrics import postprocess_vqa_generation


def load_ds(cfg, split=None):
    if cfg.task.task_name == "caption":
        ds = load_coco_ds(
            name=cfg.dataset.name,
            train_coco_dataset_root=cfg.dataset.train_coco_dataset_root,
            train_coco_annotation_file=cfg.dataset.train_coco_annotation_file,
            val_coco_dataset_root=cfg.dataset.val_coco_dataset_root,
            val_coco_annotation_file=cfg.dataset.val_coco_annotation_file,
            karpathy_path=(
                cfg.dataset.karpathy_path
                if hasattr(cfg.dataset, "karpathy_path")
                else None
            ),
            split=split,
        )
    elif cfg.task.task_name == "vqa":
        ds = load_vqav2_ds(
            version=cfg.dataset.version,
            train_path=cfg.dataset.train_path,
            val_path=cfg.dataset.val_path,
            train_coco_dataset_root=cfg.dataset.train_coco_dataset_root,
            val_coco_dataset_root=cfg.dataset.val_coco_dataset_root,
            split=split,
        )
    else:
        try:
            ds = load_hf_ds(cfg.dataset.hf_ds_name, split=split)
        except Exception as e:
            raise ValueError(f"dataset load fail with error: {e}")
    return ds


@torch.inference_mode()
def get_info_score(
    interface: Union[FlamingoInterface, IDEFICSInterface, LLMInterface, Qwen2VLInterface],
    choosed_icd_seq_list: List,
    candidate_set: Dict,
    batch_size: int,
    split_token: Optional[str] = None,
    construct_order="left",
):
    # Qwen2.5-VL doesn't support batch processing well due to:
    # 1. Variable image sizes across samples
    # 2. Different number of images per sample (different ICD counts)
    # 3. High memory usage
    # 使用PEFT模型直接推理时，内存占用与原始模型相近，可以使用正常的batch_size
    if isinstance(interface, Qwen2VLInterface):
        # 可以通过环境变量 BATCH_SIZE_OVERRIDE 来覆盖默认值
        import os
        override_batch_size = os.getenv('BATCH_SIZE_OVERRIDE')
        if override_batch_size:
            try:
                max_batch_size = int(override_batch_size)
                batch_size = min(batch_size, max_batch_size)
                logger.info(f"Using batch_size={batch_size} for Qwen2.5-VL (overridden from env)")
            except ValueError:
                batch_size = min(batch_size, 4)  # 默认值
        else:
            # 使用PEFT模型直接推理时，可以使用正常的batch_size（4）
            # 因为PEFT模型的内存占用与原始模型相近
            batch_size = min(batch_size, 4)
        logger.debug(f"Using batch_size={batch_size} for Qwen2.5-VL")
    
    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    kwargs = dict(add_image_token=True)
    if isinstance(interface, LLMInterface):
        kwargs = dict()
    test_lang_x_input = interface.gen_text_with_label(
        choosed_icd_seq_list[-1], **kwargs
    )
    prompts = interface.transfer_prompts(
        choosed_icd_seq_list, is_last_for_generation=False
    )

    x_input = interface.prepare_input(
        prompts, is_last_for_generation=False, add_eos_token=True
    ).to(interface.device)

    icd_mask_prompt = interface.concat_prompt(
        choosed_icd_seq_list[:-1],
        add_eos_token=False,
        is_last_for_generation=False,
        **kwargs,
    )
    query_mask_part = test_lang_x_input.split(split_token)[0] + split_token

    mask_context = icd_mask_prompt + query_mask_part

    mask_length = interface.get_input_token_num(mask_context)
    cond_prob = interface.get_cond_prob(x_input, mask_length=[mask_length])

    # 2. 计算P(y|x, c)
    info_score_list = []
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]

        # 2.1 拼接文本输入
        if construct_order == "left":
            add_new_icd_seq_list = [
                [new_icd] + choosed_icd_seq_list for new_icd in batch_data
            ]
        elif construct_order == "right":
            add_new_icd_seq_list = [
                choosed_icd_seq_list[:-1] + [new_icd] + [choosed_icd_seq_list[-1]]
                for new_icd in batch_data
            ]
        else:
            raise ValueError(
                f"the construct_order should be left or right, but got {construct_order}"
            )

        prompts = interface.transfer_prompts(
            add_new_icd_seq_list, is_last_for_generation=False
        )

        add_new_icd_input = interface.prepare_input(
            prompts,
            is_last_for_generation=False,
            add_eos_token=True,
        )
        # Qwen2.5-VL 的 prepare_input 返回 dict，已经包含 device 信息
        # Flamingo 的 prepare_input 返回 tensor，需要移动到 device
        if isinstance(add_new_icd_input, dict):
            # 如果是 dict，确保所有 tensor 都在正确的 device 上
            for key in add_new_icd_input.keys():
                if isinstance(add_new_icd_input[key], torch.Tensor):
                    add_new_icd_input[key] = add_new_icd_input[key].to(interface.device)
        elif isinstance(add_new_icd_input, torch.Tensor):
            # 如果是 tensor，直接移动到 device
            add_new_icd_input = add_new_icd_input.to(interface.device)
        icd_mask_prompt_list = [
            interface.concat_prompt(
                t[:-1],
                add_eos_token=False,
                is_last_for_generation=False,
                **kwargs,
            )
            for t in add_new_icd_seq_list
        ]

        mask_context_list = [
            icd_mask_prompt + query_mask_part
            for icd_mask_prompt in icd_mask_prompt_list
        ]

        mask_length_list = [
            interface.get_input_token_num(mask_context)
            for mask_context in mask_context_list
        ]
        new_cond_prob = interface.get_cond_prob(
            add_new_icd_input, mask_length=mask_length_list
        )
        sub_info_score = new_cond_prob - cond_prob
        
        # 注意：Qwen和Flamingo的计算方式完全相同，但结果符号不同
        # 这是因为模型行为不同：
        # - Flamingo：添加ICD后损失减少，new_cond_prob > cond_prob，分数为正数
        # - Qwen：添加ICD后损失增加，new_cond_prob < cond_prob，分数为负数
        # 
        # 对于Qwen的负数分数：
        # - 束搜索时，使用 topk(k, largest=False) 选择最小的k个（即绝对值最小的，最接近0的）
        # - 这样会选择 Beam 1（-9.29e-08），这是正确的
        # - 不需要转换分数，保持原始负数分数即可
        
        info_score_list.append(sub_info_score)
        
        # Clear GPU cache periodically to prevent OOM
        # This is especially important for Qwen2.5-VL which uses more memory
        # 减少清理频率以提升性能，只在必要时清理
        if isinstance(interface, Qwen2VLInterface):
            # 每20个batch清理一次缓存，减少性能影响
            # 如果内存充足，可以进一步减少清理频率
            if len(info_score_list) % 20 == 0 and len(info_score_list) > 0:
                torch.cuda.empty_cache()
    return torch.cat(info_score_list)


@torch.inference_mode()
def get_cider_score(
    interface,
    choosed_icd_seq_list: List,
    candidate_set: Dict,
    batch_size: int,
    model_name: str,
    train_ann_path: str,
    construct_order="left",
    gen_kwargs: Dict = None,
):
    output_dict = {}

    prompts = interface.transfer_prompts(
        choosed_icd_seq_list, is_last_for_generation=True
    )

    x_input = interface.prepare_input(
        prompts, is_last_for_generation=True, add_eos_token=True
    ).to(interface.device)

    origin_outputs = interface.generate(
        **x_input,
        pad_token_id=interface.tokenizer.pad_token_id,
        eos_token_id=interface.tokenizer.eos_token_id,
        **gen_kwargs,
    )

    origin_outputs = origin_outputs.tolist()
    prompt_len = int(x_input["attention_mask"].shape[1])

    generated = interface.tokenizer.batch_decode(
        [output[prompt_len:] for output in origin_outputs],
        skip_special_tokens=True,
    )
    pred_coco = [
        {"image_id": choosed_icd_seq_list[-1]["image_id"], "caption": generated[0]}
    ]

    origin_cider_score = compute_cider(pred_coco, train_ann_path, reduce_cider=False)
    origin_cider_score = origin_cider_score[choosed_icd_seq_list[-1]["image_id"]][
        "CIDEr"
    ]

    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]
        if construct_order == "left":
            add_new_icd_seq_list = [
                [new_icd] + choosed_icd_seq_list for new_icd in batch_data
            ]
        elif construct_order == "right":
            add_new_icd_seq_list = [
                choosed_icd_seq_list[:-1] + [new_icd] + [choosed_icd_seq_list[-1]]
                for new_icd in batch_data
            ]
        else:
            raise ValueError(
                f"the construct_order should be left or right, but got {construct_order}"
            )
        prompts = interface.transfer_prompts(
            add_new_icd_seq_list, is_last_for_generation=True
        )
        add_new_icd_input = interface.prepare_input(
            prompts,
            is_last_for_generation=True,
            add_eos_token=True,
        ).to(interface.device)

        outputs = interface.generate(
            **add_new_icd_input,
            pad_token_id=interface.tokenizer.pad_token_id,
            eos_token_id=interface.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        outputs = outputs.tolist()
        prompt_len = int(add_new_icd_input["attention_mask"].shape[1])

        generated = interface.tokenizer.batch_decode(
            [output[prompt_len:] for output in outputs],
            skip_special_tokens=True,
        )
        for i, data in enumerate(batch_data):
            output_dict[data["idx"]] = {}
            output_dict[data["idx"]]["prediction"] = generated[i]
            output_dict[data["idx"]]["image_id"] = data["image_id"]

    pred_coco = []
    for idx in output_dict:
        pred_coco.append(
            {
                "image_id": output_dict[idx]["image_id"],
                "caption": caption_postprocess(
                    output_dict[idx]["prediction"], model_name=model_name
                ),
            }
        )
    cider_score_info = compute_cider(pred_coco, train_ann_path, reduce_cider=False)
    cider_score = []
    for idx in cand_idx:
        img_id = candidate_set[idx]["image_id"]
        cider_score.append(cider_score_info[img_id]["CIDEr"])

    return torch.tensor(cider_score) - origin_cider_score


def caption_postprocess(text, model_name):
    if "flamingo" in model_name:
        return text.split("Output", 1)[0].replace('"', "")
    elif "idefics" in model_name:
        return text.split("Caption", 1)[0].replace('"', "").replace("\n", "")


def vqa_postprocess(text, model_name):
    if "flamingo" in model_name:
        return postprocess_vqa_generation(text)
    elif "idefics" in model_name:
        return postprocess_vqa_generation(text).replace("\n", "")
    elif "qwen" in model_name.lower() or "Qwen" in model_name:
        # Qwen2.5-VL uses the same postprocessing as flamingo
        return postprocess_vqa_generation(text)
    else:
        # Default: use postprocess_vqa_generation for unknown models
        return postprocess_vqa_generation(text)


def parse_checkpoint_filename(checkpoint_path):
    """从检查点文件名中解析训练参数
    
    Args:
        checkpoint_path: 检查点文件路径
        
    Returns:
        dict: 包含 sampler_name 和 training_params 的字典
    """
    import re
    checkpoint_filename = os.path.basename(checkpoint_path)
    logger.debug(f"解析检查点文件名: {checkpoint_filename}")
    
    # 检查点文件名格式: {model_name}_{sampler_name}_infoscore_left_beam5_shot2_cand64_sample{sample_num}[_...].ckpt
    # 例如: flamingo_3B_RandSampler_infoscore_left_beam5_shot2_cand64_sample800_best_epoch=...
    
    # 提取 sampler_name (在第一个下划线和 infoscore 之间)
    match = re.search(r'_(RandSampler|TextSimSampler|ImgSimSampler|MixSampler)_infoscore', checkpoint_filename)
    sampler_name = match.group(1) if match else None
    
    # 提取训练参数部分 (从 infoscore 开始到 sample{num} 结束)
    match = re.search(r'(infoscore_left_beam\d+_shot\d+_cand\d+_sample\d+)', checkpoint_filename)
    training_params = match.group(1) if match else None
    
    if sampler_name:
        logger.debug(f"解析到 sampler_name: {sampler_name}")
    else:
        logger.warning(f"无法从检查点文件名中解析 sampler_name: {checkpoint_filename}")
    
    if training_params:
        logger.debug(f"解析到 training_params: {training_params}")
    else:
        logger.warning(f"无法从检查点文件名中解析 training_params: {checkpoint_filename}")
    
    return {
        'sampler_name': sampler_name,
        'training_params': training_params,
    }


def get_lever_lm_path(cfg):
    # 优先从环境变量读取（避免 Hydra 解析路径中的特殊字符）
    lever_lm_path = os.environ.get('LEVER_LM_CHECKPOINT_PATH', None)
    
    if lever_lm_path:
        logger.info(f"从环境变量读取检查点路径: {lever_lm_path}")
        return lever_lm_path
    
    if cfg.lever_lm_path is None:
        # 尝试多个可能的目录路径
        possible_dirs = [
            # 格式1: results/{dataset.name}/model_cpk/
            os.path.join(cfg.result_dir, cfg.dataset.name, "model_cpk"),
            # 格式2: results/model_cpk/{task.task_name}/{ex_name}
            os.path.join(cfg.result_dir, "model_cpk", cfg.task.task_name, cfg.ex_name),
            # 格式3: results/model_cpk/{ex_name}
            os.path.join(cfg.result_dir, "model_cpk", cfg.ex_name),
        ]
        
        cpk_list = []
        for cpk_dir in possible_dirs:
            if os.path.exists(cpk_dir) and os.path.isdir(cpk_dir):
                logger.info(f"尝试在目录中查找检查点: {cpk_dir}")
                try:
                    for f in os.listdir(cpk_dir):
                        if f.endswith('.ckpt'):
                            cpk_list.append(os.path.join(cpk_dir, f))
                except Exception as e:
                    logger.warning(f"无法读取目录 {cpk_dir}: {e}")
                    continue
        
        if cpk_list:
            # 根据 default_cpk_key 过滤（如果指定了）
            if cfg.default_cpk_key and cfg.default_cpk_key != "last":
                filtered_list = list(filter(lambda x: cfg.default_cpk_key in x, cpk_list))
                if filtered_list:
                    cpk_list = filtered_list
            
            # 选择最新的检查点（如果 default_cpk_key 是 "last"）
            if cfg.default_cpk_key == "last" or not cfg.default_cpk_key:
                cpk_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            lever_lm_path = cpk_list[0]
            logger.info(f"找到检查点: {lever_lm_path}")
            return lever_lm_path
        else:
            error_msg = f"未找到检查点文件。尝试过的目录:\n"
            for cpk_dir in possible_dirs:
                error_msg += f"  - {cpk_dir}\n"
            raise ValueError(error_msg)
    else:
        lever_lm_path = cfg.lever_lm_path
    return lever_lm_path


def init_lever_lm(cfg, lever_lm_path):
    # 检查是否是 v3 GRPO checkpoint（.pt 格式）
    import os
    is_v3_checkpoint = (
        os.getenv("LEVER_LM_CHECKPOINT_VERSION") == "v3" or
        lever_lm_path.endswith(".pt") and ("grpo" in lever_lm_path or "rce" in lever_lm_path)
    )
    
    if is_v3_checkpoint:
        # v3 GRPO checkpoint 使用特殊的加载方式
        logger.info(f"检测到 v3 GRPO checkpoint，使用 v3 加载方式: {lever_lm_path}")
        from lever_lm.models.v3 import load_v3_from_grpo_checkpoint
        from lever_lm.models.adapter import PointerSelectorAdapter
        
        # 获取设备
        device = cfg.device if hasattr(cfg, 'device') and cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        
        # 加载 v3 模型（返回 PointerSelectorV3）
        pointer_selector = load_v3_from_grpo_checkpoint(lever_lm_path, device=device)
        
        # 获取配置参数
        clip_name = cfg.train.lever_lm.clip_name
        
        # 从配置中获取 encoding flags
        query_encoding_flag = cfg.train.lever_lm.get('query_encoding_flag', ['image', 'text'])
        if isinstance(query_encoding_flag, (list, tuple)) and len(query_encoding_flag) > 0:
            # 已经是列表格式，直接使用
            pass
        else:
            # 尝试从 lever_lm_ds 推断
            query_encoding_flag = []
            if hasattr(cfg.train, 'lever_lm_ds') and cfg.train.lever_lm_ds.get('query_image_field'):
                query_encoding_flag.append('image')
            if hasattr(cfg.train, 'lever_lm_ds') and cfg.train.lever_lm_ds.get('query_text_field'):
                query_encoding_flag.append('text')
            if not query_encoding_flag:
                query_encoding_flag = ['image', 'text']  # 默认值
        
        icd_encoding_flag = cfg.train.lever_lm.get('icd_encoding_flag', ['image', 'text'])
        if isinstance(icd_encoding_flag, (list, tuple)) and len(icd_encoding_flag) > 0:
            # 已经是列表格式，直接使用
            pass
        else:
            # 尝试从 lever_lm_ds 推断
            icd_encoding_flag = []
            if hasattr(cfg.train, 'lever_lm_ds') and cfg.train.lever_lm_ds.get('icd_image_field'):
                icd_encoding_flag.append('image')
            if hasattr(cfg.train, 'lever_lm_ds') and cfg.train.lever_lm_ds.get('icd_text_field'):
                icd_encoding_flag.append('text')
            if not icd_encoding_flag:
                icd_encoding_flag = ['image', 'text']  # 默认值
        
        adapter = cfg.train.lever_lm.get('adapter', False)
        norm = cfg.train.lever_lm.get('norm', True)
        K = pointer_selector.K
        
        # 包装为 PointerSelectorAdapter（提供 generation 方法）
        lever_lm = PointerSelectorAdapter(
            pointer_selector_model=pointer_selector,
            clip_name=clip_name,
            query_encoding_flag=query_encoding_flag,
            icd_encoding_flag=icd_encoding_flag,
            adapter=adapter,
            norm=norm,
            K=K,
            device=device
        )
        
        processor = AutoProcessor.from_pretrained(clip_name)
        return lever_lm, processor
    
    # v0, v1, v2, v2_lora 使用 PyTorch Lightning 格式的 checkpoint
    # PyTorch 2.6+ 默认 weights_only=True，需要设置为 False 来加载包含 omegaconf 对象的检查点
    checkpoint = torch.load(lever_lm_path, weights_only=False)
    
    # 从检查点中读取保存的超参数，特别是 index_ds_size
    # 这样可以确保模型大小与检查点匹配
    saved_index_ds_size = None
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if "cfg" in hyper_parameters:
        saved_cfg = hyper_parameters["cfg"]
        # 如果检查点中有保存的 dataset.train_ds_len，使用它来设置 index_ds_size
        if "dataset" in saved_cfg and "train_ds_len" in saved_cfg.get("dataset", {}):
            saved_index_ds_size = saved_cfg["dataset"]["train_ds_len"]
            logger.info(f"从检查点超参数读取 index_ds_size: {saved_index_ds_size}")
    
    # 如果从超参数中没有读取到，尝试从权重形状推断
    if saved_index_ds_size is None:
        state_dict = checkpoint["state_dict"]
        # 查找 lm_model.transformer.wte.weight 或 lever_lm.lm_model.transformer.wte.weight
        wte_key = None
        for key in state_dict.keys():
            if "lm_model.transformer.wte.weight" in key:
                wte_key = key
                break
        
        if wte_key is not None:
            wte_shape = state_dict[wte_key].shape
            vocab_size = wte_shape[0]  # 第一个维度是 vocab_size
            # GPT2LeverLM 中：vocab_size = index_ds_size + 3，所以需要减去 3
            saved_index_ds_size = vocab_size - 3
            logger.info(f"从检查点权重形状推断: vocab_size={vocab_size}, index_ds_size={saved_index_ds_size}")
    
    # 如果成功获取到 saved_index_ds_size，检查是否与当前配置匹配
    if saved_index_ds_size is not None:
        # 获取当前配置中的 train_ds_len
        current_train_ds_len = getattr(cfg.dataset, "train_ds_len", None)
        
        # 如果当前配置中有 train_ds_len，检查是否匹配
        if current_train_ds_len is not None and current_train_ds_len != saved_index_ds_size:
            logger.warning(
                f"检查点的 index_ds_size ({saved_index_ds_size}) 与当前配置的 train_ds_len ({current_train_ds_len}) 不匹配！"
                f"将使用检查点的 index_ds_size ({saved_index_ds_size}) 来初始化模型。"
            )
        elif current_train_ds_len is not None and current_train_ds_len == saved_index_ds_size:
            logger.info(f"检查点的 index_ds_size ({saved_index_ds_size}) 与当前配置的 train_ds_len ({current_train_ds_len}) 匹配。")
        
        # 修改配置以匹配检查点
        cfg.train.lever_lm.index_ds_size = saved_index_ds_size
        logger.info(f"使用 index_ds_size: {saved_index_ds_size} 来初始化模型")
    
    # 如果配置中有 device，在实例化模型时传递 device 参数，确保 CLIP 模型加载到正确的 GPU
    # 注意：只有 v1+ 版本的模型（build_model_v1_with_adapter）支持 device 参数
    # v0 版本的 GPT2LeverLM 不支持 device 参数
    instantiate_kwargs = {}
    if hasattr(cfg, 'device') and cfg.device:
        # 检查是否是 v1+ 版本的模型
        lever_lm_target = getattr(cfg.train.lever_lm, '_target_', '')
        if 'build_model_v1_with_adapter' in lever_lm_target or 'v1' in lever_lm_target or 'v2' in lever_lm_target or 'v3' in lever_lm_target or 'v4' in lever_lm_target:
            instantiate_kwargs['device'] = cfg.device
            logger.info(f"将在设备 {cfg.device} 上初始化模型（包括 CLIP 模型）")
        else:
            logger.info(f"v0 模型不支持 device 参数，将在模型加载后移动到设备 {cfg.device}")
    
    lever_lm = hydra.utils.instantiate(cfg.train.lever_lm, **instantiate_kwargs)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("lever_lm.", ""): v for k, v in state_dict.items()}
    
    # 检查是否有缺失的键（用于兼容旧检查点）
    model_state_dict = lever_lm.state_dict()
    missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    
    # 如果缺少 query_update_weight（新添加的参数），使用默认值初始化
    if "pointer_selector.query_update_weight" in missing_keys:
        logger.warning("检查点缺少 'query_update_weight' 参数，将使用默认值 0.6 初始化")
        # 这个参数已经在模型初始化时设置了默认值，所以不需要额外处理
    
    # 使用 strict=False 允许缺失的键（新添加的参数会使用默认值）
    lever_lm.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.info(f"缺失的键（将使用默认值）: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"意外的键（将被忽略）: {unexpected_keys}")
    
    # 如果配置中有 device，将模型移动到该设备（确保所有子模块包括 CLIP 模型都在正确的设备上）
    if hasattr(cfg, 'device') and cfg.device:
        device = cfg.device
        lever_lm = lever_lm.to(device)
        logger.info(f"已将模型移动到设备: {device}")
    
    processor = AutoProcessor.from_pretrained(cfg.train.lever_lm.clip_name)
    return lever_lm, processor
