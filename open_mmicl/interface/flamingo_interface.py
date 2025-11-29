import os
import warnings
from typing import List, Optional

import open_clip
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from open_flamingo.src.factory import _infer_decoder_layers_attr_name
from open_flamingo.src.flamingo import Flamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchFeature

from .base_interface import LVLMInterface

# MPT 模型（mpt-7b, mpt-1b）可能需要 triton_pre_mlir，但这是可选依赖
# 设置环境变量来跳过 transformers 的依赖检查
import os
if "TRANSFORMERS_NO_ADVISORY_WARNINGS" not in os.environ:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 尝试导入 triton_pre_mlir，如果失败则创建一个假的模块来绕过检查
try:
    import triton_pre_mlir
except ImportError:
    # 创建一个假的 triton_pre_mlir 模块来绕过 transformers 的依赖检查
    import sys
    import types
    fake_triton = types.ModuleType('triton_pre_mlir')
    sys.modules['triton_pre_mlir'] = fake_triton
    warnings.warn(
        "triton_pre_mlir not found. Creating a dummy module to bypass dependency check. "
        "Some MPT model features may be unavailable, but this is usually not critical for inference.",
        UserWarning
    )


class FlamingoInterface(LVLMInterface):
    def __init__(
        self,
        lang_encoder_path,
        tokenizer_path,
        flamingo_checkpoint_dir,
        cross_attn_every_n_layers,
        hf_root,
        precision,
        device,
        prompt_template,
        column_token_map,
        instruction,
        image_field,
        label_field,
        icd_join_char="<|endofchunk|>",
        load_from_local=False,
        init_device="cpu",
        use_lora=False,
        lora_checkpoint_path=None,
    ) -> None:
        super().__init__(
            precision=precision,
            device=device,
            input_ids_field_name="lang_x",
            prompt_template=prompt_template,
            column_token_map=column_token_map,
            instruction=instruction,
            icd_join_char=icd_join_char,
            image_field=image_field,
            label_field=label_field,
        )
        hf_device_map = {"transformer": self.device}

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            use_local_files=load_from_local,
            init_device=init_device,
            model_data_type=self.data_type,
            hf_device_map=hf_device_map,
        )
        if load_from_local:
            # If loading from local, use the specified path
            flamingo_checkpoint_dir = os.path.join(
                flamingo_checkpoint_dir, "checkpoint.pt"
            )
        else:
            # Use HuggingFace default cache directory (~/.cache/huggingface/)
            # This avoids storing large files in the project directory
            hf_root = "openflamingo/" + hf_root
            flamingo_checkpoint_dir = hf_hub_download(
                hf_root, "checkpoint.pt"
                # Removed local_dir parameter to use default HF cache
            )

        self.model.load_state_dict(torch.load(flamingo_checkpoint_dir), strict=False)

        # 如果使用 LoRA，加载 LoRA adapter 到 vision encoder
        if use_lora:
            if not lora_checkpoint_path or lora_checkpoint_path == "":
                logger.warning("use_lora=true but lora_checkpoint_path is not provided. LoRA will be skipped.")
            else:
                try:
                    from peft import PeftModel
                    import os
                    
                    # 将相对路径转换为绝对路径
                    if not os.path.isabs(lora_checkpoint_path):
                        # 获取项目根目录
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        lora_checkpoint_path = os.path.join(project_root, lora_checkpoint_path)
                        lora_checkpoint_path = os.path.abspath(lora_checkpoint_path)
                    
                    # 检查路径是否存在，如果不存在，尝试查找 vision_encoder_lora 子目录
                    vision_lora_path = os.path.join(lora_checkpoint_path, "vision_encoder_lora")
                    if os.path.exists(vision_lora_path):
                        lora_checkpoint_path = vision_lora_path
                        logger.info(f"Found vision encoder LoRA at: {lora_checkpoint_path}")
                    elif not os.path.exists(lora_checkpoint_path):
                        # 提供更详细的错误信息，但不抛出异常，允许继续运行
                        logger.warning(f"LoRA checkpoint not found: {lora_checkpoint_path}")
                        logger.warning("Please ensure you have trained a model with LoRA enabled (use version v3_lora).")
                        logger.warning("LoRA checkpoint should be saved at: {}/vision_encoder_lora/".format(lora_checkpoint_path))
                        logger.warning("Continuing without LoRA...")
                        # 跳过 LoRA 加载
                        lora_checkpoint_path = None
                    
                    # 如果路径存在，尝试加载 LoRA
                    if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
                        logger.info(f"Loading LoRA adapter from {lora_checkpoint_path}")
                        try:
                            # 抑制 PEFT 关于缺失 adapter keys 的警告
                            # 这些警告通常是因为训练时没有对所有层应用 LoRA，不影响已加载的 LoRA 使用
                            import warnings
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message=".*Found missing adapter keys.*")
                                # 加载 LoRA adapter 到 vision encoder
                                peft_vision_encoder = PeftModel.from_pretrained(
                                    self.model.vision_encoder,
                                    lora_checkpoint_path
                                )
                            # 确保 LoRA 模型在正确的设备和数据类型上
                            peft_vision_encoder = peft_vision_encoder.to(device=device, dtype=self.data_type)
                            
                            # 包装 PEFT 模型的 forward 和 __call__ 方法，确保位置参数被正确传递
                            # Flamingo 调用 self.vision_encoder(vision_x) 时，vision_x 是位置参数
                            # 我们需要在 PEFT 层捕获这个位置参数，并确保它被传递到底层模型
                            base_model = peft_vision_encoder.get_base_model()
                            
                            # 使用 inspect 获取底层模型的 forward 签名
                            import inspect
                            import threading
                            base_sig = inspect.signature(base_model.forward)
                            base_params = set(base_sig.parameters.keys())
                            
                            # 使用 thread-local 存储来传递位置参数
                            thread_local = threading.local()
                            
                            # 保存原始的 PEFT 和 base_model 方法
                            original_peft_forward = peft_vision_encoder.forward
                            original_peft_call = peft_vision_encoder.__call__
                            original_base_forward = base_model.forward
                            
                            # 包装 PEFT 模型的 forward 方法
                            # 关键：保存位置参数，以便在 base_model.forward 中使用
                            def filtered_peft_forward(*args, **kwargs):
                                # 如果有位置参数，保存它
                                if len(args) > 0:
                                    thread_local.saved_args = args
                                    # 过滤掉不需要的关键字参数（如 input_ids, attention_mask 等）
                                    filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                      if k in base_params}
                                    # 调用原始的 PEFT forward，传递位置参数
                                    return original_peft_forward(*args, **filtered_kwargs)
                                else:
                                    # 如果没有位置参数，检查 kwargs 中是否有 'x'
                                    if 'x' in kwargs:
                                        x_value = kwargs.pop('x')
                                        thread_local.saved_args = (x_value,)
                                        filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                          if k in base_params}
                                        return original_peft_forward(x_value, **filtered_kwargs)
                                    else:
                                        # 尝试查找其他可能的输入键
                                        possible_input_keys = ['pixel_values', 'input', 'inputs', 'image', 'images']
                                        input_value = None
                                        for key in possible_input_keys:
                                            if key in kwargs:
                                                input_value = kwargs.pop(key)
                                                thread_local.saved_args = (input_value,)
                                                break
                                        
                                        if input_value is not None:
                                            filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                              if k in base_params}
                                            return original_peft_forward(input_value, **filtered_kwargs)
                                        else:
                                            # 如果没有找到输入，清空 saved_args 并继续
                                            thread_local.saved_args = None
                                            filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                              if k in base_params}
                                            return original_peft_forward(**filtered_kwargs)
                            
                            # 包装 PEFT 模型的 __call__ 方法
                            def filtered_peft_call(*args, **kwargs):
                                # 如果有位置参数，保存它
                                if len(args) > 0:
                                    thread_local.saved_args = args
                                    # 过滤掉不需要的关键字参数
                                    filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                      if k in base_params}
                                    # 调用原始的 PEFT __call__，传递位置参数
                                    return original_peft_call(*args, **filtered_kwargs)
                                else:
                                    # 如果没有位置参数，检查 kwargs 中是否有 'x'
                                    if 'x' in kwargs:
                                        x_value = kwargs.pop('x')
                                        thread_local.saved_args = (x_value,)
                                        filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                          if k in base_params}
                                        return original_peft_call(x_value, **filtered_kwargs)
                                    else:
                                        # 尝试查找其他可能的输入键
                                        possible_input_keys = ['pixel_values', 'input', 'inputs', 'image', 'images']
                                        input_value = None
                                        for key in possible_input_keys:
                                            if key in kwargs:
                                                input_value = kwargs.pop(key)
                                                thread_local.saved_args = (input_value,)
                                                break
                                        
                                        if input_value is not None:
                                            filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                              if k in base_params}
                                            return original_peft_call(input_value, **filtered_kwargs)
                                        else:
                                            # 如果没有找到输入，清空 saved_args 并继续
                                            thread_local.saved_args = None
                                            filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                              if k in base_params}
                                            return original_peft_call(**filtered_kwargs)
                            
                            # 包装 base_model.forward，使用保存的位置参数
                            def filtered_base_forward(*args, **kwargs):
                                # 检测是否是语言模型的参数（不应该传递给 vision encoder）
                                language_model_keys = {'input_ids', 'attention_mask', 'inputs_embeds', 
                                                       'output_attentions', 'output_hidden_states', 'return_dict'}
                                has_lm_params = bool(language_model_keys.intersection(kwargs.keys()))
                                
                                # 优先使用传入的位置参数
                                if len(args) > 0:
                                    final_args = args
                                elif hasattr(thread_local, 'saved_args') and thread_local.saved_args is not None:
                                    # 使用保存的位置参数（这是最可靠的方式）
                                    final_args = thread_local.saved_args
                                    if has_lm_params:
                                        logger.debug("Detected language model parameters in kwargs, using saved_args instead")
                                elif 'x' in kwargs:
                                    # 从 kwargs 中提取 'x'
                                    final_args = (kwargs.pop('x'),)
                                else:
                                    # 尝试查找其他可能的输入键
                                    possible_input_keys = ['pixel_values', 'input', 'inputs', 'image', 'images']
                                    final_args = None
                                    for key in possible_input_keys:
                                        if key in kwargs:
                                            final_args = (kwargs.pop(key),)
                                            break
                                    
                                    if final_args is None:
                                        # 如果仍然找不到，优先检查是否有保存的参数
                                        if hasattr(thread_local, 'saved_args') and thread_local.saved_args is not None:
                                            final_args = thread_local.saved_args
                                            logger.debug(f"Using saved_args from thread_local: {type(final_args[0]) if final_args else None}")
                                        elif has_lm_params:
                                            # 如果检测到语言模型参数，这可能是 PEFT 的错误调用
                                            # 尝试使用 saved_args
                                            if hasattr(thread_local, 'saved_args') and thread_local.saved_args is not None:
                                                final_args = thread_local.saved_args
                                                logger.warning("PEFT called base_model.forward with language model parameters, using saved_args")
                                            else:
                                                # 如果确实找不到，抛出清晰的错误
                                                error_msg = (
                                                    f"filtered_base_forward called with language model parameters but no valid input. "
                                                    f"args={len(args)}, kwargs keys={list(kwargs.keys())}. "
                                                    f"This usually means PEFT is calling base_model.forward with incorrect arguments. "
                                                    f"Please check if the LoRA adapter is compatible with the vision encoder. "
                                                    f"Thread-local saved_args: {getattr(thread_local, 'saved_args', 'not set')}"
                                                )
                                                logger.error(error_msg)
                                                raise ValueError(error_msg)
                                        else:
                                            # 如果确实找不到，尝试使用第一个非 None 的 kwargs 值作为输入
                                            # 这通常不应该发生，但作为最后的回退
                                            logger.warning(f"filtered_base_forward called with no valid input: args={len(args)}, kwargs keys={list(kwargs.keys())}")
                                            logger.warning("Attempting to use first non-None kwargs value as fallback")
                                            for key, value in kwargs.items():
                                                if value is not None and not isinstance(value, (bool, int, str, list)):
                                                    # 可能是输入张量（torch.Tensor）
                                                    if hasattr(value, 'shape'):  # 检查是否是张量
                                                        final_args = (value,)
                                                        logger.warning(f"Using '{key}' as input parameter (fallback)")
                                                        break
                                        
                                        if final_args is None:
                                            # 如果还是找不到，抛出清晰的错误
                                            error_msg = (
                                                f"filtered_base_forward called with no valid input. "
                                                f"args={len(args)}, kwargs keys={list(kwargs.keys())}. "
                                                f"This usually means PEFT is calling base_model.forward with incorrect arguments. "
                                                f"Please check if the LoRA adapter is compatible with the vision encoder. "
                                                f"Thread-local saved_args: {getattr(thread_local, 'saved_args', 'not set')}"
                                            )
                                            logger.error(error_msg)
                                            raise ValueError(error_msg)
                                
                                # 过滤掉不需要的关键字参数
                                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                  if k in base_params}
                                
                                # 调用原始的 base_model.forward
                                return original_base_forward(*final_args, **filtered_kwargs)
                            
                            # 替换方法
                            peft_vision_encoder.forward = filtered_peft_forward
                            peft_vision_encoder.__call__ = filtered_peft_call
                            base_model.forward = filtered_base_forward
                            
                            self.model.vision_encoder = peft_vision_encoder
                            logger.info("LoRA adapter loaded successfully")
                        except Exception as load_error:
                            logger.warning(f"Failed to load LoRA adapter: {load_error}")
                            logger.warning("Continuing without LoRA...")
                except ImportError:
                    raise ImportError("peft library is required for LoRA support. Install it with: pip install peft")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA adapter: {e}")
                    logger.warning("Continuing without LoRA...")

        self.model.to(device=device, dtype=self.data_type, non_blocking=True)
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.image_prompt = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_prompt)

    def add_image_token(self, text):
        return self.image_prompt + text

    def concat_prompt(
        self,
        data_sample_list: list,
        add_eos_token: bool = False,
        add_image_token: bool = True,
        is_last_for_generation: bool = True,
        query_label: Optional[int] = None,
    ):
        """Return the concatenated prompt: <Instruction>[<IMAGE_TOKEN>]text1<icd_join_char> ... textn[<icd_join_char>][</s>]

        Args:
            data_sample_list (List[DataSample]): List of data samples used to generate parts of the prompt.
            add_eos_token (bool, optional): Whether to add the EOS token at the end of the prompt. Defaults to False.
            add_image_token (bool, optional): Whether to add an image token for each sample. Defaults to True.
            is_last_for_infer (bool, optional): Whether the last data sample is used as a query for Generation inference. Defaults to True.

        Returns:
            str: Concatenated prompt string.
        """
        # 处理空列表的情况（第一次迭代时可能没有 ICD 样本）
        if len(data_sample_list) == 0:
            return self.instruction
        
        prompt = self.instruction
        ice_data_sample_list = data_sample_list[:-1]
        query_data_sample = data_sample_list[-1]

        if is_last_for_generation:
            query_prompt = self.gen_text_without_label(
                query_data_sample, add_image_token=add_image_token
            )
        else:
            query_prompt = self.gen_text_with_label(
                query_data_sample, query_label, add_image_token
            )

        ice_prompt_list = [
            self.gen_text_with_label(item, add_image_token=add_image_token)
            for item in ice_data_sample_list
        ]
        for ice_prompt in ice_prompt_list:
            prompt += ice_prompt.strip(" ") + self.icd_join_char

        prompt += query_prompt
        if is_last_for_generation:
            return prompt

        if add_eos_token:
            prompt += self.tokenizer.eos_token

        return prompt

    def prepare_input(
        self,
        batch_prompts,
        add_eos_token: bool = False,
        is_last_for_generation: bool = True,
        debug=False,
    ):
        if not any(isinstance(i, list) for i in batch_prompts):
            batch_prompts = [batch_prompts]
        image_token = "<image>"

        all_images = []
        all_raw_texts = []
        for sample in batch_prompts:
            # the model was trained on samples starting with <s>
            full_text = self.instruction

            # an image can either be an image object in the item or the url, everything else is a verbatim prompt text
            image_objects = []

            for i, item in enumerate(sample):
                item_is_img = self.is_img(item)
                if item_is_img is None:
                    item = item.strip(" ")
                    full_text += item
                    if i != len(sample) - 1 or not is_last_for_generation:
                        full_text += self.icd_join_char
                else:
                    full_text += image_token
                    image_objects.append(item_is_img)

            if add_eos_token and not is_last_for_generation:
                full_text += self.tokenizer.eos_token

            if debug is True:
                print(f"{full_text=}")

            image_objects = torch.stack(
                [self.image_processor(image) for image in image_objects], dim=0
            )
            # 确保图像数据在正确的数据类型上（与模型的数据类型匹配）
            if hasattr(self, 'data_type'):
                image_objects = image_objects.to(dtype=self.data_type)
            all_raw_texts.append(full_text)
            all_images.append(image_objects)

        # max_num_images has to be at least 1 even when there are no images
        max_num_images = max(len(x) for x in all_images)
        max_num_images = max(1, max_num_images)

        output_input_ids = []
        output_images = []
        output_attention_masks = []

        text_tensor_input = self.tokenizer(
            all_raw_texts, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        for text_tensor, images in zip(text_tensor_input["input_ids"], all_images):
            image_count = (text_tensor == self.image_token_id).sum()

            local_max_num_images = min(image_count, max_num_images)
            current_images = images[:local_max_num_images]

            if len(current_images) > 0:
                padded_image_tensor = torch.zeros(
                    max_num_images, *current_images.size()[1:]
                )
                padded_image_tensor[: current_images.size(0)] = current_images
            else:
                padded_image_tensor = torch.zeros(
                    max_num_images, *self.default_image_dims
                )

            output_images.append(padded_image_tensor)

        output_input_ids = text_tensor_input["input_ids"]
        output_images = torch.stack(output_images)
        output_attention_masks = text_tensor_input["attention_mask"]

        # 确保所有输入数据在正确的设备和数据类型上（与模型匹配）
        vision_x = output_images.unsqueeze(dim=2)
        if hasattr(self, 'data_type'):
            vision_x = vision_x.to(dtype=self.data_type)
        if hasattr(self, 'device'):
            vision_x = vision_x.to(device=self.device)
            # 同时确保 lang_x 和 attention_mask 也在正确的设备上
            output_input_ids = output_input_ids.to(device=self.device)
            output_attention_masks = output_attention_masks.to(device=self.device)

        return BatchFeature(
            data={
                "lang_x": output_input_ids,
                "attention_mask": output_attention_masks,
                "vision_x": vision_x,
            }
        )


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    init_device="cpu",
    model_data_type=torch.bfloat16,
    hf_device_map="auto",
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if "llama-7b" in lang_encoder_path:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            local_files_only=use_local_files,
            trust_remote_code=True,
        )
    else:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            local_files_only=use_local_files,
            trust_remote_code=True,
            init_device=init_device,
            torch_dtype=model_data_type,
            device_map=hf_device_map,
        )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied

    logger.info(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
