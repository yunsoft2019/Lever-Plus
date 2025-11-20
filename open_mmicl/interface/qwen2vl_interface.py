from typing import Optional

import torch
from loguru import logger
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BatchFeature
from qwen_vl_utils import process_vision_info

from .base_interface import LVLMInterface


class Qwen2VLInterface(LVLMInterface):
    def __init__(
        self,
        model_name,
        load_from_local,
        precision,
        device,
        prompt_template,
        column_token_map,
        instruction,
        image_field,
        label_field,
        icd_join_char="<|endofchunk|>",
        system_prompt=None,
    ):
        super().__init__(
            precision=precision,
            device=device,
            input_ids_field_name="input_ids",
            prompt_template=prompt_template,
            column_token_map=column_token_map,
            instruction=instruction,
            icd_join_char=icd_join_char,
            image_field=image_field,
            label_field=label_field,
        )
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=load_from_local
        )
        
        # Check if flash_attn is available
        try:
            import flash_attn
            flash_attn_available = True
        except ImportError:
            flash_attn_available = False
            logger.info("flash_attn not available, using sdpa attention instead")
        
        # Determine attention implementation
        if flash_attn_available and torch.cuda.is_available():
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
        
        # Load model according to official documentation
        model_kwargs = {
            "dtype": self.data_type,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
            "local_files_only": load_from_local,
        }
        
        # Use the specified device instead of device_map="auto" to avoid auto-distribution across all GPUs
        # This ensures the model only uses the GPU specified in gpu_ids parameter
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move model to the specified device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.image_processor = self.processor.image_processor
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Qwen2.5-VL uses <image> token
        self.image_token = "<image>"
        try:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        except:
            # If image token doesn't exist, try to add it
            self.image_token_id = None
            logger.warning(f"Image token {self.image_token} not found in tokenizer")
        
        # Store system prompt
        self.system_prompt = system_prompt

    def add_image_token(self, text):
        return self.image_token + text

    def transfer_prompts(
        self, batch_data_sample_list, is_last_for_generation=True, query_label=None
    ):
        """
        transfer data sample list to prompt format for Qwen2.5-VL.
        Note: Only support one image and one text pair.
        """
        if not any(isinstance(i, list) for i in batch_data_sample_list):
            batch_data_sample_list = [batch_data_sample_list]

        prompts = []
        for data_sample_list in batch_data_sample_list:
            prompt = []
            for data_sample in data_sample_list[:-1]:
                prompt.extend(
                    [
                        data_sample[self.image_field],
                        self.gen_text_with_label(data_sample),
                    ]
                )
            prompt.append(data_sample_list[-1][self.image_field])
            if is_last_for_generation:
                prompt.append(self.gen_text_without_label(data_sample_list[-1]))
            else:
                prompt.append(
                    self.gen_text_with_label(data_sample_list[-1], label=query_label)
                )

            prompts.append(prompt)
        return prompts

    def concat_prompt(
        self,
        data_sample_list: list,
        add_eos_token: bool = False,
        add_image_token: bool = True,
        is_last_for_generation: bool = True,
        query_label: Optional[int] = None,
    ):
        """Return the concatenated prompt for Qwen2.5-VL"""
        # Handle empty list case (when there are no ICD samples yet)
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
        add_eos_token=False,
        is_last_for_generation=False,
        debug=False,
        transform=None,
    ):
        """Prepare input for Qwen2.5-VL model using official method"""
        if not any(isinstance(i, list) for i in batch_prompts):
            batch_prompts = [batch_prompts]

        all_messages = []
        
        for sample in batch_prompts:
            # Build messages in the format expected by Qwen2.5-VL
            # Format: alternating image and text items
            content = []
            
            for item in sample:
                item_is_img = self.is_img(item)
                if item_is_img is not None:
                    content.append({"type": "image", "image": item_is_img})
                else:
                    # Extract text from the item
                    text = item.strip(" ")
                    if text:
                        content.append({"type": "text", "text": text})
            
            # Create message format
            if content:
                message = []
                # Add system prompt if configured
                if self.system_prompt:
                    message.append({"role": "system", "content": self.system_prompt})
                # Add user content
                message.append({"role": "user", "content": content})
                all_messages.append(message)
                # Log the first message for debugging (only once per session)
                if len(all_messages) == 1 and not hasattr(self, '_logged_message_format'):
                    logger.debug("=" * 80)
                    logger.debug("Qwen2.5-VL Message Format (first sample):")
                    logger.debug(f"Message structure: {message}")
                    # Log system prompt if present
                    if self.system_prompt:
                        logger.debug(f"System prompt: {self.system_prompt[:200]}...")  # Show first 200 chars
                    # Log content details
                    logger.debug("User content items:")
                    for i, item in enumerate(content):
                        if item["type"] == "image":
                            logger.debug(f"  Content[{i}]: type=image, image=<PIL.Image object>")
                        else:
                            logger.debug(f"  Content[{i}]: type=text, text={item['text'][:200]}...")  # Show first 200 chars
                    logger.debug("=" * 80)
                    self._logged_message_format = True
            else:
                # Fallback: pure text
                text_content = " ".join([item.strip() for item in sample if not self.is_img(item)])
                message = []
                # Add system prompt if configured
                if self.system_prompt:
                    message.append({"role": "system", "content": self.system_prompt})
                # Add user content
                message.append({"role": "user", "content": text_content})
                all_messages.append(message)
                if len(all_messages) == 1 and not hasattr(self, '_logged_message_format'):
                    logger.debug("=" * 80)
                    logger.debug("Qwen2.5-VL Message Format (pure text, first sample):")
                    logger.debug(f"Message: {message}")
                    if self.system_prompt:
                        logger.debug(f"System prompt: {self.system_prompt[:200]}...")
                    logger.debug("=" * 80)
                    self._logged_message_format = True

        # Process with Qwen2.5-VL processor using official method
        # Step 1: Apply chat template
        texts = []
        image_inputs_list = []
        video_inputs_list = []
        
        for idx, messages in enumerate(all_messages):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=is_last_for_generation
            )
            texts.append(text)
            
            # Log the processed text for the first sample (only once per session)
            if idx == 0 and not hasattr(self, '_logged_processed_text'):
                logger.debug("=" * 80)
                logger.debug("Qwen2.5-VL Processed Text (after apply_chat_template, first sample):")
                logger.debug(f"Text length: {len(text)} characters")
                logger.debug(f"Text preview (first 500 chars):\n{text[:500]}...")
                logger.debug("=" * 80)
                self._logged_processed_text = True
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)
            
            # Log vision info for the first sample (only once per session)
            if idx == 0 and not hasattr(self, '_logged_vision_info'):
                logger.debug("=" * 80)
                logger.debug("Qwen2.5-VL Vision Info (first sample):")
                logger.debug(f"Number of images: {len(image_inputs) if image_inputs else 0}")
                logger.debug(f"Number of videos: {len(video_inputs) if video_inputs else 0}")
                if image_inputs:
                    logger.debug(f"Image shapes: {[img.size if hasattr(img, 'size') else type(img).__name__ for img in image_inputs]}")
                logger.debug("=" * 80)
                self._logged_vision_info = True
        
        # Step 2: Process with processor
        # Qwen2.5-VL: Process items one by one (sequential processing)
        # This is more reliable than batch processing because:
        # 1. Each sample may have different number of images (different ICD counts)
        # 2. Images may have different sizes
        # 3. Reduces memory usage and avoids OOM errors
        batch_inputs = []
        for text, img_inputs, vid_inputs in zip(texts, image_inputs_list, video_inputs_list):
            item_inputs = self.processor(
                text=[text],
                images=img_inputs if img_inputs else None,
                videos=vid_inputs if vid_inputs else None,
                padding=True,
                return_tensors="pt",
            )
            batch_inputs.append(item_inputs)
        
        # Pad batch inputs to same length (if multiple items)
        if len(batch_inputs) == 1:
            inputs = batch_inputs[0]
            # For Qwen2.5-VL, image_nums is needed for beam search expansion
            # Calculate image_nums from image_grid_thw shape if not present
            if 'image_grid_thw' in inputs:
                if 'image_nums' not in inputs:
                    # image_grid_thw shape is [num_images, 3]
                    num_images = inputs['image_grid_thw'].shape[0]
                    inputs['image_nums'] = torch.tensor([num_images], dtype=torch.long)
                logger.debug(f"image_grid_thw shape after single batch: {inputs['image_grid_thw'].shape}")
                logger.debug(f"image_nums: {inputs['image_nums']}")
        else:
            # Get max sequence length
            max_seq_len = max(inp["input_ids"].shape[1] for inp in batch_inputs)
            
            # Pad all inputs to same length
            padded_inputs = {}
            for key in batch_inputs[0].keys():
                if isinstance(batch_inputs[0][key], torch.Tensor):
                    if key == "input_ids" or key == "attention_mask":
                        # Pad sequences
                        padded_tensors = []
                        for inp in batch_inputs:
                            tensor = inp[key]
                            pad_length = max_seq_len - tensor.shape[1]
                            if pad_length > 0:
                                if key == "input_ids":
                                    pad_value = self.pad_token_id
                                else:
                                    pad_value = 0
                                padded = torch.nn.functional.pad(
                                    tensor, (0, pad_length), value=pad_value
                                )
                            else:
                                padded = tensor
                            padded_tensors.append(padded)
                        padded_inputs[key] = torch.cat(padded_tensors, dim=0)
                    else:
                        # Special handling for Qwen2.5-VL image_grid_thw
                        # image_grid_thw should be concatenated along dim=0 (flatten all images)
                        if key == 'image_grid_thw':
                            # Concatenate all image_grid_thw tensors along dim=0
                            padded_inputs[key] = torch.cat([inp[key] for inp in batch_inputs], dim=0)
                        else:
                            # For other tensors (like pixel_values), stack them
                            # But handle variable sizes carefully
                            try:
                                padded_inputs[key] = torch.cat([inp[key] for inp in batch_inputs], dim=0)
                            except RuntimeError:
                                # If shapes don't match, pad to max size
                                max_shape = max(inp[key].shape for inp in batch_inputs)
                                padded_tensors = []
                                for inp in batch_inputs:
                                    tensor = inp[key]
                                    # Pad to max_shape if needed
                                    if tensor.shape != max_shape:
                                        # Create padding
                                        pad_dims = []
                                        for i, (s, m) in enumerate(zip(tensor.shape, max_shape)):
                                            pad_dims.extend([0, m - s])
                                        pad_dims = pad_dims[::-1]  # Reverse for F.pad format
                                        tensor = torch.nn.functional.pad(tensor, pad_dims)
                                    padded_tensors.append(tensor)
                                padded_inputs[key] = torch.stack(padded_tensors, dim=0)
                else:
                    # For non-tensor values, keep as list
                    padded_inputs[key] = [inp[key] for inp in batch_inputs]
            
            inputs = padded_inputs
            # For Qwen2.5-VL, image_nums is needed for beam search expansion
            # Calculate image_nums from each batch item's image_grid_thw shape if not present
            if 'image_grid_thw' in inputs:
                if 'image_nums' not in inputs:
                    # Calculate image_nums for each item in the batch
                    # Each item in batch_inputs has its own image_grid_thw
                    image_nums_list = [inp['image_grid_thw'].shape[0] for inp in batch_inputs]
                    inputs['image_nums'] = torch.tensor(image_nums_list, dtype=torch.long)
                logger.debug(f"image_grid_thw shape after batch padding: {inputs['image_grid_thw'].shape}")
                logger.debug(f"image_nums: {inputs['image_nums']}")
        
        # Move to device and return as BatchFeature (consistent with Flamingo interface)
        for key in inputs.keys():
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Return as dict for Qwen2.5-VL to avoid DataLoader batch processing issues
        # DataLoader may incorrectly process BatchFeature, causing image_grid_thw shape issues
        # Returning dict ensures proper handling of image_grid_thw shape [num_images, 3]
        return inputs

