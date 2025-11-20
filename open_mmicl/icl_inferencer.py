from typing import Optional

import torch
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from open_mmicl.interface.base_interface import BaseInterface
from open_mmicl.utils import VLGenInferencerOutputHandler, PPLInferencerOutputHandler


class ICLInferecer:
    def __init__(
        self,
        interface: BaseInterface,
        train_ds,
        test_ds,
        generation_kwargs,
        other_save_field=None,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 8,
        num_proc: Optional[int] = 12,
        preprocessor_bs: Optional[int] = 100,
        output_json_filepath: Optional[str] = "./vl_icl_inference_output",
        output_json_filename: Optional[str] = "predictions",
    ) -> None:
        self.interface = interface
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.generation_kwargs = generation_kwargs
        self.other_save_field = other_save_field
        # 如果使用 CUDA，将 num_workers 设置为 0，避免 CUDA 在多进程中的初始化问题
        # 检查 device 是否为 CUDA（支持字符串和 torch.device 对象）
        if hasattr(interface, 'device'):
            device_str = str(interface.device)
            if 'cuda' in device_str.lower():
                if num_workers > 0:
                    logger.warning(f"CUDA device detected ({device_str}), setting num_workers=0 to avoid CUDA multiprocessing issues")
                self.num_workers = 0
            else:
                self.num_workers = num_workers
                logger.debug(f"Non-CUDA device detected ({device_str}), using num_workers={num_workers}")
        else:
            # 如果没有 device 属性，使用传入的 num_workers
            logger.warning(f"Interface has no 'device' attribute, using num_workers={num_workers}")
            self.num_workers = num_workers
        self.num_proc = num_proc
        self.preprocessor_bs = preprocessor_bs
        self.batch_size = batch_size
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename
    
    def _get_safe_num_workers(self):
        """获取安全的 num_workers 值，确保在使用 CUDA 时返回 0"""
        if hasattr(self.interface, 'device'):
            device_str = str(self.interface.device)
            if 'cuda' in device_str.lower() and self.num_workers > 0:
                logger.warning(f"CUDA device detected ({device_str}), forcing num_workers=0 for DataLoader")
                return 0
        return self.num_workers

    @torch.inference_mode()
    def inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list,
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ):
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename
        num = len(test_ds)
        output_handler = VLGenInferencerOutputHandler(num)
        index = 0

        test_ds = test_ds.add_column("ice_idx", ice_idx_list)

        output_handler.creat_index(test_ds)
        output_handler.save_origin_info("ice_idx", test_ds)
        for fields in self.other_save_field:
            output_handler.save_origin_info(fields, test_ds)

        def prepare_input_map(
            examples,
        ):
            ice_idx = [i for i in examples["ice_idx"]]
            prompts = []
            num_example = len(ice_idx)
            sub_data_sample = [
                {k: v[i] for k, v in examples.items()} for i in range(num_example)
            ]
            batch_data_smaple_list = []
            for i, e in enumerate(sub_data_sample):
                ice_sample_list = [train_ds[idx] for idx in ice_idx[i]]
                data_sample_list = ice_sample_list + [e]
                batch_data_smaple_list.append(data_sample_list)
            prompts = self.interface.transfer_prompts(
                batch_data_smaple_list, is_last_for_generation=True
            )

            input_tensor_dict = self.interface.prepare_input(
                prompts, is_last_for_generation=True
            )
            # Convert BatchFeature to dict to avoid DataLoader issues with image_grid_thw
            # DataLoader may incorrectly process BatchFeature, causing image_grid_thw shape issues
            if hasattr(input_tensor_dict, 'data'):
                result = dict(input_tensor_dict.data)
            elif isinstance(input_tensor_dict, dict):
                result = input_tensor_dict
            else:
                result = dict(input_tensor_dict)
            
            # Debug: Check image_grid_thw shape before returning
            if 'image_grid_thw' in result:
                logger.debug(f"image_grid_thw shape in prepare_input_map (before DataLoader): {result['image_grid_thw'].shape}")
            
            return result

        test_ds.set_transform(prepare_input_map)
        safe_num_workers = self._get_safe_num_workers()
        dataloader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=safe_num_workers,
            shuffle=False,
        )

        # 4. Inference for prompts in each batch
        logger.info("Starting inference process...")
        
        # Show beam search configuration if enabled
        num_beams = self.generation_kwargs.get('num_beams', 1)
        max_new_tokens = self.generation_kwargs.get('max_new_tokens', 'N/A')
        if num_beams > 1:
            logger.info(f"束搜索配置: num_beams={num_beams}, max_new_tokens={max_new_tokens}, length_penalty={self.generation_kwargs.get('length_penalty', 0.0)}")

        for data in tqdm(dataloader, ncols=100, desc="推理中" if num_beams == 1 else f"束搜索推理 (beams={num_beams})"):
            # 5-1. Inference with local model
            # Move to device
            data = {k: v.to(self.interface.device) for k, v in data.items()}
            
            # Fix: DataLoader adds batch dimension to tensors
            # For Qwen2.5-VL, image_grid_thw should be [num_images, 3], not [batch_size, num_images, 3]
            # Remove batch dimension if present and update image_nums accordingly
            if 'image_grid_thw' in data:
                original_shape = data['image_grid_thw'].shape
                logger.debug(f"image_grid_thw shape before fix (in inferencer): {original_shape}")
                
                # Handle different DataLoader collation scenarios
                if data['image_grid_thw'].dim() == 3 and data['image_grid_thw'].shape[0] == 1:
                    # Remove batch dimension: [1, num_images, 3] -> [num_images, 3]
                    data['image_grid_thw'] = data['image_grid_thw'].squeeze(0)
                    logger.debug(f"Fixed image_grid_thw: {original_shape} -> {data['image_grid_thw'].shape}")
                    # Update image_nums if present
                    if 'image_nums' in data:
                        if isinstance(data['image_nums'], torch.Tensor) and data['image_nums'].numel() == 1:
                            # Keep as is, it's already correct for single batch
                            pass
                        elif isinstance(data['image_nums'], torch.Tensor) and data['image_nums'].numel() > 1:
                            # Take first element for single batch
                            data['image_nums'] = data['image_nums'][0:1]
                        elif isinstance(data['image_nums'], list) and len(data['image_nums']) > 0:
                            data['image_nums'] = torch.tensor([data['image_nums'][0]], dtype=torch.long)
                    else:
                        # Calculate image_nums from image_grid_thw
                        num_images = data['image_grid_thw'].shape[0]
                        data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
                elif data['image_grid_thw'].dim() == 2:
                    # Ensure image_nums is set correctly
                    if 'image_nums' not in data:
                        num_images = data['image_grid_thw'].shape[0]
                        data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
                    elif isinstance(data['image_nums'], torch.Tensor):
                        # Ensure image_nums is 1D tensor with correct dtype
                        if data['image_nums'].dim() > 1:
                            data['image_nums'] = data['image_nums'].flatten()
                        if data['image_nums'].dtype != torch.long:
                            data['image_nums'] = data['image_nums'].long()
                    logger.debug(f"Final image_grid_thw shape: {data['image_grid_thw'].shape}")
                    logger.debug(f"Final image_nums: {data['image_nums']}")
            
            prompt_len = int(data["attention_mask"].shape[1])
            
            # Standard Qwen2.5-VL inference format: model.generate(**inputs, ...)
            outputs = self.interface.generate(
                **data,
                eos_token_id=self.interface.tokenizer.eos_token_id,
                pad_token_id=self.interface.tokenizer.pad_token_id,
                **self.generation_kwargs,
            )
            outputs = outputs.tolist()
            complete_output = self.interface.tokenizer.batch_decode(
                outputs[:], skip_special_tokens=False
            )
            output_without_sp_token = self.interface.tokenizer.batch_decode(
                outputs[:], skip_special_tokens=True
            )
            generated = self.interface.tokenizer.batch_decode(
                [output[prompt_len:] for output in outputs],
                skip_special_tokens=True,
            )
            origin_prompt = self.interface.tokenizer.batch_decode(
                [output[:prompt_len] for output in outputs],
                skip_special_tokens=True,
            )

            # 5-3. Save current output
            for prediction, output, pure_output in zip(
                generated, complete_output, output_without_sp_token
            ):
                output_handler.save_prediction_and_output(
                    prediction, [output, pure_output], origin_prompt, index
                )
                index = index + 1

        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return output_handler.results_dict

    @torch.inference_mode()
    def gen_inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list,
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ):
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename
        num = len(test_ds)
        output_handler = VLGenInferencerOutputHandler(num)
        index = 0

        test_ds = test_ds.add_column("ice_idx", ice_idx_list)

        output_handler.creat_index(test_ds)
        output_handler.save_origin_info("ice_idx", test_ds)
        for fields in self.other_save_field:
            output_handler.save_origin_info(fields, test_ds)

        def prepare_input_map(
            examples,
        ):
            ice_idx = [i for i in examples["ice_idx"]]
            prompts = []
            num_example = len(ice_idx)
            sub_data_sample = [
                {k: v[i] for k, v in examples.items()} for i in range(num_example)
            ]
            batch_data_smaple_list = []
            for i, e in enumerate(sub_data_sample):
                ice_sample_list = [train_ds[idx] for idx in ice_idx[i]]
                data_sample_list = ice_sample_list + [e]
                batch_data_smaple_list.append(data_sample_list)
            prompts = self.interface.transfer_prompts(
                batch_data_smaple_list, is_last_for_generation=True
            )

            input_tensor_dict = self.interface.prepare_input(
                prompts, is_last_for_generation=True
            )
            # Convert BatchFeature to dict to avoid DataLoader issues with image_grid_thw
            # DataLoader may incorrectly process BatchFeature, causing image_grid_thw shape issues
            if hasattr(input_tensor_dict, 'data'):
                result = dict(input_tensor_dict.data)
            elif isinstance(input_tensor_dict, dict):
                result = input_tensor_dict
            else:
                result = dict(input_tensor_dict)
            
            # Debug: Check image_grid_thw shape before returning
            if 'image_grid_thw' in result:
                logger.debug(f"image_grid_thw shape in prepare_input_map (before DataLoader): {result['image_grid_thw'].shape}")
            
            return result

        test_ds.set_transform(prepare_input_map)
        safe_num_workers = self._get_safe_num_workers()
        dataloader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=safe_num_workers,
            shuffle=False,
        )

        # 4. Inference for prompts in each batch
        logger.info("Starting inference process...")
        
        # Show beam search configuration if enabled
        num_beams = self.generation_kwargs.get('num_beams', 1)
        max_new_tokens = self.generation_kwargs.get('max_new_tokens', 'N/A')
        if num_beams > 1:
            logger.info(f"束搜索配置: num_beams={num_beams}, max_new_tokens={max_new_tokens}, length_penalty={self.generation_kwargs.get('length_penalty', 0.0)}")

        for data in tqdm(dataloader, ncols=100, desc="推理中" if num_beams == 1 else f"束搜索推理 (beams={num_beams})"):
            # 5-1. Inference with local model
            # Move to device
            data = {k: v.to(self.interface.device) for k, v in data.items()}
            
            # Fix: DataLoader adds batch dimension to tensors
            # For Qwen2.5-VL, image_grid_thw should be [num_images, 3], not [batch_size, num_images, 3]
            # Remove batch dimension if present and update image_nums accordingly
            if 'image_grid_thw' in data:
                original_shape = data['image_grid_thw'].shape
                logger.debug(f"image_grid_thw shape before fix (in inferencer): {original_shape}")
                
                # Handle different DataLoader collation scenarios
                if data['image_grid_thw'].dim() == 3 and data['image_grid_thw'].shape[0] == 1:
                    # Remove batch dimension: [1, num_images, 3] -> [num_images, 3]
                    data['image_grid_thw'] = data['image_grid_thw'].squeeze(0)
                    logger.debug(f"Fixed image_grid_thw: {original_shape} -> {data['image_grid_thw'].shape}")
                    # Update image_nums if present
                    if 'image_nums' in data:
                        if isinstance(data['image_nums'], torch.Tensor) and data['image_nums'].numel() == 1:
                            # Keep as is, it's already correct for single batch
                            pass
                        elif isinstance(data['image_nums'], torch.Tensor) and data['image_nums'].numel() > 1:
                            # Take first element for single batch
                            data['image_nums'] = data['image_nums'][0:1]
                        elif isinstance(data['image_nums'], list) and len(data['image_nums']) > 0:
                            data['image_nums'] = torch.tensor([data['image_nums'][0]], dtype=torch.long)
                    else:
                        # Calculate image_nums from image_grid_thw
                        num_images = data['image_grid_thw'].shape[0]
                        data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
                elif data['image_grid_thw'].dim() == 2:
                    # Ensure image_nums is set correctly
                    if 'image_nums' not in data:
                        num_images = data['image_grid_thw'].shape[0]
                        data['image_nums'] = torch.tensor([num_images], dtype=torch.long)
                    elif isinstance(data['image_nums'], torch.Tensor):
                        # Ensure image_nums is 1D tensor with correct dtype
                        if data['image_nums'].dim() > 1:
                            data['image_nums'] = data['image_nums'].flatten()
                        if data['image_nums'].dtype != torch.long:
                            data['image_nums'] = data['image_nums'].long()
                    logger.debug(f"Final image_grid_thw shape: {data['image_grid_thw'].shape}")
                    logger.debug(f"Final image_nums: {data['image_nums']}")
            
            prompt_len = int(data["attention_mask"].shape[1])
            
            # Standard Qwen2.5-VL inference format: model.generate(**inputs, ...)
            outputs = self.interface.generate(
                **data,
                eos_token_id=self.interface.tokenizer.eos_token_id,
                pad_token_id=self.interface.tokenizer.pad_token_id,
                **self.generation_kwargs,
            )
            outputs = outputs.tolist()
            complete_output = self.interface.tokenizer.batch_decode(
                outputs[:], skip_special_tokens=False
            )
            output_without_sp_token = self.interface.tokenizer.batch_decode(
                outputs[:], skip_special_tokens=True
            )
            generated = self.interface.tokenizer.batch_decode(
                [output[prompt_len:] for output in outputs],
                skip_special_tokens=True,
            )
            origin_prompt = self.interface.tokenizer.batch_decode(
                [output[:prompt_len] for output in outputs],
                skip_special_tokens=True,
            )

            # 5-3. Save current output
            for prediction, output, pure_output in zip(
                generated, complete_output, output_without_sp_token
            ):
                output_handler.save_prediction_and_output(
                    prediction, [output, pure_output], origin_prompt, index
                )
                index = index + 1

        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return output_handler.results_dict

    @torch.inference_mode()
    def ppl_inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list,
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ):
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename
        output_handler = PPLInferencerOutputHandler()

        output_handler.save_ice(ice_idx_list)
        test_ds = test_ds.add_column("ice_idx", ice_idx_list)

        output_handler.creat_index(test_ds)
        output_handler.save_origin_info("ice_idx", test_ds)
        for fields in self.other_save_field:
            output_handler.save_origin_info(fields, test_ds)

        ppl = []
        prediction_list = []
        label_encode_list = np.unique(np.array(test_ds[self.interface.label_field]))
        label_encode_list = label_encode_list.tolist()
        for label in label_encode_list:
            label_ppl = []

            def prepare_input_map(
                examples,
            ):
                ice_idx = [i for i in examples["ice_idx"]]
                prompts = []
                num_example = len(ice_idx)
                sub_data_sample = [
                    {k: v[i] for k, v in examples.items()} for i in range(num_example)
                ]
                batch_data_smaple_list = []
                for i, e in enumerate(sub_data_sample):
                    ice_sample_list = [train_ds[idx] for idx in ice_idx[i]]
                    data_sample_list = ice_sample_list + [e]
                    batch_data_smaple_list.append(data_sample_list)
                prompts = self.interface.transfer_prompts(
                    batch_data_smaple_list,
                    is_last_for_generation=False,
                    query_label=label,
                )

                input_tensor_dict = self.interface.prepare_input(
                    prompts, is_last_for_generation=False
                )
                return input_tensor_dict

            test_ds.set_transform(prepare_input_map)
            safe_num_workers = self._get_safe_num_workers()
            dataloader = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=safe_num_workers,
                shuffle=False,
            )
            # 4. Inference for prompts in each batch
            logger.info(f"Starting inference {label=} process...")
            index = 0
            for data in tqdm(dataloader, ncols=100):
                data = {k: v.to(self.interface.device) for k, v in data.items()}
                data_ppl = self.interface.get_ppl(data).cpu().tolist()
                sub_prompt_list = self.interface.tokenizer.batch_decode(
                    data["input_ids"],
                    skip_special_tokens=True,
                )
                for res, prompt in zip(data_ppl, sub_prompt_list):
                    label_ppl.append(res)
                    ice = self.interface.icd_join_char.join(
                        prompt.split(self.interface.icd_join_char)[:-1]
                    )
                    output_handler.save_prompt_and_ppl(label, ice, prompt, res, index)
                    index = index + 1
            ppl.append(label_ppl)

        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            prediction_list.append(label_encode_list[single_ppl.index(min(single_ppl))])
        output_handler.save_predictions(prediction_list)
        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return output_handler.results_dict
