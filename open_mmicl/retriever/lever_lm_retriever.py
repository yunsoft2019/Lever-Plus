from typing import List, Optional

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ProcessorMixin

from open_mmicl.retriever.base_retriever import BaseRetriever


class LeverLMRetriever(BaseRetriever):
    def __init__(
        self,
        index_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
        lever_lm: torch.nn.Module,
        processor: ProcessorMixin,
        query_image_field: Optional[str] = None,
        query_text_field: Optional[str] = None,
        icd_image_field: Optional[str] = None,
        icd_text_field: Optional[str] = None,
        device: str = "cpu",
        infer_batch_size: int = 1,
        infer_num_workers: int = 0,
        reverse_seq: bool = False,
        disable_stop: bool = False,  # 新增：禁用 STOP 机制
    ):
        """Initialize the LeverLMRetriever.

        Args:
            index_ds (datasets.Dataset): The dataset used for creating the index.
            test_ds (datasets.Dataset): The dataset used for testing.
            lever_lm (torch.nn.Module): ICD Language Model used for retrieval.
            processor (ProcessorMixin): The processor for preparing input data.
            query_image_field (Optional[str]): The field name for query images in the dataset.
            query_text_field (Optional[str]): The field name for query text in the dataset.
            icd_image_field (Optional[str]): The field name for images in the ICD dataset.
            icd_text_field (Optional[str]): The field name for text in the ICD dataset.
            device (str): The computing device ('cpu' or 'cuda').
            infer_batch_size (int): The batch size for inference.
            infer_num_workers (int): The number of workers for data loading during inference.
        """
        super().__init__(index_ds, test_ds)
        self.lever_lm = lever_lm
        self.processor = processor
        self.device = device
        self.query_image_field = query_image_field
        self.query_text_field = query_text_field
        self.infer_batch_size = infer_batch_size
        # 如果使用 CUDA，将 infer_num_workers 设置为 0，避免 CUDA 在多进程中的初始化问题
        if isinstance(device, str) and 'cuda' in device.lower():
            if infer_num_workers > 0:
                import warnings
                warnings.warn(f"CUDA device detected ({device}), setting infer_num_workers=0 to avoid CUDA multiprocessing issues")
            self.infer_num_workers = 0
        else:
            self.infer_num_workers = infer_num_workers
        self.icd_text_field = icd_text_field
        self.icd_image_field = icd_image_field
        self.reverse_seq = reverse_seq
        self.disable_stop = disable_stop  # 新增：禁用 STOP 机制
        self._cached_query_inputs = None  # 缓存预处理后的查询数据
        self._cached_index_emb = None  # 缓存训练集的 embedding

    def _prepare_index_embeddings(self):
        """预计算训练集的 embedding，用于 Top-K 预筛选"""
        if self._cached_index_emb is not None:
            return self._cached_index_emb
        
        # 检查模型是否支持预计算 embedding
        if not hasattr(self.lever_lm, '_extract_query_emb'):
            print("[INFO] 模型不支持预计算 embedding，跳过 Top-K 预筛选")
            return None
        
        print(f"[INFO] 预计算训练集 embedding ({len(self.index_ds)} 样本)...")
        
        # 准备训练集数据
        index_ds_ = self.index_ds.map()
        
        def prepare(examples):
            images = texts = None
            if self.icd_image_field:
                images = [i for i in examples[self.icd_image_field]]
            if self.icd_text_field:
                texts = [i for i in examples[self.icd_text_field]]
            
            data_dict = self.processor(
                images=images,
                text=texts,
                padding=True,
                return_tensors="pt",
            )
            return data_dict
        
        index_ds_.set_transform(prepare)
        dataloader = DataLoader(
            index_ds_,
            batch_size=32,  # 使用较大的 batch size 加速
            shuffle=False,
            num_workers=0,
        )
        
        all_embeddings = []
        self.lever_lm = self.lever_lm.to(self.device)
        self.lever_lm.eval()
        
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Computing index embeddings", ncols=100):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # 使用模型的 _extract_query_emb 方法提取 embedding
                emb = self.lever_lm._extract_query_emb(batch)  # [B, d_model]
                all_embeddings.append(emb.cpu())
        
        self._cached_index_emb = torch.cat(all_embeddings, dim=0)  # [N, d_model]
        print(f"[INFO] 训练集 embedding 计算完成: {self._cached_index_emb.shape}")
        
        return self._cached_index_emb

    def _prepare_query_inputs(self):
        """一次性准备并缓存所有查询输入数据"""
        if self._cached_query_inputs is not None:
            return self._cached_query_inputs
        
        test_ds_ = self.test_ds.map()

        def prepare(examples):
            images = texts = None
            if self.query_image_field:
                images = [i for i in examples[self.query_image_field]]
            if self.query_text_field:
                texts = [i for i in examples[self.query_text_field]]

            data_dict = self.processor(
                images=images,
                text=texts,
                padding=True,
                return_tensors="pt",
            )
            return data_dict

        test_ds_.set_transform(prepare)
        dataloader = DataLoader(
            test_ds_,
            batch_size=self.infer_batch_size,
            shuffle=False,
            num_workers=self.infer_num_workers,
        )
        
        # 缓存所有批次的数据
        cached_inputs = []
        for query_input in tqdm(dataloader, desc="Loading query inputs", ncols=100):
            query_input = {k: v.to(self.device) for k, v in query_input.items()}
            cached_inputs.append(query_input)
        
        self._cached_query_inputs = cached_inputs
        return self._cached_query_inputs

    def retrieve(self, ice_num) -> List[List[int]]:
        """Retrieve indices from the index dataset using the LeverLM model.

        Args:
            ice_num (int): The number of indices to retrieve for each test case.

        Returns:
            List[List[int]]: A list of lists containing the retrieved indices. Each sublist corresponds to a test case.
        """
        return self.lever_lm_generation(ice_num)

    @torch.inference_mode()
    def lever_lm_generation(self, ice_num: int) -> List[List[int]]:
        """Generate indices using the LeverLM model.

        Args:
            ice_num (int): The number of indices to generate for each test case.

        Returns:
            List[List[int]]: A list of lists containing the generated indices. Each sublist corresponds to a test case.
        """
        self.lever_lm = self.lever_lm.to(self.device)
        self.lever_lm.eval()
        
        # 重置调试标志，确保每个 shot_num 都打印一次
        if hasattr(self.lever_lm, '_debug_printed'):
            delattr(self.lever_lm, '_debug_printed')
        
        # 预计算训练集 embedding（用于 Top-K 预筛选）
        precomputed_index_emb = self._prepare_index_embeddings()
        
        icd_idx_list = []
        bos_token_id = len(self.index_ds) + 1
        query_token_id = len(self.index_ds) + 2

        # 使用缓存的查询输入，避免重复加载
        cached_query_inputs = self._prepare_query_inputs()

        # 调试统计
        total_samples = 0
        early_stop_count = 0  # 提前停止的样本数
        actual_shot_counts = []  # 每个样本实际选择的 shot 数
        
        for query_input in tqdm(cached_query_inputs, desc=f"Generating with shot_num={ice_num}", ncols=100):
            bs = len(query_input[list(query_input.keys())[0]])
            init_icd_idx = torch.tensor(
                [[bos_token_id, query_token_id] for _ in range(bs)]
            ).to(self.device)
            res = self.lever_lm.generation(
                query_input=query_input,
                init_icd_idx=init_icd_idx,
                shot_num=ice_num,
                index_ds=self.index_ds,
                processor=self.processor,
                icd_image_field=self.icd_image_field,
                icd_text_field=self.icd_text_field,
                device=self.device,
                precomputed_index_emb=precomputed_index_emb,  # 传递预计算的 embedding
                disable_stop=self.disable_stop,  # 传递禁用 STOP 标志
            )
            
            # 调试：检查每个样本实际返回的 shot 数
            for r in res:
                actual_shots = len(r) - 2  # 减去 BOS 和 QUERY_TOKEN
                actual_shot_counts.append(actual_shots)
                if actual_shots < ice_num:
                    early_stop_count += 1
                total_samples += 1
            
            res = [r[2 : 2 + ice_num] for r in res]
            icd_idx_list.extend(res)
        
        # 打印调试信息
        print(f"\n{'='*60}")
        print(f"[DEBUG] shot_num={ice_num} 调试信息:")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 提前停止(STOP)的样本数: {early_stop_count} ({early_stop_count/total_samples*100:.1f}%)")
        print(f"  - 实际 shot 数分布:")
        from collections import Counter
        shot_dist = Counter(actual_shot_counts)
        for shots, count in sorted(shot_dist.items()):
            print(f"      shot={shots}: {count} 样本 ({count/total_samples*100:.1f}%)")
        print(f"  - 前5个样本的 icd_idx: {icd_idx_list[:5]}")
        print(f"{'='*60}\n")
        if self.reverse_seq:
            icd_idx_list = [list(reversed(s)) for s in icd_idx_list]

        return icd_idx_list
