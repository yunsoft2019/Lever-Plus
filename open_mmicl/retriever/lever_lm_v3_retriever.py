"""
LeverLM V3 Retriever - 使用GRPO训练后的V3模型进行范例检索

与V2的主要区别：
1. 使用预计算的CLIP embedding而不是实时处理
2. 直接使用PointerSelectorV3模型
3. 支持从GRPO检查点加载模型
"""

from typing import List, Optional, Dict
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from open_mmicl.retriever.base_retriever import BaseRetriever


class LeverLMV3Retriever(BaseRetriever):
    """使用V3模型（GRPO训练后）进行范例检索"""
    
    def __init__(
        self,
        index_ds,
        test_ds,
        model: torch.nn.Module,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_indices: List[int],
        device: str = "cuda:0",
        batch_size: int = 32,
    ):
        """
        初始化V3 Retriever
        
        Args:
            index_ds: 索引数据集（训练集）
            test_ds: 测试数据集
            model: PointerSelectorV3模型
            query_embeddings: 查询embedding [N_test, d]
            candidate_embeddings: 候选embedding [K, d]
            candidate_indices: 候选池中样本在原始数据集中的索引
            device: 计算设备
            batch_size: 批次大小
        """
        super().__init__(index_ds, test_ds)
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        
        # embedding
        self.query_embeddings = query_embeddings.to(device)
        self.candidate_embeddings = candidate_embeddings.to(device)
        
        # 候选池索引映射：位置 -> 原始索引
        self.candidate_indices = candidate_indices
        self.pos_to_idx = {pos: idx for pos, idx in enumerate(candidate_indices)}
        
    def retrieve(self, ice_num: int) -> List[List[int]]:
        """
        检索范例
        
        Args:
            ice_num: 每个查询需要的范例数量
            
        Returns:
            List[List[int]]: 每个测试样本的范例索引列表
        """
        return self.v3_generation(ice_num)
    
    @torch.inference_mode()
    def v3_generation(self, ice_num: int) -> List[List[int]]:
        """
        使用V3模型生成范例索引
        
        Args:
            ice_num: 每个查询需要的范例数量
            
        Returns:
            List[List[int]]: 每个测试样本的范例索引列表（原始数据集索引）
        """
        all_predictions = []
        num_samples = len(self.test_ds)
        
        # 获取测试集样本的索引（用于查找对应的embedding）
        # 假设test_ds有一个索引字段或者按顺序对应
        test_indices = list(range(num_samples))
        
        # 分批处理
        for start_idx in tqdm(range(0, num_samples, self.batch_size), 
                              desc=f"V3 Retrieval (shot={ice_num})"):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = test_indices[start_idx:end_idx]
            
            # 获取batch的query embedding
            # 注意：需要根据实际的测试集索引来获取embedding
            # 这里假设测试集的索引与embedding的索引对应
            batch_query_emb = self.query_embeddings[batch_indices]  # [B, d]
            
            # 候选embedding对所有batch共享
            batch_cand_emb = self.candidate_embeddings.unsqueeze(0).expand(
                len(batch_indices), -1, -1
            )  # [B, K, d]
            
            # 使用模型预测
            predictions = self.model.predict(
                query_emb=batch_query_emb,
                cand_emb=batch_cand_emb,
                shot_num=ice_num
            )  # [B, shot_num]
            
            # 将位置索引转换为原始数据集索引
            for pred in predictions:
                original_indices = [self.pos_to_idx[p.item()] for p in pred]
                all_predictions.append(original_indices)
        
        return all_predictions


def load_v3_retriever(
    grpo_ckpt_path: str,
    index_ds,
    test_ds,
    img_emb_path: str,
    candidate_indices: List[int],
    test_indices: Optional[List[int]] = None,
    device: str = "cuda:0",
    batch_size: int = 32,
) -> LeverLMV3Retriever:
    """
    从GRPO检查点加载V3 Retriever
    
    Args:
        grpo_ckpt_path: GRPO检查点路径
        index_ds: 索引数据集
        test_ds: 测试数据集
        img_emb_path: 图像embedding路径
        candidate_indices: 候选池索引
        test_indices: 测试集样本在embedding中的索引（如果None则使用0到len(test_ds)）
        device: 计算设备
        batch_size: 批次大小
        
    Returns:
        LeverLMV3Retriever实例
    """
    from lever_lm.models.v3 import PointerSelectorV3
    
    # 加载embedding
    img_emb_data = torch.load(img_emb_path, weights_only=False)
    if not isinstance(img_emb_data, torch.Tensor):
        img_emb_data = torch.from_numpy(img_emb_data).float()
    
    # 提取候选池embedding
    candidate_embeddings = img_emb_data[candidate_indices]  # [K, d]
    
    # 提取测试集embedding
    if test_indices is None:
        test_indices = list(range(len(test_ds)))
    query_embeddings = img_emb_data[test_indices]  # [N_test, d]
    
    # 加载模型
    d_model = img_emb_data.shape[1]
    K = len(candidate_indices)
    
    # 从检查点获取shot_num
    ckpt = torch.load(grpo_ckpt_path, map_location='cpu', weights_only=False)
    model_state = ckpt.get('model_state_dict', ckpt)
    
    # 尝试推断shot_num（从模型结构或默认值）
    shot_num = 2  # 默认值
    
    model = PointerSelectorV3(
        d_model=d_model,
        K=K,
        shot_num=shot_num
    )
    model.load_state_dict(model_state)
    
    return LeverLMV3Retriever(
        index_ds=index_ds,
        test_ds=test_ds,
        model=model,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        candidate_indices=candidate_indices,
        device=device,
        batch_size=batch_size
    )
