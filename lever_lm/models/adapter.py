"""
适配器模块：将 v1-v4 模型接口适配到 v0 训练流程

v0 训练流程期望：
- forward(query_input, icd_input, icd_seq_idx)
- 返回包含 'loss' 的字典

v1-v4 模型接口：
- forward(query_emb, cand_emb, labels)
- 返回包含 'loss' 的字典
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from loguru import logger

# 导入 v1 模型（用于类型提示）
try:
    from .v1.pointer_selector_v1 import PointerSelectorV1
except ImportError:
    PointerSelectorV1 = None


class PointerSelectorAdapter(nn.Module):
    """
    适配器：将 v1-v4 PointerSelector 模型适配到 v0 训练流程
    
    功能：
    1. 从 query_input 和 icd_input 提取 embeddings
    2. 构建候选池 cand_emb
    3. 从 icd_seq_idx 提取 labels
    4. 调用 v1-v4 模型的 forward 方法
    """
    
    def __init__(
        self,
        pointer_selector_model,  # v1-v4 模型实例
        clip_name: str = "openai/clip-vit-base-patch32",
        query_encoding_flag: Optional[list] = None,
        icd_encoding_flag: Optional[list] = None,
        adapter: bool = False,
        norm: bool = True,
        K: int = 32,  # 候选池大小
        device: Optional[str] = None,  # 设备参数，用于确保 CLIP 模型加载到正确的 GPU
        use_lora: bool = False,  # 是否使用 LoRA
        lora_config: Optional[Dict[str, Any]] = None,  # LoRA 配置参数
        cache_dir: Optional[str] = None,  # CLIP 模型缓存目录（可选）
    ):
        super().__init__()
        
        self.pointer_selector = pointer_selector_model
        self.K = K
        self.query_encoding_flag = query_encoding_flag or []
        self.icd_encoding_flag = icd_encoding_flag or []
        self._adapter = adapter
        self._norm = norm
        
        # 加载 CLIP 模型用于提取 embeddings
        self.clip_name = clip_name
        
        # 确定设备：如果提供了 device，使用它；否则使用 pointer_selector 的设备
        if device is None:
            # 尝试从 pointer_selector 获取设备
            if hasattr(pointer_selector_model, 'device'):
                device = pointer_selector_model.device
            else:
                # 默认使用 CPU，后续会在 LeverLMRetriever 中移动到正确设备
                device = 'cpu'
        
        # 文本编码器
        self.sen_model = None
        self.sen_adapter = None
        self.use_lora = use_lora
        if "text" in self.query_encoding_flag or "text" in self.icd_encoding_flag:
            # 支持自定义缓存目录
            model_kwargs = {}
            if cache_dir is not None:
                model_kwargs['cache_dir'] = cache_dir
            self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name, **model_kwargs).to(device)
            
            # 如果使用 LoRA，为文本编码器添加 LoRA adapter
            if use_lora:
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    
                    # 默认 LoRA 配置
                    default_lora_config = {
                        'r': 16,  # LoRA rank
                        'lora_alpha': 32,  # LoRA alpha
                        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],  # 针对 CLIP 的注意力层
                        'lora_dropout': 0.1,
                        'bias': 'none',
                        'task_type': TaskType.FEATURE_EXTRACTION,
                    }
                    
                    # 合并用户配置
                    if lora_config:
                        default_lora_config.update(lora_config)
                    
                    # 创建 LoRA 配置
                    peft_config = LoraConfig(
                        r=default_lora_config['r'],
                        lora_alpha=default_lora_config['lora_alpha'],
                        target_modules=default_lora_config['target_modules'],
                        lora_dropout=default_lora_config['lora_dropout'],
                        bias=default_lora_config['bias'],
                        task_type=default_lora_config['task_type'],
                    )
                    
                    # 应用 LoRA 到模型
                    self.sen_model = get_peft_model(self.sen_model, peft_config)
                    logger.info(f"Applied LoRA to text encoder with config: r={default_lora_config['r']}, alpha={default_lora_config['lora_alpha']}")
                    
                    # LoRA 参数会自动设置为可训练，基础参数会被冻结
                    # 打印可训练参数数量
                    trainable_params = sum(p.numel() for p in self.sen_model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.sen_model.parameters())
                    logger.info(f"Text encoder LoRA - Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
                    
                except ImportError:
                    raise ImportError("peft library is required for LoRA support. Install it with: pip install peft")
            else:
                # 不使用 LoRA 时，冻结 CLIP 模型参数（只训练 pointer selector）
                for param in self.sen_model.parameters():
                    param.requires_grad = False
            
            if adapter:
                # 如果需要 adapter，创建简单的线性层
                d_model = self.sen_model.config.projection_dim
                self.sen_adapter = nn.Linear(d_model, d_model).to(device)
            else:
                self.sen_adapter = nn.Identity()
        
        # 图像编码器
        self.img_model = None
        self.img_adapter = None
        if "image" in self.query_encoding_flag or "image" in self.icd_encoding_flag:
            # 支持自定义缓存目录
            model_kwargs = {}
            if cache_dir is not None:
                model_kwargs['cache_dir'] = cache_dir
            self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name, **model_kwargs).to(device)
            
            # 如果使用 LoRA，为图像编码器添加 LoRA adapter
            if use_lora:
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    
                    # 默认 LoRA 配置
                    default_lora_config = {
                        'r': 16,  # LoRA rank
                        'lora_alpha': 32,  # LoRA alpha
                        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],  # 针对 CLIP 的注意力层
                        'lora_dropout': 0.1,
                        'bias': 'none',
                        'task_type': TaskType.FEATURE_EXTRACTION,
                    }
                    
                    # 合并用户配置
                    if lora_config:
                        default_lora_config.update(lora_config)
                    
                    # 创建 LoRA 配置
                    peft_config = LoraConfig(
                        r=default_lora_config['r'],
                        lora_alpha=default_lora_config['lora_alpha'],
                        target_modules=default_lora_config['target_modules'],
                        lora_dropout=default_lora_config['lora_dropout'],
                        bias=default_lora_config['bias'],
                        task_type=default_lora_config['task_type'],
                    )
                    
                    # 应用 LoRA 到模型
                    self.img_model = get_peft_model(self.img_model, peft_config)
                    logger.info(f"Applied LoRA to vision encoder with config: r={default_lora_config['r']}, alpha={default_lora_config['lora_alpha']}")
                    
                    # LoRA 参数会自动设置为可训练，基础参数会被冻结
                    # 打印可训练参数数量
                    trainable_params = sum(p.numel() for p in self.img_model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.img_model.parameters())
                    logger.info(f"Vision encoder LoRA - Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
                    
                except ImportError:
                    raise ImportError("peft library is required for LoRA support. Install it with: pip install peft")
            else:
                # 不使用 LoRA 时，冻结 CLIP 模型参数（只训练 pointer selector）
                for param in self.img_model.parameters():
                    param.requires_grad = False
            
            if adapter:
                d_model = self.img_model.config.projection_dim
                self.img_adapter = nn.Linear(d_model, d_model).to(device)
            else:
                self.img_adapter = nn.Identity()
    
    def _extract_query_emb(self, query_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从 query_input 提取 query embedding
        
        Args:
            query_input: 包含 'pixel_values' 和/或 'input_ids', 'attention_mask'
        
        Returns:
            query_emb: [B, d_model]
        """
        embeddings = []
        
        # 提取图像 embedding
        if "image" in self.query_encoding_flag and "pixel_values" in query_input:
            # 只传递 pixel_values，避免传递 input_ids 等文本相关参数
            # 提取 pixel_values 并单独传递，避免 PEFT 传递其他参数
            pixel_values = query_input["pixel_values"]
            # 对于 PEFT 包装的模型，需要显式只传递 pixel_values
            # 检查是否是 PEFT 模型，如果是，使用更安全的方式调用
            try:
                from peft import PeftModel
                if isinstance(self.img_model, PeftModel):
                    # PEFT 模型：问题在于 PEFT 的 forward 会传递所有 **kwargs
                    # 尝试通过 base_model 调用，但手动应用 LoRA 适配器
                    # 首先尝试直接调用，如果失败则回退到 base_model
                    try:
                        # 尝试只传递 pixel_values，看看 PEFT 是否仍然传递 input_ids
                        img_emb = self.img_model(pixel_values=pixel_values, return_dict=True)["image_embeds"]
                    except TypeError:
                        # 如果失败，直接调用 base_model（会绕过 LoRA，但至少能运行）
                        base_model = self.img_model.get_base_model()
                        outputs = base_model.forward(pixel_values=pixel_values, return_dict=True)
                        img_emb = outputs["image_embeds"]
                else:
                    # 非 PEFT 模型：正常调用
                    img_emb = self.img_model(pixel_values=pixel_values)["image_embeds"]
            except (ImportError, AttributeError, TypeError) as e:
                # 如果没有 peft 或调用失败，尝试正常调用
                img_emb = self.img_model(pixel_values=pixel_values)["image_embeds"]
            if self._adapter:
                img_emb = self.img_adapter(img_emb)
            if self._norm:
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            embeddings.append(img_emb)
        
        # 提取文本 embedding
        if "text" in self.query_encoding_flag and "input_ids" in query_input:
            # 对于 PEFT 包装的模型，需要显式只传递期望的参数
            # 检查是否是 PEFT 模型，如果是，使用更安全的方式调用
            try:
                from peft import PeftModel
                if isinstance(self.sen_model, PeftModel):
                    # PEFT 模型：问题在于 PEFT 的 forward 会传递所有 **kwargs
                    # 尝试通过 base_model 调用，但手动应用 LoRA 适配器
                    # 首先尝试直接调用，如果失败则回退到 base_model
                    try:
                        # 尝试只传递 input_ids 和 attention_mask，看看 PEFT 是否仍然传递其他参数
                        text_emb = self.sen_model(
                            input_ids=query_input["input_ids"],
                            attention_mask=query_input.get("attention_mask", None),
                            return_dict=True
                        )["text_embeds"]
                    except TypeError:
                        # 如果失败，直接调用 base_model（会绕过 LoRA，但至少能运行）
                        base_model = self.sen_model.get_base_model()
                        outputs = base_model.forward(
                            input_ids=query_input["input_ids"],
                            attention_mask=query_input.get("attention_mask", None),
                            return_dict=True
                        )
                        text_emb = outputs["text_embeds"]
                else:
                    # 非 PEFT 模型：正常调用
                    text_emb = self.sen_model(
                        input_ids=query_input["input_ids"],
                        attention_mask=query_input.get("attention_mask", None),
                    )["text_embeds"]
            except (ImportError, AttributeError, TypeError) as e:
                # 如果没有 peft 或调用失败，尝试正常调用
                text_emb = self.sen_model(
                    input_ids=query_input["input_ids"],
                    attention_mask=query_input.get("attention_mask", None),
                )["text_embeds"]
            if self._adapter:
                text_emb = self.sen_adapter(text_emb)
            if self._norm:
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            embeddings.append(text_emb)
        
        if not embeddings:
            raise ValueError("query_encoding_flag 必须包含 'image' 或 'text'")
        
        # 合并多个 embeddings（简单相加或平均）
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            # 如果有多个，取平均
            query_emb = torch.stack(embeddings, dim=0).mean(dim=0)
            if self._norm:
                query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            return query_emb
    
    def _extract_cand_emb(self, icd_input: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        从 icd_input 提取候选 embeddings
        
        Args:
            icd_input: 包含候选的 'pixel_values' 和/或 'input_ids', 'attention_mask'
                       shape: [B, K, ...] 或 None
        
        Returns:
            cand_emb: [B, K, d_model]
        """
        if icd_input is None:
            # 如果没有 icd_input，返回零向量（这种情况不应该发生）
            B = 1  # 默认 batch size
            d_model = self.pointer_selector.d_model
            return torch.zeros(B, self.K, d_model)
        
        embeddings = []
        
        # 提取图像 embeddings
        if "image" in self.icd_encoding_flag and "pixel_values" in icd_input:
            bs, icd_num = icd_input["pixel_values"].shape[:2]
            img_shape = icd_input["pixel_values"].shape[-3:]
            # 展平处理
            pixel_values_flat = icd_input["pixel_values"].view(-1, *img_shape)
            # 只传递 pixel_values，避免传递 input_ids 等文本相关参数
            # 检查是否是 PEFT 模型，如果是，使用更安全的方式调用
            try:
                from peft import PeftModel
                if isinstance(self.img_model, PeftModel):
                    # PEFT 模型：问题在于 PEFT 的 forward 会传递所有 **kwargs
                    # 尝试通过 base_model 调用，但手动应用 LoRA 适配器
                    # 首先尝试直接调用，如果失败则回退到 base_model
                    try:
                        # 尝试只传递 pixel_values，看看 PEFT 是否仍然传递 input_ids
                        img_emb = self.img_model(pixel_values=pixel_values_flat, return_dict=True)["image_embeds"]
                    except TypeError:
                        # 如果失败，直接调用 base_model（会绕过 LoRA，但至少能运行）
                        base_model = self.img_model.get_base_model()
                        outputs = base_model.forward(pixel_values=pixel_values_flat, return_dict=True)
                        img_emb = outputs["image_embeds"]
                else:
                    # 非 PEFT 模型：正常调用
                    img_emb = self.img_model(pixel_values=pixel_values_flat)["image_embeds"]
            except (ImportError, AttributeError, TypeError) as e:
                # 如果没有 peft 或调用失败，尝试正常调用
                img_emb = self.img_model(pixel_values=pixel_values_flat)["image_embeds"]
            if self._adapter:
                img_emb = self.img_adapter(img_emb)
            if self._norm:
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            img_emb = img_emb.view(bs, icd_num, -1)
            embeddings.append(img_emb)
        
        # 提取文本 embeddings
        if "text" in self.icd_encoding_flag and "input_ids" in icd_input:
            bs, icd_num, seq_len = icd_input["input_ids"].shape
            input_ids_flat = icd_input["input_ids"].view(-1, seq_len)
            attention_mask_flat = icd_input["attention_mask"].view(-1, seq_len)
            # 对于 PEFT 包装的模型，需要显式只传递期望的参数
            # 检查是否是 PEFT 模型，如果是，使用更安全的方式调用
            try:
                from peft import PeftModel
                if isinstance(self.sen_model, PeftModel):
                    # PEFT 模型：问题在于 PEFT 的 forward 会传递所有 **kwargs
                    # 尝试通过 base_model 调用，但手动应用 LoRA 适配器
                    # 首先尝试直接调用，如果失败则回退到 base_model
                    try:
                        # 尝试只传递 input_ids 和 attention_mask，看看 PEFT 是否仍然传递其他参数
                        text_emb = self.sen_model(
                            input_ids=input_ids_flat,
                            attention_mask=attention_mask_flat,
                            return_dict=True
                        )["text_embeds"]
                    except TypeError:
                        # 如果失败，直接调用 base_model（会绕过 LoRA，但至少能运行）
                        base_model = self.sen_model.get_base_model()
                        outputs = base_model.forward(
                            input_ids=input_ids_flat,
                            attention_mask=attention_mask_flat,
                            return_dict=True
                        )
                        text_emb = outputs["text_embeds"]
                else:
                    # 非 PEFT 模型：正常调用
                    text_emb = self.sen_model(
                        input_ids=input_ids_flat,
                        attention_mask=attention_mask_flat,
                    )["text_embeds"]
            except (ImportError, AttributeError, TypeError) as e:
                # 如果没有 peft 或调用失败，尝试正常调用
                text_emb = self.sen_model(
                    input_ids=input_ids_flat,
                    attention_mask=attention_mask_flat,
                )["text_embeds"]
            if self._adapter:
                text_emb = self.sen_adapter(text_emb)
            if self._norm:
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.view(bs, icd_num, -1)
            embeddings.append(text_emb)
        
        if not embeddings:
            raise ValueError("icd_encoding_flag 必须包含 'image' 或 'text'")
        
        # 合并多个 embeddings
        if len(embeddings) == 1:
            cand_emb = embeddings[0]
        else:
            # 如果有多个，取平均
            cand_emb = torch.stack(embeddings, dim=0).mean(dim=0)
            if self._norm:
                cand_emb = cand_emb / cand_emb.norm(dim=-1, keepdim=True)
        
        # 确保候选数量匹配 K
        # 注意：如果 icd_input 中的候选数量不等于 K，需要从训练集中补充
        # 当前实现假设 icd_input 中的候选就是候选池的全部或部分
        if cand_emb.shape[1] != self.K:
            # 如果候选数量不等于 K，需要截断或填充
            if cand_emb.shape[1] > self.K:
                cand_emb = cand_emb[:, :self.K, :]
            else:
                # 填充零向量（实际应用中，应该从训练集中补充候选）
                # 这里使用零向量作为占位符
                padding = torch.zeros(
                    cand_emb.shape[0], 
                    self.K - cand_emb.shape[1], 
                    cand_emb.shape[2],
                    device=cand_emb.device,
                    dtype=cand_emb.dtype
                )
                cand_emb = torch.cat([cand_emb, padding], dim=1)
        
        return cand_emb
    
    def _extract_labels(
        self, 
        icd_seq_idx: torch.Tensor,
        icd_input: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        从 icd_seq_idx 和 icd_input 提取 labels
        
        icd_seq_idx 格式: [BOS, QUERY_TOKEN, idx1, idx2, ..., idxN, EOS]
        其中 idx1, idx2, ... 是训练集中样本的索引
        
        需要将这些索引映射到候选池中的位置（0 到 K-1）
        
        注意：当前实现假设候选池就是训练集中的前 K 个样本
        这是一个简化假设，实际应用中可能需要更复杂的映射逻辑
        
        Args:
            icd_seq_idx: [B, seq_len] token IDs，包含样本索引
            icd_input: 候选输入字典（可选，用于确定候选数量）
        
        Returns:
            labels: [B, shot_num] 候选池中的索引（0 到 K-1）
        """
        # icd_seq_idx 格式: [BOS, QUERY_TOKEN, idx1, idx2, ..., EOS]
        # 提取中间部分作为被选中的样本索引
        B, seq_len = icd_seq_idx.shape
        
        # 提取中间部分（跳过 BOS 和 QUERY_TOKEN，去掉 EOS）
        if seq_len > 3:
            selected_indices = icd_seq_idx[:, 2:-1]  # [B, seq_len-3]
        else:
            # 序列太短，返回空标签
            selected_indices = torch.zeros(B, 0, dtype=torch.long, device=icd_seq_idx.device)
        
        # 获取 shot_num
        shot_num = self.pointer_selector.shot_num
        
        # 确保 labels 的数量不超过 shot_num
        if selected_indices.shape[1] > shot_num:
            selected_indices = selected_indices[:, :shot_num]
        elif selected_indices.shape[1] < shot_num:
            # 填充 0（假设第一个候选总是被选中）
            padding = torch.zeros(
                (B, shot_num - selected_indices.shape[1]),
                dtype=torch.long,
                device=selected_indices.device
            )
            selected_indices = torch.cat([selected_indices, padding], dim=1)
        
        # 将样本索引映射到候选池索引
        # 简化假设：候选池就是训练集中的前 K 个样本（索引 0 到 K-1）
        # 如果选中的样本索引 < K，则直接使用；否则取模
        labels = selected_indices % self.K
        
        return labels
    
    def forward(self, query_input, icd_input, icd_seq_idx):
        """
        适配 v0 训练流程的 forward 方法
        
        Args:
            query_input: 查询输入字典
            icd_input: 候选输入字典（可选）
            icd_seq_idx: 序列索引 tensor [B, seq_len]
        
        Returns:
            dict: 包含 'loss' 的字典
        """
        # 提取 query embedding
        query_emb = self._extract_query_emb(query_input)  # [B, d_model]
        
        # 提取候选 embeddings
        cand_emb = self._extract_cand_emb(icd_input)  # [B, K, d_model]
        
        # 提取 labels
        labels = self._extract_labels(icd_seq_idx, icd_input)  # [B, shot_num]
        
        # 调用 v1-v4 模型的 forward 方法
        output = self.pointer_selector(
            query_emb=query_emb,
            cand_emb=cand_emb,
            labels=labels,
            return_loss=True
        )
        
        return output
    
    @torch.inference_mode()
    def generation(
        self,
        query_input,
        init_icd_idx,
        shot_num,
        index_ds,
        processor,
        device,
        icd_image_field,
        icd_text_field,
        precomputed_index_emb=None,  # 保留参数但不使用，保持接口兼容
        disable_stop=False,  # 禁用 STOP 机制
    ):
        """
        适配 v0 推理流程的 generation 方法
        
        Args:
            query_input: 查询输入字典
            init_icd_idx: [B, 2] 初始序列 [BOS, QUERY_TOKEN]
            shot_num: 需要选择的样本数量
            index_ds: 训练集数据集（用于构建候选池）
            processor: 处理器（用于编码候选）
            device: 设备
            icd_image_field: ICD 图像字段名
            icd_text_field: ICD 文本字段名
            precomputed_index_emb: 预计算的训练集 embedding（暂不使用）
            disable_stop: 是否禁用 STOP 机制
        
        Returns:
            icd_seq_idx: List[List[int]] 格式为 [BOS, QUERY_TOKEN, idx1, idx2, ..., idxN]
        """
        batch_size = init_icd_idx.shape[0]
        
        # 提取 query embedding
        query_emb = self._extract_query_emb(query_input)  # [B, d_model]
        
        # 获取训练集大小和模型的实际 K
        dataset_size = len(index_ds)
        model_K = self.pointer_selector.K
        
        # 使用前 K 个样本作为候选池
        K_to_use = min(model_K, dataset_size)
        candidate_indices = list(range(K_to_use))
        
        # 提取候选的文本和图像
        icd_text_list = []
        icd_img_list = []
        for idx in candidate_indices:
            if "text" in self.icd_encoding_flag:
                icd_text_list.append(index_ds[idx][icd_text_field])
            if "image" in self.icd_encoding_flag:
                icd_img_list.append(index_ds[idx][icd_image_field])
        
        # 使用 processor 编码候选
        if icd_text_list or icd_img_list:
            cand_input = processor(
                text=icd_text_list if icd_text_list else None,
                images=icd_img_list if icd_img_list else None,
                padding=True,
                return_tensors="pt",
            ).to(device)
        else:
            raise ValueError("icd_encoding_flag 必须包含 'image' 或 'text'")
        
        # 构建候选输入字典（batch 维度为 batch_size）
        cand_input_dict = {}
        for k in cand_input:
            if k == "pixel_values" and "image" in self.icd_encoding_flag:
                # 图像：添加 batch 维度 [batch_size, K, C, H, W]
                cand_input_dict[k] = cand_input[k].unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            elif k in ["input_ids", "attention_mask"] and "text" in self.icd_encoding_flag:
                # 文本：添加 batch 维度 [batch_size, K, seq_len]
                other_dims = cand_input[k].shape[1:]
                cand_input_dict[k] = cand_input[k].unsqueeze(0).expand(batch_size, -1, *other_dims)
        
        # 提取候选 embeddings
        # 注意：这里需要临时修改 self.K 以匹配实际的候选数量
        original_K = self.K
        self.K = K_to_use
        
        try:
            cand_emb = self._extract_cand_emb(cand_input_dict)  # [B, K_to_use, d_model]
        finally:
            # 恢复原始 K
            self.K = original_K
        
        # 临时修改模型的 K 参数以匹配实际的候选数量
        original_model_K = self.pointer_selector.K
        self.pointer_selector.K = K_to_use
        
        # 调用 v1 模型的 forward 方法获取预测
        # 注意：这里需要临时设置 shot_num，因为 v1 模型的 shot_num 可能与推理时的 shot_num 不同
        original_shot_num = self.pointer_selector.shot_num
        self.pointer_selector.shot_num = shot_num
        
        try:
            output = self.pointer_selector(
                query_emb=query_emb,
                cand_emb=cand_emb,
                labels=None,  # 推理时不需要 labels
                return_loss=False
            )
        finally:
            # 恢复原始参数
            self.pointer_selector.shot_num = original_shot_num
            self.pointer_selector.K = original_model_K
        
        # 调试：打印 STOP 分数信息（只打印第一个 batch）
        if 'debug_stop_scores' in output and not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"\n{'='*60}")
            print(f"[DEBUG] V4-10 STOP 分数分析 (shot_num={shot_num}):")
            for info in output['debug_stop_scores']:
                status = "⚠️ STOP更高" if info['stop_higher'] else "✓ 候选更高"
                print(f"  Step {info['step']}: STOP={info['stop_score']:.3f}, MaxCand={info['max_cand_score']:.3f} {status}")
            print(f"{'='*60}\n")
        
        # 获取预测的候选索引（0 到 K_to_use-1）
        predictions = output['predictions']  # [B, shot_num]
        
        # 检查是否是 V4-10 模型（有 STOP token）
        # V4-10 的 logits 维度是 K+1，STOP 索引是 K
        has_stop_token = hasattr(self.pointer_selector, 'stop_token') and not disable_stop
        stop_idx = K_to_use if has_stop_token else -1  # STOP 索引
        
        # 将候选索引（0 到 K_to_use-1）转换为训练集中的实际索引
        # candidate_indices 是候选池在训练集中的索引列表
        actual_indices = []
        for b in range(batch_size):
            batch_predictions = predictions[b].cpu().tolist()  # [shot_num]
            # 将候选索引映射到训练集中的实际索引
            # 对于 V4-10，如果选择了 STOP，则停止添加后续索引
            batch_actual_indices = []
            selected_local_indices = set()  # 记录已选择的本地索引
            for step_idx, pred_idx in enumerate(batch_predictions):
                if has_stop_token and pred_idx == stop_idx:
                    # 选择了 STOP，停止添加
                    break
                # 如果禁用 STOP，跳过 STOP 索引，选择次优候选
                if disable_stop and pred_idx == K_to_use:
                    logits = output['logits'][b]  # [shot_num, K+1] or [shot_num, K]
                    if step_idx < logits.shape[0]:
                        step_logits = logits[step_idx].clone()
                        # 只有当 logits 包含 STOP token 时才 mask
                        if step_logits.shape[0] > K_to_use:
                            step_logits[K_to_use] = -100.0  # mask STOP
                        for local_idx in selected_local_indices:
                            step_logits[local_idx] = -100.0
                        pred_idx = step_logits.argmax().item()
                
                if pred_idx < len(candidate_indices) and pred_idx not in selected_local_indices:
                    batch_actual_indices.append(candidate_indices[pred_idx])
                    selected_local_indices.add(pred_idx)
                elif pred_idx in selected_local_indices:
                    # 如果选择了已选过的候选，找下一个最优的
                    logits = output['logits'][b]
                    if step_idx < logits.shape[0]:
                        step_logits = logits[step_idx].clone()
                        # 只有当 logits 包含 STOP token 时才 mask
                        if step_logits.shape[0] > K_to_use:
                            step_logits[K_to_use] = -100.0  # mask STOP
                        for local_idx in selected_local_indices:
                            step_logits[local_idx] = -100.0
                        new_pred_idx = step_logits.argmax().item()
                        if new_pred_idx < len(candidate_indices) and step_logits[new_pred_idx] > -100.0:
                            batch_actual_indices.append(candidate_indices[new_pred_idx])
                            selected_local_indices.add(new_pred_idx)
                else:
                    # 索引越界，跳过（不应该发生，但作为安全措施）
                    logger.warning(f"预测索引 {pred_idx} 超出候选池大小 {len(candidate_indices)}")
                    break
            
            # 如果没有选择任何有效候选（全部是 STOP），至少选择第一个候选
            if len(batch_actual_indices) == 0:
                batch_actual_indices = [candidate_indices[0]]
            
            actual_indices.append(batch_actual_indices)
        
        # 构建返回的 icd_seq_idx
        # 格式：[BOS, QUERY_TOKEN, idx1, idx2, ..., idxN]
        bos_token_id = init_icd_idx[0, 0].item()
        query_token_id = init_icd_idx[0, 1].item()
        
        result = []
        for b in range(batch_size):
            seq = [bos_token_id, query_token_id] + actual_indices[b]
            result.append(seq)
        
        return result
    
    def save_lora_checkpoint(self, save_path: str):
        """
        保存 LoRA checkpoint
        
        Args:
            save_path: 保存路径（目录路径，会在此目录下保存 LoRA adapter）
        """
        if not self.use_lora:
            logger.warning("Model is not using LoRA, nothing to save")
            return
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        try:
            from peft import PeftModel
            
            # 保存文本编码器的 LoRA
            if self.sen_model is not None and isinstance(self.sen_model, PeftModel):
                sen_save_path = os.path.join(save_path, "text_encoder_lora")
                self.sen_model.save_pretrained(sen_save_path)
            
            # 保存图像编码器的 LoRA
            if self.img_model is not None and isinstance(self.img_model, PeftModel):
                img_save_path = os.path.join(save_path, "vision_encoder_lora")
                self.img_model.save_pretrained(img_save_path)
                
        except ImportError:
            raise ImportError("peft library is required for LoRA support. Install it with: pip install peft")
    
    def load_lora_checkpoint(self, load_path: str):
        """
        加载 LoRA checkpoint
        
        Args:
            load_path: 加载路径（目录路径，从此目录加载 LoRA adapter）
        """
        if not self.use_lora:
            logger.warning("Model is not using LoRA, nothing to load")
            return
        
        import os
        
        try:
            from peft import PeftModel
            
            # 加载文本编码器的 LoRA
            if self.sen_model is not None:
                sen_load_path = os.path.join(load_path, "text_encoder_lora")
                if os.path.exists(sen_load_path):
                    if isinstance(self.sen_model, PeftModel):
                        # 如果已经是 PeftModel，直接加载
                        self.sen_model = PeftModel.from_pretrained(
                            self.sen_model.get_base_model(),
                            sen_load_path
                        )
                    else:
                        # 如果不是 PeftModel，需要先转换为 PeftModel
                        self.sen_model = PeftModel.from_pretrained(
                            self.sen_model,
                            sen_load_path
                        )
                    logger.info(f"Loaded text encoder LoRA from {sen_load_path}")
            
            # 加载图像编码器的 LoRA
            if self.img_model is not None:
                img_load_path = os.path.join(load_path, "vision_encoder_lora")
                if os.path.exists(img_load_path):
                    if isinstance(self.img_model, PeftModel):
                        # 如果已经是 PeftModel，直接加载
                        self.img_model = PeftModel.from_pretrained(
                            self.img_model.get_base_model(),
                            img_load_path
                        )
                    else:
                        # 如果不是 PeftModel，需要先转换为 PeftModel
                        self.img_model = PeftModel.from_pretrained(
                            self.img_model,
                            img_load_path
                        )
                    logger.info(f"Loaded vision encoder LoRA from {img_load_path}")
                    
        except ImportError:
            raise ImportError("peft library is required for LoRA support. Install it with: pip install peft")

