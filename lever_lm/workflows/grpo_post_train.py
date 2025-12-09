"""
GRPO Post-Training 训练脚本

来自强化学习.md 2.2节 详细训练流程：
- 阶段1：数据准备
- 阶段2：RCE预热（1-2 epochs）
- 阶段3：GRPO训练（2-5 epochs）

作者: Lever-Plus Team
日期: 2025-12-02
"""

import os
import json
import argparse
from typing import Dict, Optional
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from lever_lm.models.v3 import (
    PointerSelectorV3,
    BeamDataset,
    BeamDatasetWithEmbedding,
    RLBeamDatasetWithEmbedding,
    collate_fn_v3,
    collate_fn_rl_v3,
    load_beam_data,
    split_beam_data
)
from lever_lm.utils.reward_utils import (
    compute_temperature_schedule,
    adaptive_kl_beta
)


class GRPOTrainer:
    """GRPO Post-Training 训练器"""
    
    def __init__(
        self,
        model: PointerSelectorV3,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        # RCE配置
        rce_epochs: int = 1,
        rce_lr: float = 1e-5,
        rce_temp_start: float = 2.0,
        rce_temp_end: float = 0.5,
        use_top1_only: bool = False,  # 是否只使用Top-1 beam（回归V2方式）
        # GRPO配置
        grpo_epochs: int = 3,
        grpo_lr: float = 5e-6,
        grpo_early_epochs: int = 1,
        grpo_early_top_k: int = 3,
        grpo_late_top_k: int = 5,
        # KL配置
        kl_target_min: float = 0.01,
        kl_target_max: float = 0.1,
        kl_adjustment_factor: float = 1.5,
        # 训练配置
        gradient_clip: float = 1.0,
        warmup_ratio: float = 0.1,
        log_every: int = 10,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # RCE配置
        self.rce_epochs = rce_epochs
        self.rce_lr = rce_lr
        self.rce_temp_start = rce_temp_start
        self.rce_temp_end = rce_temp_end
        self.use_top1_only = use_top1_only
        
        # GRPO配置
        self.grpo_epochs = grpo_epochs
        self.grpo_lr = grpo_lr
        self.grpo_early_epochs = grpo_early_epochs
        self.grpo_early_top_k = grpo_early_top_k
        self.grpo_late_top_k = grpo_late_top_k
        
        # KL配置
        self.kl_target_min = kl_target_min
        self.kl_target_max = kl_target_max
        self.kl_adjustment_factor = kl_adjustment_factor
        
        # 训练配置
        self.gradient_clip = gradient_clip
        self.warmup_ratio = warmup_ratio
        self.log_every = log_every
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_old_log_probs(self, sft_model: PointerSelectorV3) -> Dict[int, torch.Tensor]:
        """
        预计算SFT模型的log概率
        
        来自强化学习.md 2.2节 阶段1：
        Old_log_probs: 从SFT模型计算（冻结参数）
        """
        print("计算SFT模型的old_log_probs...")
        sft_model.eval()
        old_log_probs_dict = {}
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Computing old_log_probs"):
                query_emb = batch["query_emb"].to(self.device)
                cand_emb = batch["cand_emb"].to(self.device)
                beam_labels = batch["beam_labels"].to(self.device)
                
                # 处理不同的batch格式
                if isinstance(batch["query_id"], (int, str)):
                    # 新格式（batch_size=1）：query_id是单个值
                    query_ids = [batch["query_id"]]
                else:
                    # 旧格式：query_ids是列表
                    query_ids = batch["query_ids"]
                
                log_probs = sft_model.compute_log_probs_per_beam(
                    query_emb, cand_emb, beam_labels
                )
                
                for i, qid in enumerate(query_ids):
                    old_log_probs_dict[qid] = log_probs[i].cpu()
        
        return old_log_probs_dict
    
    def train_rce_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        RCE预热阶段训练一个epoch
        
        来自强化学习.md 2.2节 阶段2：
        - 损失：L_RCE = Σ w_i * CE(π_new, labels_i)
        - 温度调度：τ从2.0线性降到0.5
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 计算当前epoch的温度
        total_steps = total_epochs * len(self.train_loader)
        current_step = epoch * len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 计算当前步骤的温度
            step = current_step + batch_idx
            temperature = compute_temperature_schedule(
                step, total_steps,
                self.rce_temp_start, self.rce_temp_end
            )
            
            query_emb = batch["query_emb"].to(self.device)
            cand_emb = batch["cand_emb"].to(self.device)
            beam_labels = batch["beam_labels"].to(self.device)
            beam_rewards_raw = batch["beam_rewards_raw"].to(self.device)
            
            optimizer.zero_grad()
            
            loss = self.model.compute_rce_loss(
                query_emb, cand_emb, beam_labels, beam_rewards_raw,
                temperature=temperature,
                use_top1_only=self.use_top1_only
            )
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return {"rce_loss": total_loss / num_batches}
    
    def train_grpo_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        old_log_probs_dict: Dict[int, torch.Tensor],
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        GRPO训练阶段训练一个epoch
        
        来自强化学习.md 2.2节 阶段3：
        - 损失：L_GRPO = L_PPO + β * L_KL
        - 课程学习：早期top-3，后期所有beam
        """
        self.model.train()
        
        # 课程学习：决定使用多少beam
        if epoch < self.grpo_early_epochs:
            use_top_k = self.grpo_early_top_k
        else:
            use_top_k = self.grpo_late_top_k
        
        metrics = {
            "grpo_loss": 0.0,
            "ppo_loss": 0.0,
            "kl_loss": 0.0,
            "kl": 0.0,
            "mean_ratio": 0.0,
            "mean_advantage": 0.0,
            "std_advantage": 0.0,
            "max_advantage": 0.0
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            query_emb = batch["query_emb"].to(self.device)
            cand_emb = batch["cand_emb"].to(self.device)
            beam_labels = batch["beam_labels"].to(self.device)
            beam_rewards = batch["beam_rewards"].to(self.device)
            
            # 处理不同的batch格式
            if isinstance(batch.get("query_id"), (int, str)):
                # 新格式（batch_size=1）：query_id是单个值
                query_ids = [batch["query_id"]]
            else:
                # 旧格式：query_ids是列表
                query_ids = batch["query_ids"]
            
            # 获取old_log_probs
            old_log_probs = torch.stack([
                old_log_probs_dict[qid] for qid in query_ids
            ]).to(self.device)
            
            optimizer.zero_grad()
            
            result = self.model.compute_grpo_loss(
                query_emb, cand_emb, beam_labels, beam_rewards,
                old_log_probs, use_top_k=use_top_k
            )
            
            loss = result["loss"]
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            optimizer.step()
            
            # 累积指标
            for k in metrics:
                if k == "grpo_loss":
                    metrics[k] += loss.item()
                elif k in result:
                    metrics[k] += result[k].item()
            num_batches += 1
            
            # KL自适应调整
            current_kl = result["kl"].item()
            self.model.kl_beta = adaptive_kl_beta(
                current_kl, self.model.kl_beta,
                self.kl_target_min, self.kl_target_max,
                self.kl_adjustment_factor
            )
        
        # 平均指标
        for k in metrics:
            metrics[k] /= num_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            query_emb = batch["query_emb"].to(self.device)
            cand_emb = batch["cand_emb"].to(self.device)
            beam_labels = batch["beam_labels"].to(self.device)
            beam_rewards_raw = batch["beam_rewards_raw"].to(self.device)
            
            loss = self.model.compute_rce_loss(
                query_emb, cand_emb, beam_labels, beam_rewards_raw,
                temperature=1.0
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    def save_checkpoint(self, epoch: int, phase: str, metrics: Dict[str, float], verbose: bool = False):
        """保存检查点"""
        ckpt_path = os.path.join(self.save_dir, f"{phase}_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "kl_beta": self.model.kl_beta,
            "metrics": metrics
        }, ckpt_path)
        if verbose:
            print(f"✓ 保存检查点: {ckpt_path}")
    
    def train(self, sft_checkpoint: Optional[str] = None):
        """
        完整训练流程
        
        来自强化学习.md 2.2节：
        1. 加载SFT模型，计算old_log_probs
        2. RCE预热
        3. GRPO训练
        """
        print("="*70)
        print("GRPO Post-Training")
        print("="*70)
        
        # 加载SFT检查点
        if sft_checkpoint:
            print(f"加载SFT检查点: {sft_checkpoint}")
            ckpt = torch.load(sft_checkpoint, map_location=self.device)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif "state_dict" in ckpt:
                # PyTorch Lightning格式
                state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
                self.model.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        
        # 创建SFT模型副本用于计算old_log_probs
        sft_model = PointerSelectorV3(
            d_model=self.model.d_model,
            K=self.model.K,
            shot_num=self.model.shot_num,
            num_layers=self.model.num_layers
        ).to(self.device)
        sft_model.load_state_dict(self.model.state_dict())
        sft_model.eval()
        
        # 计算old_log_probs
        old_log_probs_dict = self.compute_old_log_probs(sft_model)
        del sft_model  # 释放内存
        
        # ========== 阶段2：RCE预热 ==========
        print("\n" + "="*80)
        print("阶段2：RCE预热")
        print("="*80)
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Temperature':>12}")
        print("-"*80)
        
        rce_optimizer = AdamW(self.model.parameters(), lr=self.rce_lr)
        
        for epoch in range(self.rce_epochs):
            metrics = self.train_rce_epoch(rce_optimizer, epoch, self.rce_epochs)
            val_metrics = self.validate()
            metrics.update(val_metrics)
            
            # 计算当前温度
            total_steps = self.rce_epochs * len(self.train_loader)
            current_step = (epoch + 1) * len(self.train_loader)
            temperature = compute_temperature_schedule(
                current_step, total_steps, self.rce_temp_start, self.rce_temp_end
            )
            
            print(f"{epoch+1:6d} {metrics['rce_loss']:12.5f} {metrics['val_loss']:12.5f} {temperature:12.3f}")
            self.save_checkpoint(epoch, "rce", metrics)
        
        print("="*80)
        
        # ========== 阶段3：GRPO训练 ==========
        if self.grpo_epochs > 0:
            print("\n" + "="*100)
            print("阶段3：GRPO训练")
            print("="*100)
            print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'PPO Loss':>12} {'KL':>10} {'Adv Std':>10} {'Adv Max':>10} {'Beta':>8}")
            print("-"*100)
            
            grpo_optimizer = AdamW(self.model.parameters(), lr=self.grpo_lr)
            
            for epoch in range(self.grpo_epochs):
                metrics = self.train_grpo_epoch(
                    grpo_optimizer, old_log_probs_dict,
                    epoch, self.grpo_epochs
                )
                val_metrics = self.validate()
                metrics.update(val_metrics)
                
                std_adv = metrics.get('std_advantage', 0.0)
                max_adv = metrics.get('max_advantage', 0.0)
                print(f"{epoch+1:6d} {metrics['grpo_loss']:12.5f} {metrics['val_loss']:12.5f} {metrics['ppo_loss']:12.5f} {metrics['kl']:10.5f} {std_adv:10.4f} {max_adv:10.4f} {self.model.kl_beta:8.4f}")
                self.save_checkpoint(epoch, "grpo", metrics)
            
            print("="*100)
            print("✓ GRPO 训练完成！")
            print("="*100)
        else:
            print("\n" + "="*100)
            print("跳过 GRPO 训练（grpo_epochs=0）")
            print("="*100)
            print("✓ 仅使用 RCE 训练完成！")
            print("="*100)
        
        print("\n" + "="*100)
        print("✓ GRPO Post-Training 完成！")
        print("="*100)


def main():
    parser = argparse.ArgumentParser(description="GRPO Post-Training")
    parser.add_argument("--sft_ckpt", type=str, default=None, help="SFT模型检查点路径（可选，不提供则从头训练）")
    parser.add_argument("--beam_data", type=str, required=True, help="束搜索数据JSON路径")
    parser.add_argument("--output_dir", type=str, default="results/grpo", help="输出目录")
    parser.add_argument("--img_emb", type=str, default=None, help="图像embedding缓存路径(.pth)")
    parser.add_argument("--text_emb", type=str, default=None, help="文本embedding缓存路径(.pth)")
    parser.add_argument("--rce_epochs", type=int, default=1, help="RCE预热epochs")
    parser.add_argument("--grpo_epochs", type=int, default=3, help="GRPO训练epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--rce_lr", type=float, default=1e-4, help="RCE学习率")
    parser.add_argument("--grpo_lr", type=float, default=1e-5, help="GRPO学习率")
    parser.add_argument("--kl_beta", type=float, default=0.1, help="KL散度权重（越大越保守）")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--use_top1_only", action="store_true", help="只使用Top-1 beam训练（回归V2监督学习方式）")
    # 新的 Reward 参数（推荐使用）
    parser.add_argument("--reward_mode", type=str, default="hard_plus_soft", 
                        choices=["hard_plus_soft", "hard_plus_soft_v2", "separated", "hard_only", "soft_only", "hybrid", "legacy"],
                        help="Reward模式：hard_plus_soft（默认，reward=vqa_correct+vqa_acc_score，范围[0,2]）、hard_plus_soft_v2、separated（推荐，正负样本有明确gap）、hard_only、soft_only、hybrid、legacy")
    parser.add_argument("--hard_weight", type=float, default=1.0, help="Hard correctness权重（默认1.0）")
    parser.add_argument("--soft_weight", type=float, default=1.0, help="Soft correctness权重（默认1.0）")
    # 兼容旧的 Reward 参数（legacy 模式使用）
    parser.add_argument("--reward_alpha", type=float, default=0.0, help="Quality权重（legacy模式，默认0.0）")
    parser.add_argument("--reward_beta", type=float, default=0.0, help="Correctness权重（legacy模式，默认0.0）")
    parser.add_argument("--reward_correctness_mode", type=str, default="01", choices=["01", "pm1"], help="Correctness模式（legacy模式）")
    parser.add_argument("--use_logprob", action="store_true", help="使用logprob_score而非beam_score（legacy模式）")
    parser.add_argument("--num_layers", type=int, default=1, help="Cross-Attention层数（默认1，与v2一致）")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载束搜索数据: {args.beam_data}")
    beam_data = load_beam_data(args.beam_data)
    
    # 检测数据格式：旧格式（id_list/score_list）还是新格式（pointer_candidates）
    first_key = list(beam_data.keys())[0]
    first_data = beam_data[first_key]
    
    if "pointer_candidates" in first_data:
        # 新格式：RL数据（包含correctness）
        data_format = "rl"
        print("✓ 检测到新格式数据（RL数据，包含correctness）")
        shot_num = len(first_data["pointer_candidates"][0]["pointer"])
        # 新格式没有固定的num_beams，每个query的候选数量可能不同
        num_beams = None
    elif "id_list" in first_data and "score_list" in first_data:
        # 旧格式：传统beam数据
        data_format = "legacy"
        print("✓ 检测到旧格式数据（传统beam数据）")
        num_beams = len(first_data["id_list"])
        shot_num = len(first_data["id_list"][0]) - 1
    else:
        raise ValueError(f"未知的数据格式！数据应包含 'pointer_candidates' 或 'id_list'/'score_list'")
    
    train_data, val_data = split_beam_data(beam_data, train_ratio=0.8)
    print(f"数据划分: 训练集 {len(train_data)}, 验证集 {len(val_data)}")
    print(f"Shot数量: {shot_num}")
    if num_beams is not None:
        print(f"Beam数量: {num_beams}")
    
    # 加载embedding（只需要img_emb即可）
    if args.img_emb:
        print(f"加载图像embedding: {args.img_emb}")
        img_emb_data = torch.load(args.img_emb, weights_only=False)
        
        # 转换为tensor（如果是numpy数组）
        if not isinstance(img_emb_data, torch.Tensor):
            img_emb_data = torch.from_numpy(img_emb_data).float()
        
        # 获取候选池索引（根据数据格式提取）
        all_icd_indices = set()
        if data_format == "rl":
            # 新格式：从pointer_candidates中提取
            for qid, data in beam_data.items():
                for candidate in data.get("pointer_candidates", []):
                    pointer = candidate.get("pointer", [])
                    for idx in pointer:
                        all_icd_indices.add(idx)
        else:
            # 旧格式：从id_list中提取
            for qid, data in beam_data.items():
                for beam in data["id_list"]:
                    for idx in beam[:-1]:  # 排除最后的query_id
                        all_icd_indices.add(idx)
        candidate_indices = sorted(list(all_icd_indices))
        K = len(candidate_indices)
        
        # query embedding使用全部数据
        query_embeddings = img_emb_data  # [N, d]
        d_model = query_embeddings.shape[1]
        
        # candidate embedding只提取候选池中的样本
        candidate_embeddings = img_emb_data[candidate_indices]  # [K, d]
        
        print(f"Embedding维度: {d_model}")
        print(f"候选池大小: {K}")
        print(f"候选池Embedding形状: {candidate_embeddings.shape}")
        use_real_embedding = True
    else:
        print("注意：未提供embedding路径，使用模拟embedding")
        num_samples = max(int(k) for k in beam_data.keys()) + 1
        d_model = 768
        K = 64
        query_embeddings = torch.randn(num_samples, d_model)
        candidate_embeddings = torch.randn(K, d_model)
        candidate_indices = list(range(K))
        use_real_embedding = False
    
    # 创建数据集（根据数据格式选择）
    class DummyDS:
        def __getitem__(self, idx):
            return {}
    
    if data_format == "rl":
        # 新格式：使用RLBeamDatasetWithEmbedding
        print("\n使用 RLBeamDatasetWithEmbedding（新格式）")
        if args.batch_size != 1:
            print(f"⚠️  警告：新格式数据要求 batch_size=1，但当前设置为 {args.batch_size}，将自动调整为1")
            args.batch_size = 1
        
        train_dataset = RLBeamDatasetWithEmbedding(
            rl_data=train_data,
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            candidate_indices=candidate_indices,
            shot_num=shot_num,
            normalize_rewards=True,
            # 新的 reward 参数
            reward_mode=args.reward_mode,
            hard_weight=args.hard_weight,
            soft_weight=args.soft_weight,
            # 兼容旧接口的参数
            reward_alpha=args.reward_alpha,
            reward_beta=args.reward_beta,
            reward_correctness_mode=args.reward_correctness_mode,
            use_logprob=args.use_logprob
        )
        val_dataset = RLBeamDatasetWithEmbedding(
            rl_data=val_data,
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            candidate_indices=candidate_indices,
            shot_num=shot_num,
            normalize_rewards=True,
            # 新的 reward 参数
            reward_mode=args.reward_mode,
            hard_weight=args.hard_weight,
            soft_weight=args.soft_weight,
            # 兼容旧接口的参数
            reward_alpha=args.reward_alpha,
            reward_beta=args.reward_beta,
            reward_correctness_mode=args.reward_correctness_mode,
            use_logprob=args.use_logprob
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=1,  # 必须为1
            shuffle=True, 
            collate_fn=collate_fn_rl_v3
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1,  # 必须为1
            shuffle=False, 
            collate_fn=collate_fn_rl_v3
        )
    else:
        # 旧格式：使用BeamDatasetWithEmbedding
        print("\n使用 BeamDatasetWithEmbedding（旧格式）")
        train_dataset = BeamDatasetWithEmbedding(
            train_data, DummyDS(), query_embeddings, 
            candidate_embeddings,
            candidate_indices, num_beams, shot_num
        )
        val_dataset = BeamDatasetWithEmbedding(
            val_data, DummyDS(), query_embeddings,
            candidate_embeddings,
            candidate_indices, num_beams, shot_num
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn_v3
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_v3
        )
    
    print(f"\n创建模型...")
    print(f"  - num_layers: {args.num_layers} (Cross-Attention层数)")
    model = PointerSelectorV3(
        d_model=d_model,
        K=K,
        shot_num=shot_num,
        kl_beta=args.kl_beta,
        num_layers=args.num_layers
    )
    
    # 如果提供了SFT checkpoint，加载权重
    if args.sft_ckpt:
        print(f"加载SFT检查点: {args.sft_ckpt}")
        ckpt = torch.load(args.sft_ckpt, map_location='cpu', weights_only=False)
        
        # 处理不同格式的checkpoint
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            # PyTorch Lightning格式，需要去掉前缀并映射参数名
            state_dict = {}
            for k, v in ckpt['state_dict'].items():
                # 去掉 "lever_lm." 前缀
                if k.startswith('lever_lm.'):
                    new_key = k[len('lever_lm.'):]
                else:
                    new_key = k
                
                # 去掉 "pointer_selector." 前缀
                if new_key.startswith('pointer_selector.'):
                    new_key = new_key[len('pointer_selector.'):]
                
                # 映射旧版v2参数名到新版v3参数名
                # 旧版v2: cross_attn.xxx -> 新版v3: cross_attn_layers.0.xxx
                if new_key.startswith('cross_attn.'):
                    new_key = new_key.replace('cross_attn.', 'cross_attn_layers.0.')
                # 旧版v2: attn_norm.xxx -> 新版v3: attn_norms.0.xxx
                elif new_key.startswith('attn_norm.'):
                    new_key = new_key.replace('attn_norm.', 'attn_norms.0.')
                
                state_dict[new_key] = v
        else:
            state_dict = ckpt
        
        # 只保留v3模型需要的参数（过滤掉sen_model等不需要的参数）
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # 检查加载情况
        loaded_keys = set(filtered_state_dict.keys())
        missing_keys = model_keys - loaded_keys
        
        print(f"  - 模型参数数量: {len(model_keys)}")
        print(f"  - 成功匹配参数: {len(loaded_keys)}")
        if missing_keys:
            print(f"  - 未匹配参数: {missing_keys}")
        
        # 加载权重
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✓ SFT权重加载完成")
    else:
        print("未提供SFT检查点，从头开始训练")
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = GRPOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rce_lr=args.rce_lr,
        grpo_lr=args.grpo_lr,
        rce_epochs=args.rce_epochs,
        grpo_epochs=args.grpo_epochs,
        use_top1_only=args.use_top1_only,
        save_dir=args.output_dir
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
