"""
GRPO Post-Training 训练脚本 - V4-4 版本（Candidate Set Encoder + GRU + MMR）

基于 grpo_post_train_v4_3.py，使用 V4-4 模型

作者: Lever-Plus Team
日期: 2025-12-26
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
    BeamDataset,
    BeamDatasetWithEmbedding,
    RLBeamDatasetWithEmbedding,
    collate_fn_v3,
    collate_fn_rl_v3,
    load_beam_data,
    split_beam_data
)
from lever_lm.models.v3.pointer_selector_v4_4_rl import PointerSelectorV4_4_RL
from lever_lm.utils.reward_utils import (
    compute_temperature_schedule,
    adaptive_kl_beta
)


class GRPOTrainerV4_4:
    """GRPO Post-Training 训练器 - V4-4 版本"""
    
    def __init__(
        self,
        model: PointerSelectorV4_4_RL,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        # RCE配置
        rce_epochs: int = 1,
        rce_lr: float = 1e-5,
        rce_temp_start: float = 2.0,
        rce_temp_end: float = 0.5,
        use_top1_only: bool = False,
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
        self.rce_use_raw_reward = False
        
        # GRPO配置
        self.grpo_epochs = grpo_epochs
        self.grpo_lr = grpo_lr
        self.freeze_backbone_in_grpo = False
        self.grpo_early_epochs = grpo_early_epochs
        self.grpo_early_top_k = grpo_early_top_k
        self.grpo_late_top_k = grpo_late_top_k
        
        # KL配置
        self.kl_target_min = kl_target_min
        self.kl_target_max = kl_target_max
        self.kl_adjustment_factor = kl_adjustment_factor
        self.disable_adaptive_kl = False
        
        # 训练配置
        self.gradient_clip = gradient_clip
        self.warmup_ratio = warmup_ratio
        self.log_every = log_every
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)

    def compute_old_log_probs(self, sft_model: PointerSelectorV4_4_RL) -> Dict[int, torch.Tensor]:
        """预计算SFT模型的log概率"""
        print("计算SFT模型的old_log_probs...")
        sft_model.eval()
        old_log_probs_dict = {}
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Computing old_log_probs"):
                query_emb = batch["query_emb"].to(self.device)
                cand_emb = batch["cand_emb"].to(self.device)
                beam_labels = batch["beam_labels"].to(self.device)
                
                if isinstance(batch["query_id"], (int, str)):
                    query_ids = [batch["query_id"]]
                else:
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
        """RCE预热阶段训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        total_steps = total_epochs * len(self.train_loader)
        current_step = epoch * len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            step = current_step + batch_idx
            temperature = compute_temperature_schedule(
                step, total_steps,
                self.rce_temp_start, self.rce_temp_end
            )
            
            query_emb = batch["query_emb"].to(self.device)
            cand_emb = batch["cand_emb"].to(self.device)
            beam_labels = batch["beam_labels"].to(self.device)
            beam_rewards_raw = batch["beam_rewards_raw"].to(self.device)
            beam_rewards = batch.get("beam_rewards", beam_rewards_raw).to(self.device)
            
            if self.rce_use_raw_reward:
                reward_for_rce = beam_rewards_raw
            else:
                reward_for_rce = beam_rewards
            
            optimizer.zero_grad()
            
            loss = self.model.compute_rce_loss(
                query_emb, cand_emb, beam_labels, reward_for_rce,
                temperature=temperature,
                use_top1_only=self.use_top1_only
            )
            
            loss.backward()
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
        """GRPO训练阶段训练一个epoch"""
        self.model.train()
        
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
            
            if isinstance(batch.get("query_id"), (int, str)):
                query_ids = [batch["query_id"]]
            else:
                query_ids = batch["query_ids"]
            
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
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            optimizer.step()
            
            for k in metrics:
                if k == "grpo_loss":
                    metrics[k] += loss.item()
                elif k in result:
                    metrics[k] += result[k].item()
            num_batches += 1
            
            if not self.disable_adaptive_kl:
                current_kl = result["kl"].item()
                self.model.kl_beta = adaptive_kl_beta(
                    current_kl, self.model.kl_beta,
                    self.kl_target_min, self.kl_target_max,
                    self.kl_adjustment_factor
                )
        
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
        
        div_lambda_values = self.model.get_div_lambda_values()
        
        torch.save({
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "kl_beta": self.model.kl_beta,
            "metrics": metrics,
            "model_type": "v4_4",
            "div_lambda_values": div_lambda_values
        }, ckpt_path)
        if verbose:
            print(f"✓ 保存检查点: {ckpt_path}")
            print(f"  div_lambda: {div_lambda_values}")


    def train(self, sft_checkpoint: Optional[str] = None):
        """完整训练流程"""
        print("="*70)
        print("GRPO Post-Training - V4-4 (Candidate Set Encoder + GRU + MMR)")
        print("="*70)
        
        # 加载SFT检查点
        if sft_checkpoint:
            print(f"加载SFT检查点: {sft_checkpoint}")
            ckpt = torch.load(sft_checkpoint, map_location=self.device, weights_only=False)
            
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            else:
                state_dict = ckpt
            
            model_keys = set(self.model.state_dict().keys())
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                if k.startswith('lever_lm.'):
                    k = k[len('lever_lm.'):]
                if k.startswith('pointer_selector.'):
                    k = k[len('pointer_selector.'):]
                
                if k.startswith('cross_attn.'):
                    k = k.replace('cross_attn.', 'cross_attn_layers.0.')
                elif k.startswith('attn_norm.'):
                    k = k.replace('attn_norm.', 'attn_norms.0.')
                
                # 跳过 V2 特有的 query_update_gate
                if 'query_update_gate' in k:
                    continue
                
                if k in model_keys:
                    filtered_state_dict[k] = v
            
            missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing:
                print(f"[信息] V4-4 新增参数（随机初始化）: {len(missing)} 个")
                for k in list(missing)[:5]:
                    print(f"  - {k}")
                if len(missing) > 5:
                    print(f"  ... 还有 {len(missing) - 5} 个")
            
            print("✓ SFT权重加载完成（V4-4 特有参数已随机初始化）")
        
        # 创建SFT模型副本
        sft_model = PointerSelectorV4_4_RL(
            d_model=self.model.d_model,
            K=self.model.K,
            shot_num=self.model.shot_num,
            num_layers=self.model.num_layers,
            use_step_emb=self.model.use_step_emb,
            use_gru=self.model.use_gru,
            mmr_reduction=self.model.mmr_reduction,
            cand_encoder_layers=self.model.cand_encoder_layers,
            cand_encoder_heads=self.model.cand_encoder_heads,
            label_smoothing=0.0,
            dropout=0.5
        ).to(self.device)
        sft_model.load_state_dict(self.model.state_dict())
        sft_model.eval()
        
        old_log_probs_dict = self.compute_old_log_probs(sft_model)
        del sft_model
        
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
            
            total_steps = self.rce_epochs * len(self.train_loader)
            current_step = (epoch + 1) * len(self.train_loader)
            temperature = compute_temperature_schedule(
                current_step, total_steps, self.rce_temp_start, self.rce_temp_end
            )
            
            print(f"{epoch+1:6d} {metrics['rce_loss']:12.5f} {metrics['val_loss']:12.5f} {temperature:12.3f}")
            self.save_checkpoint(epoch, "rce", metrics)
        
        print("="*80)

        # ========== 阶段3：GRPO训练 ==========
        if self.grpo_epochs <= 0:
            print("\n" + "="*100)
            print("⚠️  GRPO epochs == 0，仅进行 RCE 预热")
            print("="*100)
            return
        
        print("\n" + "="*130)
        print("阶段3：GRPO训练")
        print("="*130)
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'PPO Loss':>12} {'KL':>10} {'Adv Std':>10} {'Adv Max':>10} {'Beta':>8} {'div_λ':>20}")
        print("-"*130)
        
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
            div_lambda = self.model.get_div_lambda_values()
            div_lambda_str = str([f"{v:.3f}" for v in div_lambda])
            
            print(f"{epoch+1:6d} {metrics['grpo_loss']:12.5f} {metrics['val_loss']:12.5f} {metrics['ppo_loss']:12.5f} {metrics['kl']:10.5f} {std_adv:10.4f} {max_adv:10.4f} {self.model.kl_beta:8.4f} {div_lambda_str:>20}")
            self.save_checkpoint(epoch, "grpo", metrics)
        
        print("="*130)
        print("✓ GRPO Post-Training (V4-4) 完成！")
        print(f"  最终 div_lambda: {self.model.get_div_lambda_values()}")
        print("="*130)



def main():
    parser = argparse.ArgumentParser(description="GRPO Post-Training - V4-4")
    parser.add_argument("--sft_ckpt", type=str, default=None, help="SFT模型检查点路径")
    parser.add_argument("--beam_data", type=str, required=True, help="束搜索数据JSON路径")
    parser.add_argument("--output_dir", type=str, default="results/grpo_v4_4", help="输出目录")
    parser.add_argument("--img_emb", type=str, default=None, help="图像embedding缓存路径(.pth)")
    parser.add_argument("--rce_epochs", type=int, default=5, help="RCE预热epochs")
    parser.add_argument("--grpo_epochs", type=int, default=50, help="GRPO训练epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--rce_lr", type=float, default=1e-4, help="RCE学习率")
    parser.add_argument("--grpo_lr", type=float, default=5e-6, help="GRPO学习率")
    parser.add_argument("--kl_beta", type=float, default=0.1, help="KL散度权重")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO Clip参数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--num_layers", type=int, default=1, help="Cross-Attention层数")
    # V4-4 特有参数
    parser.add_argument("--use_step_emb", action="store_true", default=True, help="使用step embedding")
    parser.add_argument("--no_step_emb", dest="use_step_emb", action="store_false")
    parser.add_argument("--use_gru", action="store_true", default=True, help="使用GRU decoder")
    parser.add_argument("--no_gru", dest="use_gru", action="store_false")
    parser.add_argument("--mmr_reduction", type=str, default="max", choices=["max", "mean"], help="MMR冗余计算方式")
    parser.add_argument("--cand_encoder_layers", type=int, default=1, help="Candidate Self-Attention层数")
    parser.add_argument("--cand_encoder_heads", type=int, default=1, help="Candidate Self-Attention头数")
    # 其他参数
    parser.add_argument("--rce_use_raw_reward", action="store_true", help="RCE使用原始reward")
    parser.add_argument("--disable_adaptive_kl", action="store_true", help="禁用自适应KL")
    parser.add_argument("--reward_mode", type=str, default="hard_plus_soft", help="Reward模式")
    parser.add_argument("--hard_weight", type=float, default=1.0, help="Hard权重")
    parser.add_argument("--soft_weight", type=float, default=1.0, help="Soft权重")
    parser.add_argument("--normalize_method", type=str, default="z_score_clamp", help="归一化方法")
    parser.add_argument("--normalize_clamp_value", type=float, default=3.0, help="Clamp值")
    # 兼容旧参数
    parser.add_argument("--reward_alpha", type=float, default=0.0)
    parser.add_argument("--reward_beta", type=float, default=0.0)
    parser.add_argument("--reward_correctness_mode", type=str, default="01")
    parser.add_argument("--use_logprob", action="store_true")
    parser.add_argument("--no_skip_fallback_reward", dest="skip_fallback_reward", action="store_false")
    parser.add_argument("--require_positive_query", action="store_true")
    parser.set_defaults(skip_fallback_reward=True)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载束搜索数据: {args.beam_data}")
    beam_data = load_beam_data(args.beam_data)
    
    data_keys = [k for k in beam_data.keys() if k != "_meta" and (k.isdigit() or (isinstance(k, str) and k.replace('-', '').replace('_', '').isdigit()))]
    if not data_keys:
        raise ValueError("数据中没有找到有效的query数据")
    
    first_key = data_keys[0]
    first_data = beam_data[first_key]
    
    if "query" in first_data:
        pointer_candidates = first_data.get("pointer_candidates", [])
    else:
        pointer_candidates = first_data.get("pointer_candidates", [])
    
    if pointer_candidates:
        print("✓ 检测到新格式数据（RL数据）")
        sample_pointer = pointer_candidates[0].get("pointer") or pointer_candidates[0].get("pointer_pos", [])
        shot_num = len(sample_pointer)
    else:
        raise ValueError("V4-4 训练需要 RL 格式数据")
    
    train_data, val_data = split_beam_data(beam_data, train_ratio=0.8)
    print(f"数据划分: 训练集 {len(train_data)}, 验证集 {len(val_data)}")
    print(f"Shot数量: {shot_num}")

    # 加载embedding
    if args.img_emb:
        print(f"加载图像embedding: {args.img_emb}")
        img_emb_data = torch.load(args.img_emb, weights_only=False)
        
        if not isinstance(img_emb_data, torch.Tensor):
            img_emb_data = torch.from_numpy(img_emb_data).float()
        
        query_embeddings = img_emb_data
        candidate_embeddings = img_emb_data
        d_model = query_embeddings.shape[1]
        
        all_icd_indices = set()
        for qid, data in beam_data.items():
            if qid == "_meta" or not (qid.isdigit() or (isinstance(qid, str) and qid.replace('-', '').replace('_', '').isdigit())):
                continue
            if "query" in data:
                pointer_candidates = data.get("pointer_candidates", [])
            else:
                pointer_candidates = data.get("pointer_candidates", [])
            for candidate in pointer_candidates:
                pointer = candidate.get("pointer_pos") or candidate.get("pointer", [])
                for idx in pointer:
                    all_icd_indices.add(idx)
        candidate_indices = sorted(list(all_icd_indices))
        K = len(candidate_indices)
        
        print(f"Embedding维度: {d_model}")
        print(f"【Per-Query候选池】每个query将使用自己独立的候选池")
    else:
        raise ValueError("V4-4 训练需要提供 --img_emb 参数")
    
    # 创建数据集
    print("\n使用 RLBeamDatasetWithEmbedding")
    if args.batch_size != 1:
        print(f"⚠️  警告：batch_size 将自动调整为1")
        args.batch_size = 1
    
    train_dataset = RLBeamDatasetWithEmbedding(
        rl_data=train_data,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        candidate_indices=candidate_indices,
        shot_num=shot_num,
        normalize_rewards=True,
        normalize_method=args.normalize_method,
        normalize_clamp_value=args.normalize_clamp_value,
        reward_mode=args.reward_mode,
        hard_weight=args.hard_weight,
        soft_weight=args.soft_weight,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_correctness_mode=args.reward_correctness_mode,
        use_logprob=args.use_logprob,
        skip_fallback_reward=args.skip_fallback_reward,
        require_positive_query=args.require_positive_query
    )
    val_dataset = RLBeamDatasetWithEmbedding(
        rl_data=val_data,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        candidate_indices=candidate_indices,
        shot_num=shot_num,
        normalize_rewards=True,
        normalize_method=args.normalize_method,
        normalize_clamp_value=args.normalize_clamp_value,
        reward_mode=args.reward_mode,
        hard_weight=args.hard_weight,
        soft_weight=args.soft_weight,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_correctness_mode=args.reward_correctness_mode,
        use_logprob=args.use_logprob,
        skip_fallback_reward=args.skip_fallback_reward,
        require_positive_query=args.require_positive_query
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_rl_v3)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_rl_v3)
    
    # 创建 V4-4 模型
    print(f"\n创建 V4-4 模型...")
    print(f"  - num_layers (Cross-Attn): {args.num_layers}")
    print(f"  - cand_encoder_layers: {args.cand_encoder_layers}")
    print(f"  - cand_encoder_heads: {args.cand_encoder_heads}")
    print(f"  - use_step_emb: {args.use_step_emb}")
    print(f"  - use_gru: {args.use_gru}")
    print(f"  - mmr_reduction: {args.mmr_reduction}")
    print(f"  - 架构: Candidate Self-Attn + Cross-Attn + GRU + MMR (V4-4)")
    
    model_K = 64
    print(f"  - 【Per-Query候选池】模型K值设置为{model_K}")
    
    model = PointerSelectorV4_4_RL(
        d_model=d_model,
        K=model_K,
        shot_num=shot_num,
        kl_beta=args.kl_beta,
        clip_epsilon=args.clip_epsilon,
        num_layers=args.num_layers,
        use_step_emb=args.use_step_emb,
        use_gru=args.use_gru,
        mmr_reduction=args.mmr_reduction,
        cand_encoder_layers=args.cand_encoder_layers,
        cand_encoder_heads=args.cand_encoder_heads,
        label_smoothing=0.0,
        dropout=0.5
    )
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"初始 div_lambda: {model.get_div_lambda_values()}")
    
    # 创建训练器
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = GRPOTrainerV4_4(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rce_lr=args.rce_lr,
        grpo_lr=args.grpo_lr,
        rce_epochs=args.rce_epochs,
        grpo_epochs=args.grpo_epochs,
        save_dir=args.output_dir
    )
    
    trainer.rce_use_raw_reward = args.rce_use_raw_reward
    trainer.disable_adaptive_kl = args.disable_adaptive_kl
    
    if args.rce_use_raw_reward:
        print("✓ RCE 训练将使用原始 reward")
    else:
        print("✓ RCE 训练将使用归一化后的 reward")
    
    # 开始训练
    trainer.train(sft_checkpoint=args.sft_ckpt)


if __name__ == "__main__":
    main()
