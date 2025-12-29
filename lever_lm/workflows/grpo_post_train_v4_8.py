"""
GRPO Post-Training 训练脚本 - V4-8 版本（Slot/Set Decoder）

V4-8 方案：Slot/Set Decoder（并行 slots 协同）
- 同时维护 S 个"slot"，slots 之间 self-attn 协同分工
- 每个 slot 生成一行 logits，实现并行预测

作者: Lever-Plus Team
日期: 2025-12-28
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
from tqdm import tqdm

from lever_lm.models.v3 import (
    RLBeamDatasetWithEmbedding,
    collate_fn_rl_v3,
    load_beam_data,
    split_beam_data
)
from lever_lm.models.v3.pointer_selector_v4_8_rl import PointerSelectorV4_8_RL
from lever_lm.utils.reward_utils import (
    compute_temperature_schedule,
    adaptive_kl_beta
)


class GRPOTrainerV4_8:
    """GRPO Post-Training 训练器 - V4-8 版本"""
    
    def __init__(
        self,
        model: PointerSelectorV4_8_RL,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        rce_epochs: int = 15,
        rce_lr: float = 1e-4,
        grpo_epochs: int = 50,
        grpo_lr: float = 5e-6,
        kl_target_min: float = 0.01,
        kl_target_max: float = 0.1,
        gradient_clip: float = 1.0,
        save_dir: str = "checkpoints",
        rce_use_raw_reward: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rce_epochs = rce_epochs
        self.rce_lr = rce_lr
        self.grpo_epochs = grpo_epochs
        self.grpo_lr = grpo_lr
        self.kl_target_min = kl_target_min
        self.kl_target_max = kl_target_max
        self.gradient_clip = gradient_clip
        self.save_dir = save_dir
        self.rce_use_raw_reward = rce_use_raw_reward
        os.makedirs(save_dir, exist_ok=True)

    def compute_old_log_probs(self, sft_model: PointerSelectorV4_8_RL) -> Dict:
        """预计算SFT模型的log概率"""
        print("计算SFT模型的old_log_probs...")
        sft_model.eval()
        old_log_probs_dict = {}
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Computing old_log_probs"):
                query_emb = batch["query_emb"].to(self.device)
                cand_emb = batch["cand_emb"].to(self.device)
                beam_labels = batch["beam_labels"].to(self.device)
                query_ids = batch.get("query_ids", [batch.get("query_id")])
                if isinstance(query_ids, (int, str)):
                    query_ids = [query_ids]
                log_probs = sft_model.compute_log_probs_per_beam(query_emb, cand_emb, beam_labels)
                for i, qid in enumerate(query_ids):
                    old_log_probs_dict[qid] = log_probs[i].cpu()
        return old_log_probs_dict

    def train_rce_epoch(self, optimizer, epoch, total_epochs):
        """RCE预热阶段训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in self.train_loader:
            query_emb = batch["query_emb"].to(self.device)
            cand_emb = batch["cand_emb"].to(self.device)
            beam_labels = batch["beam_labels"].to(self.device)
            beam_rewards = batch["beam_rewards_raw" if self.rce_use_raw_reward else "beam_rewards"].to(self.device)
            optimizer.zero_grad()
            loss = self.model.compute_rce_loss(query_emb, cand_emb, beam_labels, beam_rewards, temperature=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return {"rce_loss": total_loss / num_batches}

    def train_grpo_epoch(self, optimizer, old_log_probs_dict, epoch):
        """GRPO训练阶段训练一个epoch"""
        self.model.train()
        metrics = {"grpo_loss": 0.0, "ppo_loss": 0.0, "kl_loss": 0.0, "kl": 0.0, "std_advantage": 0.0}
        num_batches = 0
        for batch in self.train_loader:
            query_emb = batch["query_emb"].to(self.device)
            cand_emb = batch["cand_emb"].to(self.device)
            beam_labels = batch["beam_labels"].to(self.device)
            beam_rewards = batch["beam_rewards"].to(self.device)
            query_ids = batch.get("query_ids", [batch.get("query_id")])
            if isinstance(query_ids, (int, str)):
                query_ids = [query_ids]
            old_log_probs = torch.stack([old_log_probs_dict[qid] for qid in query_ids]).to(self.device)
            optimizer.zero_grad()
            result = self.model.compute_grpo_loss(query_emb, cand_emb, beam_labels, beam_rewards, old_log_probs)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            optimizer.step()
            metrics["grpo_loss"] += result["loss"].item()
            metrics["ppo_loss"] += result["ppo_loss"].item()
            metrics["kl_loss"] += result["kl_loss"].item()
            metrics["kl"] += result["kl"].item()
            metrics["std_advantage"] += result["std_advantage"].item()
            num_batches += 1
            # 自适应KL
            self.model.kl_beta = adaptive_kl_beta(result["kl"].item(), self.model.kl_beta, self.kl_target_min, self.kl_target_max, 1.5)
        for k in metrics:
            metrics[k] /= num_batches
        return metrics

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            query_emb = batch["query_emb"].to(self.device)
            cand_emb = batch["cand_emb"].to(self.device)
            beam_labels = batch["beam_labels"].to(self.device)
            beam_rewards = batch["beam_rewards_raw"].to(self.device)
            loss = self.model.compute_rce_loss(query_emb, cand_emb, beam_labels, beam_rewards, temperature=1.0)
            total_loss += loss.item()
            num_batches += 1
        return {"val_loss": total_loss / num_batches}

    def save_checkpoint(self, epoch, phase, metrics):
        """保存检查点"""
        ckpt_path = os.path.join(self.save_dir, f"{phase}_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch, "phase": phase, "model_state_dict": self.model.state_dict(),
            "kl_beta": self.model.kl_beta, "metrics": metrics, "model_type": "v4_8",
            "num_slot_layers": self.model.num_slot_layers, "use_slot_cand_attn": self.model.use_slot_cand_attn
        }, ckpt_path)
        print(f"✓ 保存检查点: {ckpt_path}")

    def train(self, sft_checkpoint: Optional[str] = None):
        """完整训练流程"""
        print("="*70)
        print(f"GRPO Post-Training - V4-8 (Slot Decoder {self.model.num_slot_layers} layers)")
        print("="*70)
        
        # 加载SFT检查点
        if sft_checkpoint:
            print(f"加载SFT检查点: {sft_checkpoint}")
            ckpt = torch.load(sft_checkpoint, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            if isinstance(state_dict, dict) and "state_dict" not in state_dict and "model_state_dict" not in state_dict:
                pass
            else:
                state_dict = {k.replace("lever_lm.", "").replace("pointer_selector.", ""): v for k, v in state_dict.items()}
            
            model_keys = set(self.model.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in model_keys}
            missing, _ = self.model.load_state_dict(filtered, strict=False)
            if missing:
                print(f"[信息] V4-8 新增参数（随机初始化）: {len(missing)} 个")
            print("✓ SFT权重加载完成")

        # 创建SFT模型副本
        sft_model = PointerSelectorV4_8_RL(
            d_model=self.model.d_model, K=self.model.K, shot_num=self.model.shot_num,
            num_layers=self.model.num_layers, num_slot_layers=self.model.num_slot_layers,
            use_slot_cand_attn=self.model.use_slot_cand_attn, label_smoothing=0.0, dropout=0.5
        ).to(self.device)
        sft_model.load_state_dict(self.model.state_dict())
        sft_model.eval()
        old_log_probs_dict = self.compute_old_log_probs(sft_model)
        del sft_model

        # RCE预热
        print("\n" + "="*60)
        print("阶段1：RCE预热")
        print("="*60)
        rce_optimizer = AdamW(self.model.parameters(), lr=self.rce_lr)
        for epoch in range(self.rce_epochs):
            metrics = self.train_rce_epoch(rce_optimizer, epoch, self.rce_epochs)
            val_metrics = self.validate()
            metrics.update(val_metrics)
            print(f"RCE Epoch {epoch+1}: Train Loss = {metrics['rce_loss']:.5f}, Val Loss = {metrics['val_loss']:.5f}")
            self.save_checkpoint(epoch, "rce", metrics)

        # GRPO训练
        if self.grpo_epochs > 0:
            print("\n" + "="*80)
            print("阶段2：GRPO训练")
            print("="*80)
            grpo_optimizer = AdamW(self.model.parameters(), lr=self.grpo_lr)
            for epoch in range(self.grpo_epochs):
                metrics = self.train_grpo_epoch(grpo_optimizer, old_log_probs_dict, epoch)
                val_metrics = self.validate()
                metrics.update(val_metrics)
                print(f"GRPO Epoch {epoch+1}: Loss = {metrics['grpo_loss']:.5f}, Val = {metrics['val_loss']:.5f}, KL = {metrics['kl']:.5f}, Adv Std = {metrics['std_advantage']:.4f}")
                self.save_checkpoint(epoch, "grpo", metrics)
        
        print("="*70)
        print(f"✓ V4-8 训练完成！")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="GRPO Post-Training - V4-8")
    parser.add_argument("--sft_ckpt", type=str, default=None)
    parser.add_argument("--beam_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/grpo_v4_8")
    parser.add_argument("--img_emb", type=str, required=True)
    parser.add_argument("--rce_epochs", type=int, default=15)
    parser.add_argument("--grpo_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rce_lr", type=float, default=1e-4)
    parser.add_argument("--grpo_lr", type=float, default=5e-6)
    parser.add_argument("--kl_beta", type=float, default=0.1)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_slot_layers", type=int, default=2)
    parser.add_argument("--use_slot_cand_attn", action="store_true", default=True)
    parser.add_argument("--no_slot_cand_attn", dest="use_slot_cand_attn", action="store_false")
    parser.add_argument("--rce_use_raw_reward", action="store_true")
    parser.add_argument("--reward_mode", type=str, default="hard_plus_soft")
    parser.add_argument("--hard_weight", type=float, default=1.0)
    parser.add_argument("--soft_weight", type=float, default=1.0)
    parser.add_argument("--normalize_method", type=str, default="z_score_clamp")
    parser.add_argument("--normalize_clamp_value", type=float, default=3.0)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载束搜索数据: {args.beam_data}")
    beam_data = load_beam_data(args.beam_data)
    
    data_keys = [k for k in beam_data.keys() if k != "_meta" and (str(k).isdigit() or str(k).replace('-', '').replace('_', '').isdigit())]
    first_key = data_keys[0]
    first_data = beam_data[first_key]
    pointer_candidates = first_data.get("pointer_candidates", [])
    sample_pointer = pointer_candidates[0].get("pointer") or pointer_candidates[0].get("pointer_pos", [])
    shot_num = len(sample_pointer)
    
    train_data, val_data = split_beam_data(beam_data, train_ratio=0.8)
    print(f"数据划分: 训练集 {len(train_data)}, 验证集 {len(val_data)}, shot_num={shot_num}")

    # 加载embedding
    print(f"加载图像embedding: {args.img_emb}")
    img_emb_data = torch.load(args.img_emb, weights_only=False)
    if not isinstance(img_emb_data, torch.Tensor):
        img_emb_data = torch.from_numpy(img_emb_data).float()
    query_embeddings = img_emb_data
    candidate_embeddings = img_emb_data
    d_model = query_embeddings.shape[1]
    
    all_icd_indices = set()
    for qid, data in beam_data.items():
        if qid == "_meta" or not (str(qid).isdigit() or str(qid).replace('-', '').replace('_', '').isdigit()):
            continue
        for candidate in data.get("pointer_candidates", []):
            pointer = candidate.get("pointer_pos") or candidate.get("pointer", [])
            for idx in pointer:
                all_icd_indices.add(idx)
    candidate_indices = sorted(list(all_icd_indices))
    K = 64
    print(f"Embedding维度: {d_model}, K={K}")
    
    # 创建数据集
    train_dataset = RLBeamDatasetWithEmbedding(
        rl_data=train_data, query_embeddings=query_embeddings, candidate_embeddings=candidate_embeddings,
        candidate_indices=candidate_indices, shot_num=shot_num, normalize_rewards=True,
        normalize_method=args.normalize_method, normalize_clamp_value=args.normalize_clamp_value,
        reward_mode=args.reward_mode, hard_weight=args.hard_weight, soft_weight=args.soft_weight
    )
    val_dataset = RLBeamDatasetWithEmbedding(
        rl_data=val_data, query_embeddings=query_embeddings, candidate_embeddings=candidate_embeddings,
        candidate_indices=candidate_indices, shot_num=shot_num, normalize_rewards=True,
        normalize_method=args.normalize_method, normalize_clamp_value=args.normalize_clamp_value,
        reward_mode=args.reward_mode, hard_weight=args.hard_weight, soft_weight=args.soft_weight
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_rl_v3)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_rl_v3)
    
    # 创建模型
    print(f"\n创建 V4-8 模型...")
    print(f"  - num_layers: {args.num_layers}")
    print(f"  - num_slot_layers: {args.num_slot_layers}")
    print(f"  - use_slot_cand_attn: {args.use_slot_cand_attn}")
    
    model = PointerSelectorV4_8_RL(
        d_model=d_model, K=K, shot_num=shot_num, kl_beta=args.kl_beta, clip_epsilon=args.clip_epsilon,
        num_layers=args.num_layers, num_slot_layers=args.num_slot_layers, use_slot_cand_attn=args.use_slot_cand_attn,
        label_smoothing=0.0, dropout=0.5
    )
    
    # 创建训练器
    trainer = GRPOTrainerV4_8(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        rce_epochs=args.rce_epochs, rce_lr=args.rce_lr, grpo_epochs=args.grpo_epochs, grpo_lr=args.grpo_lr,
        save_dir=args.output_dir, rce_use_raw_reward=args.rce_use_raw_reward
    )
    
    # 开始训练
    trainer.train(sft_checkpoint=args.sft_ckpt)


if __name__ == "__main__":
    main()
