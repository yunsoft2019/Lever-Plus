"""
GRPO Post-Training 集成测试

测试内容：
- 步骤7：RCE集成测试（完整的RCE训练循环）
- 步骤8：GRPO集成测试（完整的GRPO训练循环）

作者: Lever-Plus Team
日期: 2025-12-02
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from lever_lm.models.v3 import (
    PointerSelectorV3,
    BeamDatasetWithEmbedding,
    collate_fn_v3
)
from lever_lm.utils import (
    compute_temperature_schedule,
    adaptive_kl_beta
)


def create_mock_data(num_samples=50, num_beams=5, shot_num=2, d_model=768, K=32):
    """创建模拟数据"""
    beam_data = {}
    for i in range(num_samples):
        query_id = str(100 + i)
        # 创建beam_labels，确保索引在[0, K)范围内
        beam_data[query_id] = {
            'id_list': [[j % K, (j+1) % K, 100+i] for j in range(0, num_beams*2, 2)],
            'score_list': [0.05 - j*0.008 for j in range(num_beams)]
        }
    
    class MockDS:
        def __getitem__(self, idx):
            return {'image': f'img_{idx}', 'question': f'q_{idx}'}
    
    max_query_id = max(int(k) for k in beam_data.keys()) + 1
    query_emb = torch.randn(max_query_id, d_model)
    cand_emb = torch.randn(K, d_model)
    cand_indices = list(range(K))
    
    ds = BeamDatasetWithEmbedding(
        beam_data=beam_data,
        index_ds=MockDS(),
        query_embeddings=query_emb,
        candidate_embeddings=cand_emb,
        candidate_indices=cand_indices,
        beam_size=num_beams,
        shot_num=shot_num
    )
    
    return ds


def test_rce_integration():
    """
    步骤7：RCE集成测试
    
    测试完整的RCE训练循环：
    1. 数据加载
    2. 模型前向传播
    3. RCE损失计算
    4. 反向传播
    5. 参数更新
    6. 温度调度
    """
    print("="*70)
    print("步骤7：RCE集成测试")
    print("="*70)
    
    # 配置
    d_model = 768
    K = 32
    shot_num = 2
    batch_size = 8
    num_epochs = 2
    lr = 1e-4
    temp_start = 2.0
    temp_end = 0.5
    
    # 创建数据
    ds = create_mock_data(num_samples=32, d_model=d_model, K=K, shot_num=shot_num)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_v3)
    print(f"✓ 数据加载: {len(ds)}样本, {len(loader)}批次")
    
    # 创建模型
    model = PointerSelectorV3(d_model=d_model, K=K, shot_num=shot_num)
    optimizer = AdamW(model.parameters(), lr=lr)
    print(f"✓ 模型创建: {sum(p.numel() for p in model.parameters()):,}参数")
    
    # 训练循环
    total_steps = num_epochs * len(loader)
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in loader:
            # 计算温度
            temperature = compute_temperature_schedule(
                global_step, total_steps, temp_start, temp_end
            )
            
            query_emb = batch["query_emb"]
            cand_emb = batch["cand_emb"]
            beam_labels = batch["beam_labels"]
            beam_rewards_raw = batch["beam_rewards_raw"]
            
            optimizer.zero_grad()
            
            # RCE损失
            loss = model.compute_rce_loss(
                query_emb, cand_emb, beam_labels, beam_rewards_raw,
                temperature=temperature
            )
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
        
        avg_loss = epoch_loss / len(loader)
        final_temp = compute_temperature_schedule(global_step-1, total_steps, temp_start, temp_end)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, τ={final_temp:.3f}")
    
    print("✓ RCE集成测试通过！")
    return True


def test_grpo_integration():
    """
    步骤8：GRPO集成测试
    
    测试完整的GRPO训练循环：
    1. 计算old_log_probs（模拟SFT模型）
    2. GRPO损失计算（PPO + KL）
    3. 课程学习（top-k beam）
    4. KL自适应调整
    5. 反向传播和参数更新
    """
    print("\n" + "="*70)
    print("步骤8：GRPO集成测试")
    print("="*70)
    
    # 配置
    d_model = 768
    K = 32
    shot_num = 2
    batch_size = 8
    num_epochs = 3
    lr = 5e-5
    early_epochs = 1
    early_top_k = 3
    late_top_k = 5
    
    # 创建数据
    ds = create_mock_data(num_samples=32, d_model=d_model, K=K, shot_num=shot_num)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_v3)
    print(f"✓ 数据加载: {len(ds)}样本, {len(loader)}批次")
    
    # 创建模型
    model = PointerSelectorV3(d_model=d_model, K=K, shot_num=shot_num)
    optimizer = AdamW(model.parameters(), lr=lr)
    print(f"✓ 模型创建: {sum(p.numel() for p in model.parameters()):,}参数")
    
    # 预计算old_log_probs
    print("  计算old_log_probs...")
    old_log_probs_dict = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            query_emb = batch["query_emb"]
            cand_emb = batch["cand_emb"]
            beam_labels = batch["beam_labels"]
            query_ids = batch["query_ids"]
            
            log_probs = model.compute_log_probs_per_beam(query_emb, cand_emb, beam_labels)
            for i, qid in enumerate(query_ids):
                old_log_probs_dict[qid] = log_probs[i]
    print(f"✓ old_log_probs计算完成: {len(old_log_probs_dict)}样本")
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        
        # 课程学习
        use_top_k = early_top_k if epoch < early_epochs else late_top_k
        
        epoch_metrics = {
            "loss": 0.0,
            "ppo_loss": 0.0,
            "kl": 0.0
        }
        
        for batch in loader:
            query_emb = batch["query_emb"]
            cand_emb = batch["cand_emb"]
            beam_labels = batch["beam_labels"]
            beam_rewards = batch["beam_rewards"]
            query_ids = batch["query_ids"]
            
            # 获取old_log_probs
            old_log_probs = torch.stack([old_log_probs_dict[qid] for qid in query_ids])
            
            optimizer.zero_grad()
            
            # GRPO损失
            result = model.compute_grpo_loss(
                query_emb, cand_emb, beam_labels, beam_rewards,
                old_log_probs, use_top_k=use_top_k
            )
            
            loss = result["loss"]
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # KL自适应
            current_kl = result["kl"].item()
            model.kl_beta = adaptive_kl_beta(current_kl, model.kl_beta)
            
            # 累积指标
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["ppo_loss"] += result["ppo_loss"].item()
            epoch_metrics["kl"] += current_kl
        
        # 平均
        for k in epoch_metrics:
            epoch_metrics[k] /= len(loader)
        
        print(f"  Epoch {epoch+1}: loss={epoch_metrics['loss']:.4f}, "
              f"ppo={epoch_metrics['ppo_loss']:.4f}, kl={epoch_metrics['kl']:.4f}, "
              f"β={model.kl_beta:.4f}, top_k={use_top_k}")
    
    print("✓ GRPO集成测试通过！")
    return True


def test_full_pipeline():
    """
    步骤9：端到端测试
    
    测试完整的训练流程：RCE预热 -> GRPO训练
    """
    print("\n" + "="*70)
    print("步骤9：端到端测试")
    print("="*70)
    
    # 配置
    d_model = 768
    K = 32
    shot_num = 2
    batch_size = 8
    
    # 创建数据
    ds = create_mock_data(num_samples=32, d_model=d_model, K=K, shot_num=shot_num)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_v3)
    
    # 创建模型
    model = PointerSelectorV3(d_model=d_model, K=K, shot_num=shot_num)
    
    # ===== RCE预热 =====
    print("\n阶段1: RCE预热")
    rce_optimizer = AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for batch in loader:
        rce_optimizer.zero_grad()
        loss = model.compute_rce_loss(
            batch["query_emb"], batch["cand_emb"],
            batch["beam_labels"], batch["beam_rewards_raw"],
            temperature=1.5
        )
        loss.backward()
        rce_optimizer.step()
    print(f"  ✓ RCE预热完成, 最终loss: {loss.item():.4f}")
    
    # ===== 计算old_log_probs =====
    print("\n阶段2: 计算old_log_probs")
    old_log_probs_dict = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            log_probs = model.compute_log_probs_per_beam(
                batch["query_emb"], batch["cand_emb"], batch["beam_labels"]
            )
            for i, qid in enumerate(batch["query_ids"]):
                old_log_probs_dict[qid] = log_probs[i]
    print(f"  ✓ old_log_probs计算完成")
    
    # ===== GRPO训练 =====
    print("\n阶段3: GRPO训练")
    grpo_optimizer = AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for batch in loader:
        old_log_probs = torch.stack([old_log_probs_dict[qid] for qid in batch["query_ids"]])
        
        grpo_optimizer.zero_grad()
        result = model.compute_grpo_loss(
            batch["query_emb"], batch["cand_emb"],
            batch["beam_labels"], batch["beam_rewards"],
            old_log_probs, use_top_k=3
        )
        result["loss"].backward()
        grpo_optimizer.step()
        
        # KL自适应
        model.kl_beta = adaptive_kl_beta(result["kl"].item(), model.kl_beta)
    
    print(f"  ✓ GRPO训练完成, 最终loss: {result['loss'].item():.4f}, kl: {result['kl'].item():.4f}")
    
    print("\n" + "="*70)
    print("✓ 端到端测试通过！")
    print("="*70)
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRPO Post-Training 集成测试")
    print("="*70)
    
    # 运行测试
    rce_ok = test_rce_integration()
    grpo_ok = test_grpo_integration()
    e2e_ok = test_full_pipeline()
    
    # 汇总
    print("\n" + "="*70)
    print("测试汇总")
    print("="*70)
    print(f"  步骤7 RCE集成测试: {'✓ 通过' if rce_ok else '✗ 失败'}")
    print(f"  步骤8 GRPO集成测试: {'✓ 通过' if grpo_ok else '✗ 失败'}")
    print(f"  步骤9 端到端测试: {'✓ 通过' if e2e_ok else '✗ 失败'}")
    
    if rce_ok and grpo_ok and e2e_ok:
        print("\n✓ 所有集成测试通过！")
    else:
        print("\n✗ 存在失败的测试")
