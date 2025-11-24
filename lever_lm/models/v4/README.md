# Pointer Selector V4: V3 + Offline RL（RCE + GRPO）

## 简介
- 在 V3（Bi-Encoder + 排序学习）的基础上，新增离线强化学习阶段：先 RCE 预热，再 GRPO（PPO-clip + KL）后训练。
- 目的：利用束搜索的多条 beam 及分数，进一步优化候选排序与端到端指标（VQA Acc）。

## 训练流程
1) 监督（已完成）：V1/V3 训练得到初始策略 π_old（推荐使用 V3: pairwise, ls=0.1, dp=0.2）。
2) 离线RL：
   - RCE 预热：CE 按 `softmax(score/τ)` 加权，稳定进入 RL。
   - GRPO：组相对优势 + PPO-clip + KL 正则（脱离在线环境，基于离线 beam）。

## 主要文件
- `models/v4/pointer_selector_v4.py`：
  - 与 V3 相同主干；
  - 新增 log-prob 计算、RCE/GRPO 辅助函数、V4 配置。
- `workflows/pointer_rl_train.py`：离线RL训练脚本（RCE + GRPO）。

## 运行
### 训练（离线RL）
```bash
./scripts/pointer_rl_train.sh
```
关键参数（见 `workflows/arguments.py`）：
- `--init_ckpt`: 初始化checkpoint（通常使用V3 last/best）
- `--rce_epochs`: RCE轮数（默认1）
- `--grpo_epochs`: GRPO轮数（默认2）
- `--clip_epsilon`: PPO裁剪阈值（默认0.2）
- `--kl_beta`: KL正则系数（默认0.01）
- `--reward_norm`: 奖励归一化方式（zscore/minmax）
- `--tau`: RCE温度

### 推理
```bash
./scripts/pointer_inference_v4.sh
```

## 注意事项
- 当前演示版 `pointer_rl_train.py` 对“池内位置映射”简化处理，建议按 SFT 训练时的严格对齐逻辑替换；
- 若 beam 的分数范围极小，需做分组归一化（z-score）并检验方差；
- 监控：Val CE@K / NDCG、KL、优势分布与稳定性。

## 期望收益
- 在 V3 的基础上进一步提升排序质量与 VQA 准确率（通常 0.2%~1.0% 额外增益，视数据质量与超参数而定）。
