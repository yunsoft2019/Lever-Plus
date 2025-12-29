# Lever-Plus PointerSelector 升级方案清单（保持 RCE / GRPO 不变）

> 基于你提供的项目包（`Lever-Plus-main`）分析：  
> - 当前线上使用：`lever_lm/models/v3/pointer_selector_v3.py`（RL：RCE + GRPO）  
> - 核心选择器来自：`lever_lm/models/v2/pointer_selector_v2.py`（Bi-Encoder + 多层 Cross-Attn + 静态余弦打分）  
> - 你报告（`2025-12-25测试报告.md`）里也提到：GRPO “策略几乎没有变化”、shot≥3 不稳定/下降，这和 V2 的“静态排序 top-k”机制高度一致。

本文把“只改模型（forward/结构），**RCE/GRPO 代码保持不变**”作为硬约束，并把所有可行升级方案按**成功概率（预计带来稳定提升的可能性）从高到低**排序，方便你逐步探索。

---

## 0. 硬约束（确保不改 RCE / GRPO）

你的 `PointerSelectorV3` 的 RCE/GRPO 逻辑依赖这些事实（必须保持）：

1. **forward 接口不变**  
   `forward(query_emb, cand_emb, labels=None, return_loss=True) -> dict`  
   返回至少包含：  
   - `logits: [B, S, K_actual]`
   - `predictions: [B, S]`
   - 可选 `loss`

2. **logits 的最后一维必须等于 `cand_emb.shape[1]`**  
   因为 `compute_rce_loss()` 里用 `actual_K = cand_emb.shape[1]` 去 reshape logits：  
   ```py
   logits_for_loss = logits.reshape(-1, actual_K)
   ```
   所以如果你在模型内部偷偷把 K 变成 K+1，会直接炸（除非你同时把 cand_emb 也扩成 K+1）。

3. **Teacher Forcing 的语义不变**  
   - 训练/计算 logprob 时：每一步用 `labels[:, step]` 更新 mask（以及你新引入的“状态”）
   - 推理时：用 `argmax` 的 pred 更新 mask（以及状态）

4. **支持 per-query 动态 K**  
   你项目里已经修复了 per-query 候选池（V2/V3 多处用 `actual_K = cand_emb.shape[1]`），升级方案必须继续遵守。

---

## 1. 现状复盘：V2 其实不是“真正的 pointer net”（为什么 shot≥3 容易掉）

你现在的 `PointerSelectorV2.forward()` 的核心循环是：

- 每个 step 都用**同一个** `query_proj` 计算 `scores = query_proj @ cand_proj^T`
- 只用 `selected_mask` 禁止重复

因此它在数学上非常接近：**静态打分 + 逐步去重的 top-k 排序**。  
这会导致典型问题：

- shot=1/2：选最相似的几个通常没毛病  
- shot≥3：开始大量选到“互相很像”的 demo（冗余），甚至把噪声/错误 demo 塞进 prompt → 正确率下降  
- GRPO：就算你用 RL 训练，模型的“动作空间”实际上很难表达组合互补（因为每一步分数没随历史更新）

> 所以最核心的升级方向：**让 step t 的打分显式依赖已选历史（history-aware / set-aware）**。

---

## 2. 方案总览（按成功概率排序）

| 排名 | 方案代号 | 预计成功概率 | 实现难度 | 主要收益点 | 是否保持 RCE/GRPO 不变 |
|---:|---|---|---|---|---|
| 1 | V4-1：Cross-Attn + **Query 状态更新（V1 思路回归）** | 很高 | 很低 | 立刻让多步选择“有记忆” | ✅ |
| 2 | V4-2：Cross-Attn + **GRU Pointer Decoder** | 很高 | 低-中 | 更强的 history-aware 组合能力 | ✅ |
| 3 | V4-3：在 V4-1/2 上加 **Learnable MMR 多样性残差** | 高 | 低 | 专治 shot≥3 冗余 | ✅ |
| 4 | V4-4：**Candidate Set Encoder（Self-Attn）** + V4-2 | 中-高 | 中 | 候选之间先“互相看一眼”，更会去重 | ✅ |
| 5 | V4-5：把“点积”换成 **Additive/Bilinear Attention 打分头** | 中-高 | 中 | 解决 embedding 不完全同空间的问题 | ✅ |
| 6 | V4-6：**Coverage / Topic 原型覆盖**（自监督、无需额外标签） | 中 | 中 | 强化互补覆盖、减少重复 | ✅ |
| 7 | V4-7：**(N)DPP / log-det 风格的集合增益**（近似） | 中-低 | 高 | 强集合建模，但数值/工程更难 | ✅ |
| 8 | V4-8：**Slot/Set Decoder（并行 slots 协同）** + 兼容输出 logits | 中-低 | 高 | 直接做“集合预测”，有潜力 | ✅ |
| 9 | V4-9：Two-Stage **Coarse-to-Fine TopM 精排**（速度+可能更稳） | 中-低 | 中 | 主要利好效率，也可能提升鲁棒 | ✅ |
| 10 | V4-10：**STOP 自适应 shot**（需把 cand_pool 扩成 K+1） | 不确定 | 中-高 | 解决“shot 多反而伤”的根因 | ✅（但需要上游改 cand_emb） |

> 推荐探索路径：从 V4-1 → V4-2 → V4-3 走三步，你大概率就能看到“shot≥3 不再明显掉”的趋势；再往后逐步加 set encoder / coverage。

---

# 方案 1：V4-1（最推荐）Cross-Attn + Query 状态更新（V1 思路回归）

## 为什么成功概率很高（基于你项目）
- 你项目里 **V1 就有“选完一个 demo 更新 query”的机制**（`PointerSelectorV1.forward()` 里 `current_query = alpha * current_query + (1-alpha) * next_icd`）
- 但 V2 引入 Cross-Attn 后把“状态更新”丢了 → 多步选择退化成静态 top-k
- 所以最小改动：**保留 V2 的 Cross-Attn 增强，再加回“随已选 demo 更新 query_state”**  
  这能直接把 multi-shot 选择变成条件概率链：`p(a1|q) p(a2|q,a1) ...`

## 改动点（仅改模型，不碰 V3 的 RCE/GRPO）
文件：`lever_lm/models/v2/pointer_selector_v2.py`

- 在 `__init__` 新增一个可学习 gate（标量或向量都行）
- 在 `forward()` 的 step loop 里，选完一个 idx 后更新 `query_state`

## 伪代码（严格贴合你 V2 的变量/shape）

```py
class PointerSelectorV2(nn.Module):
    def __init__(...):
        ...
        # ✅ 新增：可学习的融合权重（建议做成向量 gating，更强）
        # 方案A：标量 gate（最简单）
        self.query_update_weight = nn.Parameter(torch.tensor(0.6))
        # 方案B：向量 gate（更强，推荐）
        # self.query_update_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(query_emb, cand_emb, labels=None, return_loss=True):
        B = query_emb.shape[0]
        device = query_emb.device
        actual_K = cand_emb.shape[1]
        input_dim = query_emb.shape[-1]

        # 1) input_proj
        query_reduced = self.input_proj(query_emb)                     # [B,H]
        cand_reduced  = self.input_proj(cand_emb.reshape(-1,input_dim)).reshape(B,actual_K,self.hidden_dim)

        # 2) 多层 Cross-Attn 只增强一次（保持原逻辑）
        query_for_attn = query_reduced.unsqueeze(1)                    # [B,1,H]
        for l in range(self.num_layers):
            attn_out,_ = self.cross_attn_layers[l](query_for_attn, cand_reduced, cand_reduced)
            query_for_attn = self.attn_norms[l](attn_out + query_for_attn)
        query_enhanced = query_for_attn.squeeze(1)                     # [B,H]

        # 3) 投影 + dropout + L2 normalize
        query_state = F.normalize(self.dropout(self.query_proj(query_enhanced)), p=2, dim=-1)  # [B,H]
        cand_proj   = F.normalize(self.dropout(self.cand_proj(cand_reduced)),   p=2, dim=-1)  # [B,K,H]

        selected_mask = torch.zeros(B, actual_K, dtype=torch.bool, device=device)
        all_logits, predictions = [], []

        for step in range(self.shot_num):
            # 4) 用“当前 query_state”打分（✅不再是静态 query）
            scores = (query_state.unsqueeze(1) @ cand_proj.transpose(1,2)).squeeze(1)         # [B,K]
            scores = scores / self.temperature.to(device)
            scores = scores.masked_fill(selected_mask, -100.0)

            all_logits.append(scores)
            pred = scores.argmax(dim=-1)                                                      # [B]
            predictions.append(pred)

            # 5) Teacher forcing 的 idx（训练用 label，推理用 pred）
            if labels is not None and step < labels.shape[1]:
                idx = labels[:, step]                                                         # [B]
            else:
                idx = pred

            # 6) 更新 mask（避免 inplace）
            selected_mask = selected_mask.scatter(1, idx.unsqueeze(1), True)

            # 7) 取出被选 demo 的 embedding
            chosen = cand_proj.gather(1, idx.view(B,1,1).expand(-1,1,self.hidden_dim)).squeeze(1)  # [B,H]

            # 8) ✅更新 query_state（让后续 step 条件化）
            # 方案A：标量 gate（与 V1 一致）
            alpha = torch.sigmoid(self.query_update_weight)                                   # scalar
            query_state = F.normalize(alpha * query_state + (1 - alpha) * chosen, p=2, dim=-1)

            # 方案B：向量 gate（更强）
            # g = torch.sigmoid(self.query_update_gate(torch.cat([query_state, chosen], dim=-1)))  # [B,H]
            # query_state = F.normalize(g*query_state + (1-g)*chosen, p=2, dim=-1)

        logits = torch.stack(all_logits, dim=1)        # [B,S,K]
        preds  = torch.stack(predictions, dim=1)       # [B,S]
        out = {"logits": logits, "predictions": preds}
        if return_loss and labels is not None:
            out["loss"] = self.compute_loss(logits, labels)
        return out
```

## 你应该重点看什么指标（快速判断是否有效）
- **shot3/shot4 是否不再显著下降**（你报告里 shot3 经常掉）
- GRPO 日志里：`Adv Std` 是否变大、`mean_ratio` 是否更容易偏离 1（说明策略真的在变）
- 推理时 top-k 的“集合重复率”是否降低：  
  `mean_{batch} mean_{i<j} cosine(cand[idx_i], cand[idx_j])` 应该更低

---

# 方案 2：V4-2 Cross-Attn + GRU Pointer Decoder（真正的“有状态指针网络”）

## 为什么成功概率很高
V4-1 的 gated update 是线性融合；GRU 可以学更复杂的“记忆/遗忘/组合策略”，在多步选择里通常更强。

## 改动点
仍然只改 `PointerSelectorV2`（或复制一个 `PointerSelectorV4Core`），V3 的 RL 代码无需动。

新增模块：
- `self.decoder = nn.GRUCell(hidden_dim, hidden_dim)`

## 伪代码（forward 核心）

```py
class PointerSelectorV2(nn.Module):
    def __init__(...):
        ...
        self.decoder_gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        # 可选：step embedding，让不同 step 学到不同策略
        self.step_emb = nn.Embedding(self.shot_num, self.hidden_dim)

    def forward(...):
        ...  # 同 V2：得到 query_enhanced, cand_proj

        # 初始 hidden state
        h = F.normalize(self.dropout(self.query_proj(query_enhanced)), p=2, dim=-1)  # [B,H]
        cand_proj = F.normalize(self.dropout(self.cand_proj(cand_reduced)), p=2, dim=-1)

        selected_mask = zeros([B,K])
        for step in range(self.shot_num):

            h_step = h + self.step_emb(step)     # [B,H]  (可选但推荐)
            h_step = F.normalize(h_step, p=2, dim=-1)

            # 用当前状态指向 candidates
            scores = (h_step.unsqueeze(1) @ cand_proj.transpose(1,2)).squeeze(1) / temperature
            scores = scores.masked_fill(selected_mask, -100.0)
            ...

            idx = labels[:,step] if training else pred
            selected_mask = selected_mask.scatter(1, idx.unsqueeze(1), True)

            chosen = cand_proj.gather(1, idx.view(B,1,1).expand(-1,1,H)).squeeze(1)  # [B,H]

            # ✅ GRU 更新 hidden state（history-aware）
            h = self.decoder_gru(chosen, h)        # [B,H]
            h = F.normalize(h, p=2, dim=-1)
```

## 工程建议
- GRU 初期可能不稳定：可以先只用 RCE 预热 1~2 epoch，再开 GRPO（你已有流程）。
- 如果担心过拟合：保留你当前的 dropout（0.1~0.5）+ weight decay。

---

# 方案 3：V4-3 在 V4-1/2 上加 Learnable MMR 多样性残差（shot≥3 必做）

## 为什么成功概率高
你现象里“shot1/2 提升更明显，shot3 下降”非常像冗余导致。MMR（Maximum Marginal Relevance）的思想就是：  
> 选第 t 个时，不只看“和 query 的相关性”，还要惩罚“和已选集合的相似度”。

这可以只在模型 forward 里做，不改 RCE/GRPO。

## 改动点
- forward 里维护 `selected_embs`（已选 cand_proj）
- 每步把冗余项从 scores 里减掉
- λ 做成可学习（per-step）

新增参数：
```py
self.div_lambda = nn.Parameter(torch.zeros(self.shot_num))  # 初始化为 0，等价于原模型，可平滑迁移
```

## 伪代码（在 step loop 内加入）

```py
selected_embs = []  # list of [B,H]

for step in range(S):
    base = (state.unsqueeze(1) @ cand_proj.transpose(1,2)).squeeze(1) / temperature  # [B,K]

    if step > 0:
        sel = torch.stack(selected_embs, dim=1)     # [B,step,H]
        # cosine，因为都 normalize 了，点积就是 cosine
        sim = torch.einsum("bkh,bth->bkt", cand_proj, sel)   # [B,K,step]
        redundancy = sim.max(dim=-1).values                   # [B,K] (也可用 mean)
        base = base - torch.relu(self.div_lambda[step]) * redundancy

    scores = base.masked_fill(selected_mask, -100.0)

    idx = labels[:,step] if training else argmax(scores)
    chosen = cand_proj.gather(1, idx.view(B,1,1).expand(-1,1,H)).squeeze(1)
    selected_embs.append(chosen)
    ...
```

## 小技巧（更优雅、更强）
- `redundancy` 用 `max` 往往比 `mean` 更像“去重”
- `div_lambda` 用 `softplus` 或 `relu` 保证非负，避免模型学出“鼓励重复”的奇怪行为

---

# 方案 4：V4-4 Candidate Set Encoder（Self-Attn）+ V4-2（更会处理候选重复）

## 直觉
V2 只有 query→cand 的 cross-attn，没有 cand↔cand 的交互。  
但“重复/冗余”本质是候选之间的关系，所以让 candidates 先 self-attn 一次通常更稳。

## 改动点
在 `cand_reduced` 上加一到两层 self-attn encoder：

新增模块：
```py
self.cand_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=1,
        dim_feedforward=hidden_dim*4,
        dropout=dropout, batch_first=True
    ),
    num_layers=1 or 2
)
```

## 伪代码（插入在 cand_reduced 后）

```py
cand_ctx = self.cand_encoder(cand_reduced)   # [B,K,H]
cand_proj = F.normalize(self.dropout(self.cand_proj(cand_ctx)), p=2, dim=-1)

# query_enhanced 仍然用 cross-attn（可选：让 query 也 attend cand_ctx）
...
```

> 推荐搭配 V4-2（GRU decoder）：cand_ctx 表达更强，stateful decoder 决策更强。

---

# 方案 5：V4-5 把 dot-product 换成 Additive / Bilinear Attention 打分头（提升可表达性）

## 适用场景
当你觉得：
- query embedding（来自某个 adapter/CLIP 分支）和 candidate embedding 的“可线性对齐性”不够  
- 纯 dot-product 过于刚性

可以换 scoring head，而**不改变整体管线**。

## 5.1 Additive（Bahdanau）Attention 伪代码

新增模块：
```py
self.attn_Wq = nn.Linear(H, H, bias=True)
self.attn_Wc = nn.Linear(H, H, bias=False)
self.attn_v  = nn.Linear(H, 1, bias=False)
```

step 内打分：
```py
# h: [B,H], cand_proj: [B,K,H]
q = self.attn_Wq(h).unsqueeze(1)             # [B,1,H]
c = self.attn_Wc(cand_proj)                  # [B,K,H]
scores = self.attn_v(torch.tanh(q + c)).squeeze(-1)   # [B,K]
scores = scores.masked_fill(selected_mask, -100.0)
```

## 5.2 Bilinear Attention 伪代码

新增模块：
```py
self.bilinear = nn.Bilinear(H, H, 1, bias=False)   # 输出标量
```

打分：
```py
# 广播到 [B,K]
scores = self.bilinear(h.unsqueeze(1).expand(-1,K,-1), cand_proj).squeeze(-1)
```

---

# 方案 6：V4-6 Coverage / Topic 原型覆盖（自监督、无需额外标签）

## 目标
让模型学会“互补覆盖”：后续 demo 更倾向于覆盖前面没覆盖到的“原型/簇”。

## 为什么不需要额外输入（严格兼容你 forward 签名）
我们在模型里引入 M 个“原型向量”（learnable prototypes），用 soft cluster 的方式给每个 candidate 一个 topic 分布；再让 query 预测自己需要哪些 topics。

## 新增模块
```py
M = 16  # topics/prototypes 数
self.topic_prototypes = nn.Parameter(torch.randn(M, H))
self.query_topic_head = nn.Linear(H, M, bias=True)
self.cover_lambda = nn.Parameter(torch.tensor(0.0))  # 先从0开始
```

## 伪代码（step loop 内加 coverage gain）

```py
# 预计算：每个 candidate 的 topic 分布
# cand_proj: [B,K,H], prototypes: [M,H]
proto = F.normalize(self.topic_prototypes, p=2, dim=-1)                 # [M,H]
topic_logits = cand_proj @ proto.t()                                     # [B,K,M]
topic_probs  = F.softmax(topic_logits, dim=-1)                           # [B,K,M]

# query 需要的 topics
need = F.softmax(self.query_topic_head(h0), dim=-1)                      # [B,M]

covered = torch.zeros(B, M, device=device)                               # [B,M]
for step in range(S):
    base = score(h, cand_proj)                                           # [B,K]

    # coverage gain：倾向选择能覆盖未覆盖 topic 的候选
    uncovered = (1.0 - covered).clamp(min=0.0, max=1.0)                  # [B,M]
    gain = torch.einsum("bm,bkm->bk", need * uncovered, topic_probs)      # [B,K]

    scores = base + torch.relu(self.cover_lambda) * gain
    scores = scores.masked_fill(selected_mask, -100.0)

    idx = labels[:,step] if training else argmax(scores)
    selected_mask = selected_mask.scatter(1, idx.unsqueeze(1), True)

    # 更新 covered（Teacher forcing 也用 idx）
    chosen_topic = topic_probs.gather(1, idx.view(B,1,1).expand(-1,1,M)).squeeze(1)  # [B,M]
    covered = (covered + chosen_topic).clamp(max=1.0)

    # 更新 state（可用 V4-1 或 V4-2）
    ...
```

## 注意
- M 不要太大（16/32 足够），否则不稳定
- `cover_lambda` 建议初始化 0，先让模型学相关性，再慢慢学覆盖

---

# 方案 7：V4-7 (N)DPP / log-det 风格集合增益（近似实现）

> 这是“更学术、更强集合建模”的路线，但实现和数值稳定性更挑战，成功概率中等偏低。

## 简化版可落地思路：用低秩特征做 logdet 近似增益
新增：
```py
r = 32
self.dpp_proj = nn.Linear(H, r, bias=False)
self.dpp_lambda = nn.Parameter(torch.tensor(0.0))
```

每个 candidate 的 DPP 特征：
```py
B = F.normalize(self.dpp_proj(cand_proj), p=2, dim=-1)   # [B,K,r]
```

增量多样性（近似）：  
用 “与已选集合的最大相似度” 近似 logdet 增益：
```py
sim = einsum("bkr,btr->bkt", B, B_selected)      # [B,K,t]
div_gain = torch.log(1e-6 + 1 - sim.max(-1).values.pow(2))  # [B,K]
scores = base + relu(dpp_lambda) * div_gain
```

> 如果你想做更“真”的 logdet：需要维护 Cholesky 分解/逆矩阵，工程量明显更大；建议先用上面的近似版本验证方向。

---

# 方案 8：V4-8 Slot/Set Decoder（并行 slots 协同）但仍输出 [B,S,K] logits

## 直觉
与其自回归一步步挑，不如同时维护 S 个“slot”，slots 之间 self-attn 协同分工，然后每个 slot 生成一行 logits。

为了兼容你现有训练（labels 是序列），我们仍然按 step 输出 logits，但 logits 由 slot 产生。

## 新增模块
```py
self.slot_emb = nn.Embedding(self.shot_num, H)     # 每个 slot 一个 learnable embedding
self.slot_self_attn = nn.MultiheadAttention(H, 1, dropout=attn_dropout, batch_first=True)
self.slot_norm = nn.LayerNorm(H)
```

## 伪代码

```py
# 初始化 slots
slots = query_proj.unsqueeze(1).expand(-1,S,-1) + self.slot_emb.weight.unsqueeze(0)  # [B,S,H]

# slots 自身协同
attn_out,_ = self.slot_self_attn(slots, slots, slots)
slots = self.slot_norm(slots + attn_out)                                           # [B,S,H]
slots = F.normalize(slots, p=2, dim=-1)

# 每个 slot 对 candidates 打分：得到 [B,S,K]
logits = torch.einsum("bsh,bkh->bsk", slots, cand_proj) / temperature

# 为了保证“不重复”，推理时仍然可用你现在的 greedy mask：
# step0 用 logits[:,0], mask idx0
# step1 用 logits[:,1] 但把 idx0 mask 掉
# ...
```

> 这条路线潜力大，但你需要仔细处理 “slot 输出的顺序” 与 labels 序列对齐的问题（否则监督会混乱）。  
> 推荐：训练时 labels 就按 reward/beam 排好固定顺序（你 RL 数据里通常是按 reward 降序），slot 也固定对应 step。

---

# 方案 9：V4-9 Two-Stage Coarse-to-Fine（TopM 精排，保持输出 K 不变）

## 目标
- 主要：提升速度、稳定性（把复杂计算集中在 topM）  
- 次要：有时也能提升质量（减少噪声候选干扰）

## 伪代码（每 step）

```py
# cheap score（点积）
cheap = dot(h, cand_proj)                           # [B,K]
cheap = cheap.masked_fill(selected_mask, -100.0)

# 选 topM 做精排
top_val, top_idx = cheap.topk(M, dim=-1)           # [B,M]
cand_sub = cand_proj.gather(1, top_idx[...,None].expand(-1,-1,H))  # [B,M,H]

# heavy refine（比如一个小 cross-attn / MLP）
refined_sub = refine(h, cand_sub)                  # [B,M]  输出 refined 分数

# scatter 回全量 K
scores = torch.full([B,K], -100.0, device=device)
scores = scores.scatter(1, top_idx, refined_sub)

# logits 仍是 [B,K]，完全兼容 RCE/GRPO
```

---

# 方案 10：V4-10 STOP 自适应 shot（需要把 cand_pool 扩成 K+1，但 RCE/GRPO 仍不变）

> 你报告里 shot3 常掉，这个方案直接解决“多选反而伤”的根因。  
> 但它不是纯模型改动：**需要上游构造 cand_emb 时追加一个 STOP 候选**，否则会违反 `logits.last_dim == cand_emb.shape[1]` 的硬约束。

## 上游改动（最小）
在生成 cand_emb 的地方（例如 embedding export / sampler）：
```py
stop_vec = stop_token.expand(B,1,d_model)                 # stop_token: learnable 或常量
cand_emb = torch.cat([cand_emb, stop_vec], dim=1)         # [B, K+1, d]
# labels / beam_labels 也允许出现 index = K 代表 STOP
# 若提前 STOP，则后续 step 全填 STOP（保持序列长度 S 不变）
```

## 模型 forward 伪代码（关键是“遇到 STOP 后冻结”）

```py
ended = torch.zeros(B, dtype=torch.bool, device=device)

for step in range(S):
    scores = score(h, cand_proj)                           # [B,K+1]
    scores = scores.masked_fill(selected_mask, -100.0)

    # 如果已经 ended：只允许选 STOP
    # stop_idx = actual_K-1 (因为 cand_emb 已经扩成 K+1)
    stop_idx = actual_K - 1
    scores = torch.where(
        ended.unsqueeze(1),
        torch.full_like(scores, -100.0).scatter(1, torch.full([B,1], stop_idx, device=device), 0.0),
        scores
    )

    idx = labels[:,step] if training else argmax(scores)

    ended = ended | (idx == stop_idx)

    selected_mask = selected_mask.scatter(1, idx.unsqueeze(1), True)
    update_state(...)
```

---

## 最后：我建议你怎么“逐步探索”（最稳）

1) **先做 V4-1**（最小改动，高概率立竿见影）  
2) 如果 shot≥3 仍掉：直接叠 **V4-3（MMR 多样性残差）**  
3) 需要更强：把状态更新从 gate 升级为 **V4-2（GRU）**  
4) 再追求稳定与上限：加 **V4-4（cand set encoder）** 或 **V4-6（coverage）**  
5) 如果你确认“shot 越多越伤”是系统性现象：再做 **V4-10 STOP**

---

## 附：你改完以后最容易踩的坑（与现有代码一致）

- **不要 in-place 改 selected_mask**：继续用 `selected_mask = selected_mask.scatter(...)`  
- **mask 值建议保持 -100.0**：你 `compute_loss` 里会 clamp 到 `min=-100`，这套数值稳定是配套的  
- **teacher forcing 时，状态更新也必须用 label idx**：否则 `compute_log_probs()` 会算错条件概率  
- **动态 K**：所有地方都用 `actual_K = cand_emb.shape[1]`，不要用 `self.K`

---

（完）


---

# 方案 5 实验结果：V4-5 Additive/Bilinear Attention（2025-12-27 更新）

## 5.3 训练配置

| 配置项 | V4-5 | 说明 |
|--------|------|------|
| **RL_DATA** | rl_data_k64_v3_balanced.json | 与方案五相同 |
| **RCE_EPOCHS** | **15** | 增加到15（因为Attention参数随机初始化需要更多训练） |
| GRPO_EPOCHS | 50 | 与方案五相同 |
| KL_BETA | 0.1 | 与方案五相同 |
| GRPO_LR | 5e-6 | 与方案五相同 |

## 5.4 实验结果（800 samples）

### V4-5 Additive Attention（Epoch 1）

| Shot | Baseline | 方案五 Epoch 2 | **V4-5 Additive Epoch 1** | V4-5 vs Baseline | V4-5 vs 方案五 |
|------|----------|----------------|---------------------------|------------------|----------------|
| **1** | 48.55% | 50.15% | **50.05%** | **+1.50%** ⬆️ | -0.10% ⬇️ |
| **2** | 47.75% | 48.33% | **48.65%** | **+0.90%** ⬆️ | **+0.32%** ⬆️ |
| **3** | 48.15% | 47.40% | 47.48% | -0.67% ⬇️ | +0.08% ⬆️ |
| **4** | 47.45% | 47.52% | **47.77%** | **+0.32%** ⬆️ | **+0.25%** ⬆️ |
| **平均** | 47.98% | 48.35% | **48.49%** | **+0.51%** ⬆️ | **+0.14%** ⬆️ |

### V4-5 Bilinear Attention（Epoch 2）

| Shot | Baseline | 方案五 Epoch 2 | **V4-5 Bilinear Epoch 2** | V4-5 vs Baseline | V4-5 vs 方案五 |
|------|----------|----------------|---------------------------|------------------|----------------|
| **1** | 48.55% | 50.15% | **50.15%** | **+1.60%** ⬆️ | **0.00%** ➡️ |
| **2** | 47.75% | 48.33% | 47.55% | -0.20% ⬇️ | -0.78% ⬇️ |
| **3** | 48.15% | 47.40% | 47.10% | -1.05% ⬇️ | -0.30% ⬇️ |
| **4** | 47.45% | 47.52% | 47.15% | -0.30% ⬇️ | -0.37% ⬇️ |
| **平均** | 47.98% | 48.35% | 47.99% | +0.01% ⬆️ | -0.36% ⬇️ |

### Additive vs Bilinear 对比

| Shot | V4-5 Additive | V4-5 Bilinear | 差异 | 更优 |
|------|---------------|---------------|------|------|
| **1** | 50.05% | **50.15%** | +0.10% | Bilinear |
| **2** | **48.65%** | 47.55% | -1.10% | **Additive** 🏆 |
| **3** | **47.48%** | 47.10% | -0.38% | **Additive** 🏆 |
| **4** | **47.77%** | 47.15% | -0.62% | **Additive** 🏆 |
| **平均** | **48.49%** | 47.99% | -0.50% | **Additive** 🏆 |

## 5.5 结论

1. **V4-5 Additive Attention 是目前最优方案**：平均准确率 48.49%，超过方案五 0.14%
2. **Additive 明显优于 Bilinear**：平均差距 0.50%
3. **Additive 在 Shot 2/4 上表现最好**：
   - Shot 2: 48.65%（超过方案五 0.32%）
   - Shot 4: 47.77%（超过方案五 0.25%）
4. **Bilinear 不推荐**：仅 Shot 1 与方案五持平，其他 shot 均较差
5. **训练注意事项**：需要增加 RCE epochs 到 15，因为 Attention 参数是随机初始化的

## 5.6 Checkpoint 位置

- V4-5 Additive: `results/okvqa/model_cpk/v3_plan_v4_5_additive/grpo_epoch1.pt`
- V4-5 Bilinear: `results/okvqa/model_cpk/v3_plan_v4_5_bilinear/grpo_epoch2.pt`

## 5.7 推荐使用

```bash
# 训练 V4-5 Additive（推荐）
bash scripts/train_v3_plan_v4_5.sh [gpu_id] additive

# 推理
bash scripts/inference_v4_5.sh [gpu_id] 1 additive
```


---

# 方案 7 实现：V4-7 (N)DPP / log-det 风格集合增益（2025-12-27 实现）

## 7.1 实现概述

V4-7 方案已完成实现，核心特点：
- 使用低秩特征做 logdet 近似增益
- 用 "与已选集合的最大相似度" 近似 logdet 增益
- 强集合建模，增强多样性选择能力

## 7.2 核心改动

### 新增模块
```python
# DPP 低秩投影矩阵：将 hidden_dim 投影到 dpp_rank 维
self.dpp_proj = nn.Linear(hidden_dim, dpp_rank, bias=False)

# 可学习的 DPP 增益权重
self.dpp_lambda = nn.Parameter(torch.tensor(dpp_lambda_init if dpp_lambda_init != 0.0 else -2.0))
```

### 核心算法（每步计算 diversity gain）
```python
# 预计算 DPP 特征
dpp_features = F.normalize(self.dpp_proj(cand_proj_norm), p=2, dim=-1)  # [B, K, r]

# 计算每个候选与已选集合的相似度
if step > 0 and len(selected_dpp_features) > 0:
    selected_stack = torch.stack(selected_dpp_features, dim=1)  # [B, step, r]
    sim = torch.einsum("bkr,btr->bkt", dpp_features, selected_stack)  # [B, K, step]
    max_sim = sim.max(dim=-1).values  # [B, K]
    
    # DPP diversity gain: log(1 - sim^2) 的近似
    diversity_gain = torch.log(1e-6 + 1.0 - max_sim.pow(2).clamp(max=0.999))
    scores = base_scores + F.softplus(self.dpp_lambda) * diversity_gain
```

## 7.3 文件结构

| 文件 | 说明 |
|------|------|
| `lever_lm/models/v2/pointer_selector_v4_7.py` | V4-7 基础模型 |
| `lever_lm/models/v3/pointer_selector_v4_7_rl.py` | V4-7 RL 版本（RCE + GRPO） |
| `lever_lm/workflows/grpo_post_train_v4_7.py` | V4-7 训练 workflow |
| `scripts/train_v3_plan_v4_7.sh` | 训练脚本 |
| `scripts/inference_v4_7.sh` | 推理脚本 |
| `scripts/convert_v4_7_to_v2_format.py` | Checkpoint 转换脚本 |

## 7.4 训练配置

| 配置项 | V4-7 | 说明 |
|--------|------|------|
| **RL_DATA** | rl_data_k64_v3_balanced.json | 与其他方案相同 |
| **RCE_EPOCHS** | **15** | 增加到15（因为 DPP 参数需要更多训练） |
| GRPO_EPOCHS | 50 | 与其他方案相同 |
| KL_BETA | 0.1 | 与其他方案相同 |
| GRPO_LR | 5e-6 | 与其他方案相同 |
| **dpp_rank** | 32 | DPP 低秩投影维度（可调整：16/32/64） |

## 7.5 使用方法

### 训练
```bash
# 使用默认 dpp_rank=32
bash scripts/train_v3_plan_v4_7.sh [gpu_id]

# 指定 dpp_rank
bash scripts/train_v3_plan_v4_7.sh [gpu_id] 64
```

### 推理
```bash
# 使用 RCE epoch 2
bash scripts/inference_v4_7.sh [gpu_id] rce 2 32

# 使用 GRPO epoch 1
bash scripts/inference_v4_7.sh [gpu_id] grpo 1 32
```

## 7.6 预期效果

根据文档分析，V4-7 的预期效果：
- **成功概率**：中-低（因为数值/工程更难）
- **主要收益**：强集合建模，增强多样性
- **适用场景**：当 shot≥3 出现明显冗余时

## 7.7 实验结果

（待训练完成后更新）

| Shot | Baseline | V4-5 Additive | **V4-7 DPP** | V4-7 vs Baseline | V4-7 vs V4-5 |
|------|----------|---------------|--------------|------------------|--------------|
| **1** | 48.55% | 50.05% | - | - | - |
| **2** | 47.75% | 48.65% | - | - | - |
| **3** | 48.15% | 47.48% | - | - | - |
| **4** | 47.45% | 47.77% | - | - | - |
| **平均** | 47.98% | 48.49% | - | - | - |

## 7.8 注意事项

1. **dpp_rank 选择**：
   - 默认 32，与 hidden_dim=256 配合
   - 太小（<16）可能表达能力不足
   - 太大（>64）可能过拟合

2. **dpp_lambda 初始化**：
   - 默认 0.0（实际使用 softplus(-2.0) ≈ 0.127）
   - 训练过程中会自动学习

3. **数值稳定性**：
   - diversity_gain 使用 `log(1e-6 + 1.0 - sim^2.clamp(max=0.999))` 避免数值问题
   - 当 sim 接近 1 时，gain 为负（惩罚重复）
