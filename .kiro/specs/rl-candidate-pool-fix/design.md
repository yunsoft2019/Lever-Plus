# Design Document: RL 候选池维度一致性修复 + 全局正样本注入

## Overview

本设计解决 RL 训练阶段候选池维度与 SFT 不一致的问题。核心改动包括：
1. 修复 `RLBeamDatasetWithEmbedding` 使用完整 64 候选池
2. 规范化 RL 数据格式，保存 `candidate_pool_indices`
3. 实现全局正样本挖掘和注入机制
4. 添加训练轨迹质量控制（Commit-on-improve）

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RL 数据处理流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 原始 RL 数据  │───▶│ 全局正样本   │───▶│ 注入到 64    │      │
│  │ (K 不固定)   │    │ 挖掘         │    │ 候选池       │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│                                          ┌──────────────┐      │
│                                          │ 轨迹质量控制  │      │
│                                          │ (Commit-on-  │      │
│                                          │  improve)    │      │
│                                          └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│                                          ┌──────────────┐      │
│                                          │ 规范化 RL    │      │
│                                          │ 数据 (K=64)  │      │
│                                          └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              RLBeamDatasetWithEmbedding                   │  │
│  │  - candidate_pool_indices: [64] (全局索引)                │  │
│  │  - cand_emb: [64, d] (从 img_emb_data 按索引取)           │  │
│  │  - trajectories: [[pos_i, pos_j], ...] (局部索引 0~63)    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. RL 数据格式规范

```python
# 新的 RL 数据格式（每个 query）
{
    "query_id": 123,
    "candidate_pool_indices": [1001, 2034, ...],  # 长度=64，全局索引
    "pointer_candidates": [
        {
            "pointer": [3, 27],  # 局部索引（0~63）
            "pointer_global": [1001, 2034],  # 对应的全局索引（可选，用于调试）
            "vqa_correct": 1,
            "vqa_acc_score": 1.0,
            "gen_method": "beam",
            ...
        },
        ...
    ]
}
```

### 2. RLBeamDatasetWithEmbedding 修改

```python
class RLBeamDatasetWithEmbedding(Dataset):
    def __init__(
        self,
        rl_data: Dict,
        img_emb_data: torch.Tensor,  # [N, d] 完整的 embedding 数据
        expected_pool_size: int = 64,  # 期望的候选池大小
        ...
    ):
        # 断言检查
        for query_id, query_data in rl_data.items():
            pool_indices = query_data.get("candidate_pool_indices", [])
            assert len(pool_indices) == expected_pool_size, \
                f"Query {query_id}: 候选池大小 {len(pool_indices)} != {expected_pool_size}"
    
    def __getitem__(self, index):
        # 使用 candidate_pool_indices 构建 cand_emb
        pool_indices = sample["candidate_pool_indices"]
        cand_emb = self.img_emb_data[pool_indices]  # [64, d]
        
        # pointer 已经是局部索引，直接使用
        beam_labels = sample["beam_labels"]  # [[pos_i, pos_j], ...]
        
        return {
            "cand_emb": cand_emb,  # [64, d]
            "beam_labels": beam_labels,
            ...
        }
```

### 3. 全局正样本挖掘器

```python
class GlobalPositiveMiner:
    def __init__(
        self,
        img_emb_data: torch.Tensor,  # [N, d]
        vqa_interface,  # VQA 模型接口
        dataset,  # 原始数据集
        search_top_k: int = 300,  # 初筛 Top-K
        max_eval_pairs: int = 50,  # 最大评测 pair 数
    ):
        pass
    
    def mine_positives_for_query(
        self,
        query_id: int,
        current_pool_indices: List[int],
    ) -> List[Dict]:
        """
        为单个 query 挖掘全局正样本
        
        Returns:
            List[Dict]: 找到的正样本 pair 列表
            [{"global_indices": [idx1, idx2], "acc_score": 0.8}, ...]
        """
        pass
```

### 4. 候选池注入器

```python
class CandidatePoolInjector:
    def __init__(
        self,
        img_emb_data: torch.Tensor,
        max_inject_per_query: int = 8,  # 每个 query 最多注入数量
    ):
        pass
    
    def inject_positives(
        self,
        query_data: Dict,
        positive_icds: List[int],  # 要注入的全局索引
    ) -> Dict:
        """
        将正样本 ICD 注入到 64 候选池中
        
        策略：
        1. 计算当前池中每个 ICD 与 query 的相似度
        2. 选择相似度最低的 ICD 进行替换
        3. 更新 candidate_pool_indices
        4. 重新映射所有 trajectory 的 pointer
        
        Returns:
            更新后的 query_data
        """
        pass
```

### 5. 轨迹质量控制器

```python
class TrajectoryQualityController:
    def __init__(
        self,
        target_trajectory_count: int = 12,  # 目标轨迹数
        positive_quota: int = 4,  # 正样本配额
        partial_quota: int = 2,  # 部分正确配额
    ):
        pass
    
    def should_commit(
        self,
        current_trajectories: List[Dict],
        new_trajectory: Dict,
    ) -> bool:
        """
        判断新轨迹是否应该提交
        
        Commit-on-improve 规则：
        1. 让 max(acc_score) 变大
        2. 让 unique(acc_score) 变多
        """
        pass
    
    def select_final_trajectories(
        self,
        all_trajectories: List[Dict],
    ) -> List[Dict]:
        """
        从所有轨迹中选择最终的 M 条
        
        配额分配：
        - acc_score=1.0: 2~4 条
        - acc_score=0.6: 1~2 条
        - acc_score=0: 补齐剩余
        """
        pass
```

## Data Models

### RL 数据结构

```python
@dataclass
class RLQueryData:
    query_id: int
    candidate_pool_indices: List[int]  # 长度=64，全局索引
    pointer_candidates: List[TrajectoryData]

@dataclass
class TrajectoryData:
    pointer: List[int]  # 长度=2，局部索引 [0, 63]
    pointer_global: List[int]  # 长度=2，全局索引（可选）
    vqa_correct: int  # 0 或 1
    vqa_acc_score: float  # [0, 1]
    gen_method: str  # "beam", "sample", "random", "global_positive_transfer"
```

### 索引映射

```python
# 局部索引 -> 全局索引
def local_to_global(local_idx: int, pool_indices: List[int]) -> int:
    return pool_indices[local_idx]

# 全局索引 -> 局部索引
def global_to_local(global_idx: int, pool_indices: List[int]) -> int:
    return pool_indices.index(global_idx)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 候选池大小不变性
*For any* query 在 RL 数据集中, 其 candidate_pool_indices 的长度应始终等于 64
**Validates: Requirements 1.1, 2.1, 2.3, 4.1**

### Property 2: Pointer 范围有效性
*For any* trajectory 中的 pointer [pos_i, pos_j], 两个索引都应在 [0, 63] 范围内
**Validates: Requirements 1.4, 2.2**

### Property 3: 索引映射一致性
*For any* trajectory, 其 pointer（局部索引）通过 candidate_pool_indices 映射后应得到正确的全局索引
**Validates: Requirements 2.4, 4.3**

### Property 4: 注入数量限制
*For any* query 的注入操作, 注入的新 ICD 数量不应超过 max_inject_per_query
**Validates: Requirements 4.5**

### Property 5: Commit-on-improve 有效性
*For any* 被提交的新 trajectory, 它应满足以下条件之一：提升 max(acc_score) 或增加 unique(acc_score)
**Validates: Requirements 5.1, 5.2**

### Property 6: 轨迹数量固定
*For any* query 在最终数据集中, 其 trajectory 数量应等于 target_trajectory_count
**Validates: Requirements 5.3**

## Error Handling

1. **候选池大小不匹配**
   - 检测：`len(candidate_pool_indices) != 64`
   - 处理：抛出 `AssertionError` 并打印具体 query_id 和实际大小

2. **Pointer 索引越界**
   - 检测：`pointer[i] < 0 or pointer[i] >= 64`
   - 处理：抛出 `IndexError` 并打印具体 trajectory 信息

3. **全局索引不存在**
   - 检测：`global_idx >= len(img_emb_data)`
   - 处理：抛出 `IndexError` 并提示检查数据一致性

4. **注入后无改善**
   - 检测：注入后 oracle@64 未提升
   - 处理：记录警告日志，但不阻止流程

## Testing Strategy

### 单元测试

1. **候选池大小断言测试**
   - 测试正常数据（K=64）能正常加载
   - 测试异常数据（K!=64）抛出断言错误

2. **索引映射测试**
   - 测试 local_to_global 和 global_to_local 的正确性
   - 测试注入后映射的一致性

3. **Commit-on-improve 逻辑测试**
   - 测试提升 max(acc_score) 的情况
   - 测试增加 unique(acc_score) 的情况
   - 测试不满足条件被拒绝的情况

### Property-Based Testing

使用 `hypothesis` 库进行属性测试：

1. **Property 1 测试**：生成随机 RL 数据，验证候选池大小始终为 64
2. **Property 2 测试**：生成随机 trajectory，验证 pointer 范围
3. **Property 3 测试**：生成随机注入操作，验证映射一致性
4. **Property 5 测试**：生成随机 trajectory 序列，验证 commit 逻辑

### 集成测试

1. **端到端数据处理测试**
   - 输入：原始 RL 数据（K 不固定）
   - 输出：规范化 RL 数据（K=64）
   - 验证：所有 query 的候选池大小、pointer 范围、映射一致性

2. **训练兼容性测试**
   - 使用规范化数据初始化 `RLBeamDatasetWithEmbedding`
   - 验证 `cand_emb.shape[1] == 64`
   - 验证训练循环能正常运行
