# Requirements Document

## Introduction

修复 RL 训练阶段的候选池维度一致性问题，并实现全局正样本注入机制。当前 RL 训练将候选池从 SFT 阶段的 64 个缩小到约 20 个，导致训练-推理任务不一致，SFT 学到的能力被破坏。本需求旨在：
1. 恢复 RL 训练时的候选池维度为 64（与 SFT 一致）
2. 通过全局正样本挖掘，将能产生正样本的 ICD 注入到 64 候选池中
3. 确保 RL 训练的改进能迁移到真实推理场景

## Glossary

- **Pointer Selector**: 指针选择器模型，从候选池中选择 ICD 的神经网络
- **ICD (In-Context Demonstration)**: 上下文示例，用于 few-shot 学习的示例
- **候选池 (Candidate Pool)**: 每个 query 对应的 K 个 ICD 候选集合（K=64）
- **局部索引 (Local Index)**: 候选池内的位置索引（0~63）
- **全局索引 (Global Index)**: 数据集中的样本索引（0~8000+）
- **Pointer/Trajectory**: 长度为 2 的序列，表示选择的两个 ICD 的局部索引 [pos_i, pos_j]
- **正样本 (Positive Sample)**: 能使 VQA 模型得到正确答案的 ICD 序列（vqa_correct=1）
- **负样本 (Negative Sample)**: 不能得到正确答案的 ICD 序列（vqa_correct=0）
- **All-Zero Query**: 在当前 64 候选池中找不到任何正样本的 query
- **Oracle@64**: 在 64 候选池内能达到的最高 acc_score

## Requirements

### Requirement 1: 候选池维度一致性修复

**User Story:** As a 研究员, I want RL 训练时的候选池维度与 SFT 一致（K=64）, so that 训练-推理任务定义一致，SFT 学到的能力不被破坏。

#### Acceptance Criteria

1. WHEN 加载 RL 训练数据 THEN the RLBeamDatasetWithEmbedding SHALL 确保每个 query 的候选池大小为 64
2. WHEN 候选池大小不等于 64 THEN the System SHALL 抛出断言错误并提示具体问题
3. WHEN 构建 cand_emb THEN the System SHALL 使用完整的 64 个候选 embedding，而不是从 trajectory 中提取
4. WHEN trajectory 中的 pointer 索引超出 [0, 63] 范围 THEN the System SHALL 抛出断言错误

### Requirement 2: RL 数据格式规范化

**User Story:** As a 研究员, I want RL 数据包含完整的 64 候选池信息, so that 训练时能正确构建候选池。

#### Acceptance Criteria

1. WHEN 生成 RL 数据 THEN the System SHALL 为每个 query 保存 candidate_pool_indices 字段（长度=64 的全局索引列表）
2. WHEN 保存 trajectory THEN the System SHALL 使用局部索引（0~63）表示 pointer
3. WHEN 合并或优化 RL 数据 THEN the System SHALL 保持 candidate_pool_indices 不变（长度始终为 64）
4. WHEN 替换候选池中的 ICD THEN the System SHALL 更新 candidate_pool_indices 并重新映射所有 trajectory 的 pointer

### Requirement 3: 全局正样本挖掘

**User Story:** As a 研究员, I want 从全局候选池中挖掘能产生正样本的 ICD, so that All-Zero query 的候选池具备可救性。

#### Acceptance Criteria

1. WHEN 处理 All-Zero query THEN the System SHALL 从全局候选池中搜索能产生正样本的 ICD pair
2. WHEN 全局搜索 THEN the System SHALL 使用 embedding 相似度作为初筛条件，限制搜索范围（如 Top-300）
3. WHEN 找到正样本 pair THEN the System SHALL 记录该 pair 的全局索引和 acc_score
4. WHEN 达到搜索预算上限或找到足够正样本 THEN the System SHALL 停止搜索以节省算力

### Requirement 4: 正样本注入到 64 候选池

**User Story:** As a 研究员, I want 将全局挖掘到的正样本 ICD 注入到 64 候选池中, so that 在不扩大候选池的前提下提升 oracle 上限。

#### Acceptance Criteria

1. WHEN 注入正样本 ICD THEN the System SHALL 保持候选池大小为 64 不变
2. WHEN 选择被替换的 ICD THEN the System SHALL 优先替换相似度最低或冗余度最高的 ICD
3. WHEN 注入新 ICD THEN the System SHALL 更新 candidate_pool_indices 并重新映射所有 trajectory 的 pointer
4. WHEN 注入完成 THEN the System SHALL 验证 oracle@64 是否提升、All-Zero 占比是否下降
5. WHEN 注入数量 THEN the System SHALL 限制每个 query 最多注入 m 个新 ICD（建议 m=4~12）

### Requirement 5: 训练轨迹质量控制

**User Story:** As a 研究员, I want 控制训练轨迹的数量和质量, so that 避免分布稀释导致性能下降。

#### Acceptance Criteria

1. WHEN 添加新 trajectory THEN the System SHALL 只保留能提升 max(acc_score) 或增加 unique(acc_score) 的轨迹
2. WHEN trajectory 不满足提升条件 THEN the System SHALL 丢弃该轨迹不入库
3. WHEN 每个 query 的轨迹数量 THEN the System SHALL 固定为 M 条（建议 M=12~16）
4. WHEN 分配轨迹配额 THEN the System SHALL 按 acc_score 分层：正样本(1.0) 2~4 条、部分正确(0.6) 1~2 条、负样本(0) 补齐剩余

### Requirement 6: 验证与监控

**User Story:** As a 研究员, I want 验证修复效果并监控关键指标, so that 确保改进有效且可迁移。

#### Acceptance Criteria

1. WHEN 数据集初始化 THEN the System SHALL 打印候选池维度统计（确认全部为 64）
2. WHEN 注入完成 THEN the System SHALL 统计并打印 All-Zero query 数量变化
3. WHEN 注入完成 THEN the System SHALL 统计并打印 oracle@64 均值变化
4. WHEN 训练完成 THEN the System SHALL 对比三组结果：base+原64、base+注入后64、RL+注入后64
