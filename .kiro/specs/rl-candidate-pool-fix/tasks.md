# Implementation Plan

## Phase A: 候选池维度一致性修复

- [ ] 1. 修改 RLBeamDatasetWithEmbedding 支持固定 64 候选池
  - [ ] 1.1 添加 candidate_pool_indices 字段解析逻辑
    - 从 rl_data 中读取 candidate_pool_indices
    - 如果不存在，从 pointer_candidates 中提取并补齐到 64
    - _Requirements: 1.1, 2.1_
  - [ ] 1.2 添加候选池大小断言检查
    - 检查 len(candidate_pool_indices) == 64
    - 检查 pointer 索引在 [0, 63] 范围内
    - _Requirements: 1.2, 1.4_
  - [ ]* 1.3 Write property test for 候选池大小不变性
    - **Property 1: 候选池大小不变性**
    - **Validates: Requirements 1.1, 2.1, 2.3, 4.1**
  - [ ] 1.4 修改 __getitem__ 使用 candidate_pool_indices 构建 cand_emb
    - cand_emb = img_emb_data[candidate_pool_indices]
    - 确保 cand_emb.shape[1] == 64
    - _Requirements: 1.3_
  - [ ]* 1.5 Write property test for Pointer 范围有效性
    - **Property 2: Pointer 范围有效性**
    - **Validates: Requirements 1.4, 2.2**

- [ ] 2. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Phase B: RL 数据格式规范化脚本

- [ ] 3. 创建 RL 数据规范化脚本
  - [ ] 3.1 实现 normalize_rl_data 函数
    - 为每个 query 添加 candidate_pool_indices 字段
    - 将 pointer 从全局索引转换为局部索引
    - 保持候选池大小为 64
    - _Requirements: 2.1, 2.2, 2.3_
  - [ ] 3.2 实现索引映射工具函数
    - local_to_global(local_idx, pool_indices)
    - global_to_local(global_idx, pool_indices)
    - _Requirements: 2.4_
  - [ ]* 3.3 Write property test for 索引映射一致性
    - **Property 3: 索引映射一致性**
    - **Validates: Requirements 2.4, 4.3**

- [ ] 4. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Phase C: 全局正样本挖掘与注入

- [ ] 5. 实现全局正样本挖掘器
  - [ ] 5.1 实现 GlobalPositiveMiner 类
    - 使用 embedding 相似度初筛 Top-K 候选
    - 调用 VQA 模型评测 pair
    - 记录正样本 pair 的全局索引和 acc_score
    - _Requirements: 3.1, 3.2, 3.3_
  - [ ] 5.2 实现搜索预算控制
    - 达到 max_eval_pairs 或找到足够正样本时停止
    - _Requirements: 3.4_

- [ ] 6. 实现候选池注入器
  - [ ] 6.1 实现 CandidatePoolInjector 类
    - 计算当前池中每个 ICD 与 query 的相似度
    - 选择相似度最低的 ICD 进行替换
    - _Requirements: 4.2_
  - [ ] 6.2 实现注入后的 pointer 重映射
    - 更新 candidate_pool_indices
    - 重新映射所有 trajectory 的 pointer
    - _Requirements: 4.3_
  - [ ]* 6.3 Write property test for 注入数量限制
    - **Property 4: 注入数量限制**
    - **Validates: Requirements 4.5**
  - [ ] 6.4 实现注入效果验证
    - 统计 oracle@64 变化
    - 统计 All-Zero 占比变化
    - _Requirements: 4.4_

- [ ] 7. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Phase D: 轨迹质量控制

- [ ] 8. 实现轨迹质量控制器
  - [ ] 8.1 实现 TrajectoryQualityController 类
    - should_commit 方法：判断新轨迹是否应提交
    - select_final_trajectories 方法：选择最终 M 条轨迹
    - _Requirements: 5.1, 5.2, 5.3_
  - [ ]* 8.2 Write property test for Commit-on-improve 有效性
    - **Property 5: Commit-on-improve 有效性**
    - **Validates: Requirements 5.1, 5.2**
  - [ ] 8.3 实现轨迹配额分配
    - 正样本(1.0): 2~4 条
    - 部分正确(0.6): 1~2 条
    - 负样本(0): 补齐剩余
    - _Requirements: 5.4_
  - [ ]* 8.4 Write property test for 轨迹数量固定
    - **Property 6: 轨迹数量固定**
    - **Validates: Requirements 5.3**

- [ ] 9. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Phase E: 集成与验证

- [ ] 10. 创建完整的数据处理流水线脚本
  - [ ] 10.1 整合所有组件
    - 加载原始 RL 数据
    - 规范化数据格式
    - 全局正样本挖掘（仅 All-Zero）
    - 注入到 64 候选池
    - 轨迹质量控制
    - 保存最终数据
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_
  - [ ] 10.2 添加验证与监控输出
    - 打印候选池维度统计
    - 打印 All-Zero 数量变化
    - 打印 oracle@64 均值变化
    - _Requirements: 6.1, 6.2, 6.3_

- [ ] 11. Final Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.
