#!/usr/bin/env python3
"""
修复Qwen束搜索数据文件中的负数分数，统一为正数（取绝对值）
这样可以与Flamingo的计分方式保持一致
"""

import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_qwen_scores_in_file(file_path: str):
    """修复单个文件中的负数分数"""
    logger.info(f"处理文件: {file_path}")
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计修改信息
    total_samples = 0
    total_scores = 0
    fixed_scores = 0
    
    # 遍历所有样本
    for query_id, sample_data in data.items():
        if 'score_list' not in sample_data:
            continue
        
        total_samples += 1
        score_list = sample_data['score_list']
        total_scores += len(score_list)
        
        # 检查并修复负数分数
        fixed_list = []
        for score in score_list:
            if score < 0:
                fixed_list.append(abs(score))
                fixed_scores += 1
            else:
                fixed_list.append(score)
        
        # 更新分数列表
        sample_data['score_list'] = fixed_list
    
    # 保存修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"完成: 处理了 {total_samples} 个样本，共 {total_scores} 个分数，修复了 {fixed_scores} 个负数分数")
    return total_samples, total_scores, fixed_scores

def main():
    """主函数：处理所有Qwen数据文件"""
    # 数据文件目录
    data_dir = Path("/mnt/share/yiyun/Projects/Lever-Plus/results/okvqa/generated_data")
    
    # 查找所有Qwen数据文件
    qwen_files = list(data_dir.glob("*Qwen2_5-VL-3B-Instruct*.json"))
    
    if not qwen_files:
        logger.warning(f"未找到Qwen数据文件，目录: {data_dir}")
        return
    
    logger.info(f"找到 {len(qwen_files)} 个Qwen数据文件")
    
    # 统计信息
    total_samples = 0
    total_scores = 0
    total_fixed = 0
    
    # 处理每个文件
    for file_path in qwen_files:
        samples, scores, fixed = fix_qwen_scores_in_file(str(file_path))
        total_samples += samples
        total_scores += scores
        total_fixed += fixed
    
    logger.info("=" * 60)
    logger.info(f"全部完成！")
    logger.info(f"总计: {total_samples} 个样本，{total_scores} 个分数，修复了 {total_fixed} 个负数分数")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

