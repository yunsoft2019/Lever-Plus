#!/bin/bash
# 包装脚本，确保在正确目录下运行
PROJECT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
cd "$PROJECT_DIR"
eval "$(conda shell.bash hook)"
conda activate lever_env
bash scripts/generate_data.sh "$@"
