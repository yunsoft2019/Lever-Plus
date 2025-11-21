#!/bin/bash

# 压缩 results 目录为 8GB 分卷的脚本
# 使用 7za 进行分段压缩

# 设置变量
SOURCE_DIR="/mnt/share/yiyun/Projects/Lever-Plus/results"
ARCHIVE_NAME="results_$(date +%Y%m%d_%H%M%S).7z"
OUTPUT_DIR="/mnt/share/yiyun/Projects/Lever-Plus"
VOLUME_SIZE="8G"

# 检查 7za 是否安装
if ! command -v 7za &> /dev/null; then
    echo "错误: 7za 未安装。请先安装 7zip:"
    echo "  Ubuntu/Debian: sudo apt-get install p7zip-full"
    echo "  CentOS/RHEL: sudo yum install p7zip p7zip-plugins"
    exit 1
fi

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录不存在: $SOURCE_DIR"
    exit 1
fi

# 显示压缩信息
echo "=========================================="
echo "开始压缩 results 目录"
echo "=========================================="
echo "源目录: $SOURCE_DIR"
echo "输出文件: $OUTPUT_DIR/$ARCHIVE_NAME"
echo "分卷大小: $VOLUME_SIZE"
echo "=========================================="
echo ""

# 切换到输出目录
cd "$OUTPUT_DIR" || exit 1

# 执行压缩
# -a: 添加文件到压缩包
# -v{size}: 创建分卷，每个分卷大小为指定值
# -mx=1: 压缩级别（1=最快压缩，9=最好压缩，0=不压缩）
# -mmt=on: 启用多线程
# -m0=lzma2: 使用 LZMA2 压缩算法
# -mmt=on: 多线程压缩
echo "开始压缩，这可能需要一些时间..."
echo ""

7za a \
    -v${VOLUME_SIZE} \
    -mx=1 \
    -mmt=on \
    -m0=lzma2 \
    "$ARCHIVE_NAME" \
    "$SOURCE_DIR"

# 检查压缩是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "压缩完成！"
    echo "=========================================="
    echo "压缩文件位置: $OUTPUT_DIR/$ARCHIVE_NAME.*"
    echo ""
    echo "生成的分卷文件："
    ls -lh "$OUTPUT_DIR"/${ARCHIVE_NAME}.* 2>/dev/null | awk '{print $9, "(" $5 ")"}'
    echo ""
    echo "总大小："
    du -sh "$OUTPUT_DIR"/${ARCHIVE_NAME}.* 2>/dev/null | tail -1
    echo ""
    echo "解压命令示例："
    echo "  7za x ${ARCHIVE_NAME}.001"
    echo "  或"
    echo "  7za x ${ARCHIVE_NAME}.7z.001"
else
    echo ""
    echo "错误: 压缩失败！"
    exit 1
fi

