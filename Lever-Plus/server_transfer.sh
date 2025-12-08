#!/bin/bash
# 服务器间数据传输脚本
# 使用方法: ./server_transfer.sh [source] [destination] [method]

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 显示使用方法
show_usage() {
    echo -e "${BLUE}使用方法:${NC}"
    echo "  $0 <源路径> <目标服务器用户@IP:目标路径> [方法]"
    echo ""
    echo -e "${BLUE}示例:${NC}"
    echo "  $0 /data/files user@192.168.1.100:/home/user/backup"
    echo "  $0 /data/files user@192.168.1.100:/home/user/backup rsync"
    echo ""
    echo -e "${BLUE}方法选项:${NC}"
    echo "  scp    - 使用 scp 传输（默认）"
    echo "  rsync  - 使用 rsync 传输（推荐，支持断点续传）"
}

# 检查参数
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

SOURCE=$1
DEST=$2
METHOD=${3:-scp}

# 检查源路径是否存在
if [ ! -e "$SOURCE" ]; then
    echo -e "${RED}错误: 源路径 '$SOURCE' 不存在${NC}"
    exit 1
fi

echo -e "${GREEN}开始传输数据...${NC}"
echo -e "${BLUE}源:${NC} $SOURCE"
echo -e "${BLUE}目标:${NC} $DEST"
echo -e "${BLUE}方法:${NC} $METHOD"
echo ""

# 根据方法选择传输工具
case $METHOD in
    scp)
        echo -e "${GREEN}使用 SCP 传输...${NC}"
        scp -r -o StrictHostKeyChecking=no "$SOURCE" "$DEST"
        ;;
    rsync)
        echo -e "${GREEN}使用 RSYNC 传输...${NC}"
        rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no" "$SOURCE" "$DEST"
        ;;
    *)
        echo -e "${RED}错误: 未知的方法 '$METHOD'${NC}"
        show_usage
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 传输完成！${NC}"
else
    echo ""
    echo -e "${RED}✗ 传输失败！${NC}"
    exit 1
fi
