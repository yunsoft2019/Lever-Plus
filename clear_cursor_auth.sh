#!/bin/bash

# Cursor 认证信息清除脚本
# 此脚本将清除 Cursor 的所有认证信息，允许您重新登录

echo "正在清除 Cursor 的认证信息..."

# 1. 清除 ~/.config/Cursor 中的认证相关文件
echo "1. 清除 ~/.config/Cursor 配置..."
if [ -d ~/.config/Cursor ]; then
    # 备份设置文件（如果需要保留其他设置）
    if [ -f ~/.config/Cursor/User/settings.json ]; then
        cp ~/.config/Cursor/User/settings.json ~/.config/Cursor/User/settings.json.backup
    fi
    
    # 查找并清除可能包含认证信息的文件
    find ~/.config/Cursor -type f \( -name "*token*" -o -name "*auth*" -o -name "*credential*" -o -name "*session*" -o -name "*user*" \) -delete 2>/dev/null
    
    # 清除 Storage 目录（如果存在，通常包含认证信息）
    if [ -d ~/.config/Cursor/User/workspaceStorage ]; then
        echo "   清除 workspaceStorage..."
        rm -rf ~/.config/Cursor/User/workspaceStorage/*
    fi
    
    if [ -d ~/.config/Cursor/User/globalStorage ]; then
        echo "   清除 globalStorage..."
        rm -rf ~/.config/Cursor/User/globalStorage/*
    fi
fi

# 2. 清除 ~/.cursor 目录中的认证信息
echo "2. 清除 ~/.cursor 目录..."
if [ -d ~/.cursor ]; then
    # 备份 ide_state.json（如果需要）
    if [ -f ~/.cursor/ide_state.json ]; then
        cp ~/.cursor/ide_state.json ~/.cursor/ide_state.json.backup
    fi
    
    # 查找并清除认证相关文件
    find ~/.cursor -type f \( -name "*token*" -o -name "*auth*" -o -name "*credential*" -o -name "*session*" \) -delete 2>/dev/null
fi

# 3. 使用 Python keyring 清除认证令牌
echo "3. 清除密钥环中的认证信息..."
python3 << 'PYTHON_SCRIPT'
import keyring
import sys

# 常见的 Cursor 密钥环服务名称
services_to_clear = [
    'cursor',
    'Cursor',
    'com.cursor',
    'com.cursor.Cursor',
    'vscode',
    'code',
]

cleared = False
for service in services_to_clear:
    try:
        # 尝试获取并删除
        password = keyring.get_password(service, 'account')
        if password:
            keyring.delete_password(service, 'account')
            print(f"   已清除密钥环: {service}/account")
            cleared = True
    except:
        pass
    
    try:
        password = keyring.get_password(service, 'token')
        if password:
            keyring.delete_password(service, 'token')
            print(f"   已清除密钥环: {service}/token")
            cleared = True
    except:
        pass

if not cleared:
    print("   未在密钥环中找到 Cursor 相关条目（可能已清除或使用其他存储方式）")
PYTHON_SCRIPT

# 4. 清除可能的缓存
echo "4. 清除缓存..."
if [ -d ~/.cache/Cursor ]; then
    rm -rf ~/.cache/Cursor/*
    echo "   已清除 ~/.cache/Cursor"
fi

# 5. 清除本地存储（如果存在）
echo "5. 清除本地存储..."
if [ -d ~/.local/share/Cursor ]; then
    rm -rf ~/.local/share/Cursor/*
    echo "   已清除 ~/.local/share/Cursor"
fi

echo ""
echo "✅ 清除完成！"
echo ""
echo "请执行以下操作："
echo "1. 完全关闭 Cursor（确保所有窗口都已关闭）"
echo "2. 重新启动 Cursor"
echo "3. 使用新账号登录"
echo ""
echo "注意：已创建备份文件（.backup），如需恢复可以手动恢复。"


