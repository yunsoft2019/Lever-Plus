# 清除 Cursor 账号信息指南

## 已执行的清除操作

脚本 `clear_cursor_auth.sh` 已经清除了以下位置：
- `~/.config/Cursor` 中的认证相关文件
- `~/.cursor` 中的认证相关文件
- 密钥环中的认证令牌
- 缓存文件

## 如果仍然提示旧账号，请手动执行以下步骤：

### 方法 1：完全清除 Cursor 配置（最彻底）

```bash
# 1. 完全关闭 Cursor（确保所有窗口都已关闭）

# 2. 备份当前配置（可选）
mkdir -p ~/cursor_backup
cp -r ~/.config/Cursor ~/cursor_backup/ 2>/dev/null
cp -r ~/.cursor ~/cursor_backup/ 2>/dev/null

# 3. 删除 Cursor 配置目录
rm -rf ~/.config/Cursor
rm -rf ~/.cursor

# 4. 清除缓存
rm -rf ~/.cache/Cursor 2>/dev/null
rm -rf ~/.local/share/Cursor 2>/dev/null

# 5. 重新启动 Cursor
```

### 方法 2：仅清除认证信息（保留其他设置）

```bash
# 1. 关闭 Cursor

# 2. 删除可能包含认证信息的目录
rm -rf ~/.config/Cursor/User/workspaceStorage 2>/dev/null
rm -rf ~/.config/Cursor/User/globalStorage 2>/dev/null
rm -rf ~/.config/Cursor/User/History 2>/dev/null

# 3. 查找并删除认证相关文件
find ~/.config/Cursor -type f \( -name "*token*" -o -name "*auth*" -o -name "*credential*" -o -name "*session*" \) -delete 2>/dev/null
find ~/.cursor -type f \( -name "*token*" -o -name "*auth*" -o -name "*credential*" -o -name "*session*" \) -delete 2>/dev/null

# 4. 清除密钥环
python3 << 'EOF'
import keyring
services = ['cursor', 'Cursor', 'com.cursor', 'com.cursor.Cursor', 'vscode', 'code']
for service in services:
    for username in ['account', 'token', 'user', 'email', 'auth']:
        try:
            keyring.delete_password(service, username)
        except:
            pass
EOF

# 5. 重新启动 Cursor
```

### 方法 3：使用 Cursor 内置的登出功能

如果 Cursor 有内置的登出功能：
1. 打开 Cursor
2. 进入设置（Settings）
3. 查找"账户"（Account）或"认证"（Authentication）选项
4. 点击"登出"（Sign Out）或"注销"（Logout）

### 方法 4：检查浏览器存储（如果 Cursor 使用 WebView）

```bash
# Cursor 可能使用 Chromium 的存储
rm -rf ~/.config/Cursor/Default/Local\ Storage 2>/dev/null
rm -rf ~/.config/Cursor/Default/Session\ Storage 2>/dev/null
rm -rf ~/.config/Cursor/Default/IndexedDB 2>/dev/null
```

## 验证清除是否成功

清除后，重新启动 Cursor，应该会提示您登录新账号，而不是显示旧账号信息。

## 注意事项

- 清除配置会删除所有 Cursor 设置，包括主题、快捷键等
- 如果只想清除账号信息，使用方法 2
- 建议先备份重要配置（如 settings.json）

