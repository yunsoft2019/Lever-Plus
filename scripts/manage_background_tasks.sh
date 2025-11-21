#!/bin/bash

# 管理后台任务脚本
# 用法: 
#   bash scripts/manage_background_tasks.sh list          # 列出所有运行中的任务
#   bash scripts/manage_background_tasks.sh stop <pid>     # 停止指定PID的任务
#   bash scripts/manage_background_tasks.sh stop-all      # 停止所有generate_data任务
#   bash scripts/manage_background_tasks.sh logs          # 查看所有日志文件

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$PROJECT_DIR/logs/generate_data"

action=${1:-list}

case "$action" in
    list)
        echo "=========================================="
        echo "📋 运行中的 generate_data 任务"
        echo "=========================================="
        # 查找所有generate_data.sh进程
        ps aux | grep "[g]enerate_data.sh" | while read line; do
            pid=$(echo $line | awk '{print $2}')
            cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            echo "PID: $pid"
            echo "命令: $cmd"
            echo "---"
        done
        
        # 列出所有PID文件
        if [ -d "$LOGS_DIR" ]; then
            echo ""
            echo "📁 PID文件列表:"
            find "$LOGS_DIR" -name "*.pid" 2>/dev/null | while read pidfile; do
                if [ -f "$pidfile" ]; then
                    pid=$(cat "$pidfile")
                    logfile="${pidfile%.pid}"
                    if ps -p "$pid" > /dev/null 2>&1; then
                        echo "✅ $logfile (PID: $pid) - 运行中"
                    else
                        echo "❌ $logfile (PID: $pid) - 已停止"
                    fi
                fi
            done
        fi
        ;;
    
    stop)
        pid=${2:-""}
        if [ -z "$pid" ]; then
            echo "❌ 请提供PID: bash scripts/manage_background_tasks.sh stop <pid>"
            exit 1
        fi
        
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "🛑 停止进程 $pid..."
            kill "$pid"
            sleep 2
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "⚠️  进程仍在运行，强制停止..."
                kill -9 "$pid"
            fi
            echo "✅ 进程已停止"
        else
            echo "⚠️  进程 $pid 不存在或已停止"
        fi
        ;;
    
    stop-all)
        echo "🛑 停止所有 generate_data 任务..."
        pids=$(ps aux | grep "[g]enerate_data.sh" | awk '{print $2}')
        if [ -z "$pids" ]; then
            echo "✅ 没有运行中的任务"
        else
            for pid in $pids; do
                echo "停止进程 $pid..."
                kill "$pid" 2>/dev/null
            done
            sleep 2
            # 强制停止仍在运行的进程
            pids=$(ps aux | grep "[g]enerate_data.sh" | awk '{print $2}')
            for pid in $pids; do
                kill -9 "$pid" 2>/dev/null
            done
            echo "✅ 所有任务已停止"
        fi
        ;;
    
    logs)
        echo "=========================================="
        echo "📁 日志文件列表"
        echo "=========================================="
        if [ -d "$LOGS_DIR" ]; then
            find "$LOGS_DIR" -name "*.log" -type f | sort | while read logfile; do
                size=$(du -h "$logfile" | cut -f1)
                lines=$(wc -l < "$logfile" 2>/dev/null || echo "0")
                echo "$logfile"
                echo "  大小: $size, 行数: $lines"
                echo "  查看: tail -f $logfile"
                echo "---"
            done
        else
            echo "日志目录不存在: $LOGS_DIR"
        fi
        ;;
    
    *)
        echo "用法:"
        echo "  bash scripts/manage_background_tasks.sh list          # 列出所有运行中的任务"
        echo "  bash scripts/manage_background_tasks.sh stop <pid>   # 停止指定PID的任务"
        echo "  bash scripts/manage_background_tasks.sh stop-all     # 停止所有generate_data任务"
        echo "  bash scripts/manage_background_tasks.sh logs         # 查看所有日志文件"
        ;;
esac



