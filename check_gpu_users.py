#!/usr/bin/env python3
"""查询GPU 0-6的使用者用户名"""
import subprocess
import re

def get_gpu_users():
    """获取GPU使用者和用户名"""
    try:
        # 获取nvidia-smi输出
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_index,pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print("无法执行nvidia-smi命令")
            return
        
        lines = result.stdout.strip().split('\n')
        
        gpu_users = {}
        
        for line in lines:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                try:
                    gpu_idx = int(parts[0])
                    pid = int(parts[1])
                    
                    # 获取进程的用户名
                    try:
                        ps_result = subprocess.run(
                            ['ps', '-o', 'user=', '-p', str(pid)],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        username = ps_result.stdout.strip()
                        
                        if gpu_idx not in gpu_users:
                            gpu_users[gpu_idx] = []
                        gpu_users[gpu_idx].append({
                            'pid': pid,
                            'user': username,
                            'process': parts[2] if len(parts) > 2 else 'unknown',
                            'memory': parts[3] if len(parts) > 3 else 'unknown'
                        })
                    except:
                        pass
                except ValueError:
                    continue
        
        # 显示结果
        print("=" * 60)
        print("GPU 0-6 使用情况：")
        print("=" * 60)
        
        for gpu_idx in range(7):
            if gpu_idx in gpu_users:
                users = set([u['user'] for u in gpu_users[gpu_idx]])
                print(f"\nGPU {gpu_idx}:")
                print(f"  使用者: {', '.join(users)}")
                for app in gpu_users[gpu_idx]:
                    print(f"    - PID {app['pid']}: {app['user']} ({app['process']}, {app['memory']})")
            else:
                print(f"\nGPU {gpu_idx}: 空闲")
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print("错误：找不到nvidia-smi命令")
    except subprocess.TimeoutExpired:
        print("错误：nvidia-smi命令执行超时")
    except Exception as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    get_gpu_users()
