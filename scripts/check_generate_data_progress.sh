#!/bin/bash
# 查看数据生成进度

python3 << 'EOF'
import re
import os

logs = {
    "rand_sampler (GPU 1)": "logs/generate_data_vqa_vqav2_20251222_192348.log",
    "text_sim_sampler (GPU 2)": "logs/generate_data_vqa_vqav2_textsim_20251222_192541.log",
    "img_sim_sampler (GPU 3)": "logs/generate_data_vqa_vqav2_imgsim_20251222_192705.log",
    "mix_sampler (GPU 4)": "logs/generate_data_vqa_vqav2_mix_20251222_192817.log"
}

print("=" * 80)
print("数据生成进度")
print("=" * 80)

for name, log_file in logs.items():
    if not os.path.exists(log_file):
        print(f"\n{name}: 日志文件不存在")
        continue
        
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 提取剩余样本数
        remaining_match = re.search(r'processing (\d+) remaining samples', content)
        skip_match = re.search(r'skipping (\d+) already processed', content)
        
        if remaining_match and skip_match:
            remaining = int(remaining_match.group(1))
            skipped = int(skip_match.group(1))
            total = remaining + skipped
            current = skipped
            
            # 提取最新进度条
            # 格式1: 2599it [1:56:31, 166.07s/it]
            it_match = re.findall(r'(\d+)it\s+\[([^\]]+)\]', content)
            if it_match:
                latest_it = it_match[-1]
                current = int(latest_it[0])
                time_info = latest_it[1]
                # 提取速度
                speed_match = re.search(r'(\d+\.?\d*)s/it', time_info)
                speed = float(speed_match.group(1)) if speed_match else None
            else:
                # 格式2: 84%|...| 2302/2740 [1:52:08<19:12:33, 157.52s/it]
                pct_match = re.findall(r'(\d+)%\|[^\|]+\|\s+(\d+)/(\d+)\s+\[([^\]]+)\]', content)
                if pct_match:
                    latest_pct = pct_match[-1]
                    current = int(latest_pct[1])
                    total = int(latest_pct[2])
                    time_info = latest_pct[3]
                    speed_match = re.search(r'(\d+\.?\d*)s/it', time_info)
                    speed = float(speed_match.group(1)) if speed_match else None
                else:
                    time_info = ""
                    speed = None
            
            remaining = total - current
            percent = (current / total * 100) if total > 0 else 0
            
            print(f"\n{name}:")
            print(f"  进度: {current}/{total} ({percent:.1f}%)")
            print(f"  剩余: {remaining} 个样本")
            
            if speed:
                remaining_time_sec = remaining * speed
                hours = int(remaining_time_sec // 3600)
                minutes = int((remaining_time_sec % 3600) // 60)
                if hours > 0:
                    print(f"  预计剩余时间: {hours}小时{minutes}分钟")
                else:
                    print(f"  预计剩余时间: {minutes}分钟")
                print(f"  当前速度: {speed:.1f}秒/样本")
        else:
            print(f"\n{name}: 无法提取进度信息")
            
    except Exception as e:
        print(f"\n{name}: 读取失败 - {e}")

print("\n" + "=" * 80)
EOF



