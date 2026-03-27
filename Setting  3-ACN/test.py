import json
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_FILENAME = "acn_caltech_data.json"


def calculate_detailed_distributions():
    # 1. 读取数据
    if not os.path.exists(DATA_FILENAME):
        print(f"[错误] 找不到文件 {DATA_FILENAME}")
        return

    print(f"正在读取 {DATA_FILENAME} 并计算详细分布...\n")
    with open(DATA_FILENAME, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 2. 提取列表
    durations = []
    energies = []

    for item in raw_data:
        try:
            # 简单的清洗逻辑，与之前保持一致
            if item['kWhDelivered'] > 0.5:
                # 计算时长 (小时)
                t_conn = item['connectionTime']
                t_disc = item['disconnectTime']
                # 注意：这里假设您已经把 ISO 字符串解析或者直接计算差值
                # 为了通用性，我们重新解析一下（如果原始数据是字符串）
                from dateutil import parser
                t1 = parser.parse(t_conn)
                t2 = parser.parse(t_disc)
                dur_h = (t2 - t1).total_seconds() / 3600.0

                if 0.25 < dur_h < 48:  # 稍微放宽一点上限查看长尾
                    durations.append(dur_h)
                    energies.append(item['kWhDelivered'])
        except:
            continue

    total_samples = len(durations)
    print(f"有效样本总数: {total_samples}")

    # ==========================================
    # 3. 计算时长分布 (Duration Distribution)
    # ==========================================
    # 设定分箱：0-1h, 1-2h, ..., 23-24h, >24h
    max_duration_bin = 24
    # 使用 np.floor 向下取整，例如 1.5h -> 1 (代表 1-2h 区间)
    duration_bins = np.floor(durations).astype(int)
    # 截断超过 24 小时的
    duration_bins = np.clip(duration_bins, 0, max_duration_bin)

    d_counts = np.bincount(duration_bins, minlength=max_duration_bin + 1)
    d_probs = d_counts / total_samples

    print("\n" + "=" * 50)
    print("【停留时长分布】(Duration Distribution)")
    print(f"分箱单位: 1 Hour (例如 '2' 代表 2.0h <= t < 3.0h)")
    print("-" * 50)
    print(f"{'区间 (Hours)':<15} | {'计数 (Count)':<10} | {'概率 (Prob)':<10} | {'累积 (CDF)':<10}")
    print("-" * 50)

    cdf = 0
    for i in range(len(d_probs)):
        prob = d_probs[i]
        cdf += prob
        if prob > 0.0001:  # 只打印有数据的行，避免刷屏
            label = f"{i}-{i + 1} h" if i < max_duration_bin else f"> {max_duration_bin} h"
            print(f"{label:<15} | {d_counts[i]:<10d} | {prob:.4f}     | {cdf:.4f}")

    # ==========================================
    # 4. 计算能量分布 (Energy Distribution)
    # ==========================================
    # 设定分箱：每 2 kWh 一个区间
    energy_bin_size = 2.0
    max_energy_kwh = 60.0

    # 离散化：例如 3.5 kWh / 2 = 1.75 -> 1 (代表 2-4 kWh)
    e_indices = np.floor(np.array(energies) / energy_bin_size).astype(int)
    max_idx = int(max_energy_kwh / energy_bin_size)
    e_indices = np.clip(e_indices, 0, max_idx)

    e_counts = np.bincount(e_indices, minlength=max_idx + 1)
    e_probs = e_counts / total_samples

    print("\n" + "=" * 50)
    print("【充电能量分布】(Energy Distribution)")
    print(f"分箱单位: {energy_bin_size} kWh")
    print("-" * 50)
    print(f"{'区间 (kWh)':<15} | {'计数 (Count)':<10} | {'概率 (Prob)':<10} | {'累积 (CDF)':<10}")
    print("-" * 50)

    cdf_e = 0
    for i in range(len(e_probs)):
        prob = e_probs[i]
        cdf_e += prob
        if prob > 0.0001:
            start_e = i * energy_bin_size
            end_e = (i + 1) * energy_bin_size
            label = f"{start_e:g}-{end_e:g} kWh" if i < max_idx else f"> {max_energy_kwh} kWh"
            print(f"{label:<15} | {e_counts[i]:<10d} | {prob:.4f}     | {cdf_e:.4f}")

    # ==========================================
    # 5. 提供可复制的数组
    # ==========================================
    print("\n" + "=" * 50)
    print("【可复制数组】(Copy & Paste Arrays)")
    print("Duration Probs (0h, 1h, ... 24h+):")
    print(np.array2string(d_probs, separator=', ', formatter={'float_kind': lambda x: f"{x:.4f}"}))
    print(f"\nEnergy Probs (Step={energy_bin_size}kWh):")
    print(np.array2string(e_probs, separator=', ', formatter={'float_kind': lambda x: f"{x:.4f}"}))


if __name__ == "__main__":
    calculate_detailed_distributions()