#    API_TOKEN = "PWOXeQ6svF8H-Jm-eiH2I9cD5T36gPKEtRvfX4_RYKc"
import json
import pytz
from datetime import datetime
from acnportal import acndata
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from dateutil import parser
import matplotlib.pyplot as plt

# ==========================================
# 配置部分
# ==========================================
API_TOKEN = "PWOXeQ6svF8H-Jm-eiH2I9cD5T36gPKEtRvfX4_RYKc"
SAVE_FILENAME = "acn_caltech_data.json"


def download_and_save():
    # ... (保持原本下载逻辑不变) ...
    site = "caltech"
    start_time = datetime(2018, 11, 1).replace(tzinfo=pytz.UTC)
    end_time = datetime(2019, 12, 1).replace(tzinfo=pytz.UTC)

    print(f"[1/3] 正在连接服务器，下载范围: {start_time.date()} 到 {end_time.date()}...")
    client = acndata.DataClient(API_TOKEN)

    try:
        sessions = list(client.get_sessions_by_time(site, start_time, end_time))
        print(f"[2/3] 下载成功！共获取 {len(sessions)} 条原始记录。")

        cleaned_data = []
        for s in sessions:
            try:
                if 'connectionTime' not in s or 'disconnectTime' not in s or 'kWhDelivered' not in s:
                    continue
                conn_time = s['connectionTime']
                disc_time = s['disconnectTime']
                energy = s['kWhDelivered']
                if conn_time is None or disc_time is None or energy is None:
                    continue
                record = {
                    "connectionTime": conn_time.isoformat(),
                    "disconnectTime": disc_time.isoformat(),
                    "kWhDelivered": float(energy)
                }
                cleaned_data.append(record)
            except Exception:
                continue

        print(f"[3/3] 正在保存 {len(cleaned_data)} 条有效数据到本地...")
        with open(SAVE_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2)
        print("成功保存。")

    except Exception as e:
        print(f"\n[失败] {e}")


def calculate_gmm_parameters():
    # 1. 读取数据
    if not os.path.exists(SAVE_FILENAME):
        print(f"[错误] 找不到文件 {SAVE_FILENAME}。请先运行下载部分。")
        return

    print(f"[1/4] 读取文件: {SAVE_FILENAME} ...")
    with open(SAVE_FILENAME, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 2. 特征提取
    feature_matrix = []
    discrete_arrival_hours = []
    unique_days = set()  # <--- [新增] 用于统计总天数

    for item in raw_data:
        try:
            t_conn = parser.parse(item['connectionTime'])
            t_disc = parser.parse(item['disconnectTime'])
            energy = item['kWhDelivered']

            arrival_hour = t_conn.hour + t_conn.minute / 60.0
            duration_hours = (t_disc - t_conn).total_seconds() / 3600.0

            # 清洗逻辑: 能量>0.5, 15min < 时长 < 24h
            if energy > 0.5 and 0.25 < duration_hours < 24:
                feature_matrix.append([arrival_hour, duration_hours, energy])

                # 离散化到达时间 (用于统计 Lambda)
                rounded_hour = int(round(arrival_hour))
                if rounded_hour == 24: rounded_hour = 0
                discrete_arrival_hours.append(rounded_hour)

                # 记录日期，用于计算总天数
                unique_days.add(t_conn.date())

        except Exception:
            continue

    X = np.array(feature_matrix)
    print(f"[2/4] 有效样本数: {len(X)} 条")

    # --- [新增] 计算 Arrival Rate (Lambda) ---
    total_days = len(unique_days)
    print(f"      统计天数范围: {total_days} 天")

    arrival_counts = np.bincount(discrete_arrival_hours, minlength=24)
    arrival_rates = arrival_counts / total_days  # Lambda = Count / Days

    # 打印 Lambda 以便复制
    print("\n" + "=" * 60)
    print("【到达率 Lambda (Vehicles/Hour)】")
    print("arrival_rates = " + np.array2string(arrival_rates, separator=', ',
                                               formatter={'float_kind': lambda x: f'{x:.4f}'}))
    print("=" * 60)

    # 3. GMM 拟合 (保持不变)
    print(f"[3/4] 正在运行 EM 算法拟合双峰模型...")
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X)

    means = gmm.means_
    weights = gmm.weights_
    covariances = gmm.covariances_

    # 排序：Cluster 1 为短停，Cluster 2 为长停
    sort_idx = np.argsort(means[:, 1])
    means = means[sort_idx]
    weights = weights[sort_idx]
    covariances = covariances[sort_idx]
    stds = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))

    # 打印参数 (保持不变)
    print("\n" + "=" * 60)
    print("【GMM 参数计算完成】")
    print(f"# Cluster 1 (Short): w={weights[0]:.2f}, E={means[0][2]:.2f}, T={means[0][1]:.2f}")
    print(f"# Cluster 2 (Long) : w={weights[1]:.2f}, E={means[1][2]:.2f}, T={means[1][1]:.2f}")
    print("=" * 60)

    # 画图：传入计算好的 arrival_rates
    plot_gmm_results(X, means, stds, weights, arrival_rates, total_days)


def plot_gmm_results(X, means, stds, weights, arrival_rates, total_days):
    """
    画图验证拟合效果
    Fig 3 更改为 Arrival Rate (Lambda)
    """
    plt.figure(figsize=(16, 12))

    # =========================================
    # 图 3: 到达率分布 (Arrival Rate) - 左上 (Pos 1)
    # =========================================
    plt.subplot(2, 2, 1)

    hours = np.arange(24)
    # 画柱状图，显示 Lambda
    plt.bar(hours, arrival_rates, color='skyblue', label='Arrival Rate ($\lambda$)')

    plt.title(f'Fig 1: Arrival Rate')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Vehicles per Hour ($\lambda$)')  # Y轴含义变更
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # =========================================
    # 图 1: 能量分布 (Energy Distribution) - 右上 (Pos 2)
    # =========================================
    plt.subplot(2, 2, 2)
    plt.hist(X[:, 2], bins=50, density=False, alpha=0.5, color='gray', label='Real Data')

    x_axis = np.linspace(0, 80, 1000)
    y1 = weights[0] * (1 / (stds[0][2] * np.sqrt(2 * np.pi))) * np.exp(
        - (x_axis - means[0][2]) ** 2 / (2 * stds[0][2] ** 2))
    y2 = weights[1] * (1 / (stds[1][2] * np.sqrt(2 * np.pi))) * np.exp(
        - (x_axis - means[1][2]) ** 2 / (2 * stds[1][2] ** 2))

    plt.plot(x_axis, y1, '--', label='Cluster 1 (Short)')
    plt.plot(x_axis, y2, '--', label='Cluster 2 (Long)')
    plt.plot(x_axis, y1 + y2, 'r-', lw=2, label='Total Fit')

    plt.title('Fig 2: Energy Distribution (GMM Fit)')
    plt.xlabel('Energy (kWh)')
    plt.legend()

    # =========================================
    # 图 4: 时长与能量的相关性 - 左下 (Pos 3)
    # =========================================
    plt.subplot(2, 2, 3)
    # 使用散点图
    plt.scatter(X[:, 1], X[:, 2], alpha=0.3, s=15, c='mediumseagreen', edgecolors='none')

    plt.title('Fig 3: Duration vs. Energy ')
    plt.xlabel('Stay Duration (Hours)')
    plt.ylabel('Energy Delivered (kWh)')
    plt.xlim(0, 24)
    plt.ylim(0, 80)
    plt.grid(True, linestyle='--', alpha=0.5)

    # =========================================
    # 图 2: 时长分布 (Duration Distribution) - 右下 (Pos 4)
    # =========================================
    plt.subplot(2, 2, 4)
    plt.hist(X[:, 1], bins=50, density=False, alpha=0.5, color='gray', label='Real Data')
    x_axis_dur = np.linspace(0, 24, 1000)

    y1_d = weights[0] * (1 / (stds[0][1] * np.sqrt(2 * np.pi))) * np.exp(
        - (x_axis_dur - means[0][1]) ** 2 / (2 * stds[0][1] ** 2))
    y2_d = weights[1] * (1 / (stds[1][1] * np.sqrt(2 * np.pi))) * np.exp(
        - (x_axis_dur - means[1][1]) ** 2 / (2 * stds[1][1] ** 2))

    plt.plot(x_axis_dur, y1 + y2_d, 'r-', lw=2, label='Total Fit')
    plt.title('Fig 4: Stay Duration Distribution')
    plt.xlabel('Duration (Hours)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 如果没有数据文件，打开下面这行的注释先下载
    # download_and_save()

    calculate_gmm_parameters()