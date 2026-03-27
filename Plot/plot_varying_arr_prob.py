

import matplotlib.pyplot as plt
import numpy as np

# 您提供的原始函数
def get_time_varying_prob(t, period=24):
    """
    模拟住宅区/公共充电到达率 (Residential Profile)。
    特征：
    1. 深夜 (0-6点) 几乎为 0。
    2. 上午 (8-11点) 有一个小高峰 (外出办事)。
    3. 傍晚 (16-19点) 是主高峰 (下班回家)。
    """
    hour_of_day = t % 24

    # 使用两个高斯分布叠加来模拟到达率
    # 下班高峰：均值 17:30 (17.5), 标准差 2.0
    evening_peak = 0.8 * np.exp(-((hour_of_day - 17.5) ** 2) / (2 * 2.0 ** 2))

    # 上午小高峰：均值 09:00 (9.0), 标准差 1.5
    morning_peak = 0.3 * np.exp(-((hour_of_day - 9.0) ** 2) / (2 * 1.5 ** 2))

    # 基础噪声
    base_rate = 0.05

    prob = base_rate + evening_peak + morning_peak
    return max(0.0, min(1.0, prob))
# 生成时间点 (0到100)
t_values = np.linspace(0, 100, 1000)

# 计算每个时间点的概率值
prob_values = [get_time_varying_prob(t) for t in t_values]

# 创建图像
plt.figure(figsize=(15, 7))

# 绘制概率随时间变化的曲线
plt.plot(t_values, prob_values, label='generate probability')

# 添加图表标题、坐标轴标签和图例
plt.title('Vehicle Arrival Probability Over Time', fontsize=16)
plt.xlabel('time', fontsize=12)
plt.ylabel('probability', fontsize=12)
plt.xticks(np.arange(0, 101, 12))  # 每隔12个单位显示一个刻度
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# 显示图像
plt.show()