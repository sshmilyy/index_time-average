import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('RAW_data.csv')


# 2. 数据处理：提取时刻 (0-24)
df['FullStartTime'] = pd.to_datetime(df['StartDate'] + ' ' + df['StartTime'])
# 转换为连续的小时数 (例如 8.5 表示 8:30)
df['DecimalHour'] = df['FullStartTime'].dt.hour + df['FullStartTime'].dt.minute / 60.0

# 3. 统计每小时的总到达数 (Total Counts per Hour)
# 将一天分为 24 个时间窗
bins = np.arange(0, 25, 1) # [0, 1, 2, ... 24]
total_counts, bin_edges = np.histogram(df['DecimalHour'], bins=bins)
x_hours = (bin_edges[:-1] + bin_edges[1:]) / 2  # 取中间点: 0.5, 1.5, ... 23.5

# =========================================================
# 4. 核心计算：归一化 (Normalize) 获取 Lambda
# =========================================================
# 您的公式：总数 / (365天 * 27个站点)
NUM_DAYS = 365
NUM_STATIONS = 27

lambda_values = total_counts / (NUM_DAYS * NUM_STATIONS)
print(lambda_values)