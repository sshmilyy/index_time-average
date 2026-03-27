import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取并预处理数据
# 请确保 Raw_Data.csv 在同一目录下
df = pd.read_csv('Raw_Data.csv')

# 合并日期时间
df['FullStartTime'] = pd.to_datetime(df['StartDate'] + ' ' + df['StartTime'])
df['t'] = df['FullStartTime'].dt.round('h').dt.hour  # 四舍五入取整点

# 2. 强制将 'PluginDuration' 转换为数字
df['PluginDuration'] = pd.to_numeric(df['PluginDuration'], errors='coerce')

# 3. 清洗数据
df = df.dropna(subset=['PluginDuration'])
# 过滤掉极端的异常值（保留 0 到 1440 分钟内的数据）
df = df[(df['PluginDuration'] > 0) & (df['PluginDuration'] < 1440)]

# =========================================================
# 计算关键统计量 (用于 Fig 1)
# =========================================================
# 1. 统计总天数 (用于计算 Lambda)
total_days = df['FullStartTime'].dt.date.nunique()
print(f"数据覆盖总天数: {total_days} 天")

# 2. 计算每小时的 Arrival Rate (Lambda)
# 统计每个小时出现的总次数 (0-23)
hourly_counts = df['t'].value_counts().sort_index()
# 补全可能缺失的小时 (比如凌晨3点没有车，要补0)
hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
# 计算 Lambda = 总次数 / 总天数
arrival_rates = hourly_counts / (total_days * 27)

# =========================================================
# 绘图设置
# =========================================================
sns.set(style="whitegrid", context="notebook")
fig = plt.figure(figsize=(20, 14))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# ---------------------------------------------------------
# Chart 1: Arrival Rate (Lambda)
# ---------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1)
# 使用 barplot 绘制计算好的 lambda 值
sns.barplot(x=arrival_rates.index, y=arrival_rates.values, color='#3498db', ax=ax1)

ax1.set_title(f'Fig 1: Arrival Rate', fontsize=16, fontweight='bold')
ax1.set_xlabel('Hour of Day (0-23)', fontsize=12)
ax1.set_ylabel('Arrival Rate (Vehicles/Hour)', fontsize=12)


# ---------------------------------------------------------
# Chart 2: Energy Distribution (Density)
# ---------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
# stat='density' 将纵坐标改为概率密度
sns.histplot(df['EnergySupplied'], bins=50, kde=True, stat='density', color='#e67e22', ax=ax2)

ax2.set_title('Fig 2: Energy Supplied Distribution', fontsize=16, fontweight='bold')
ax2.set_xlabel('Energy (kWh)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_xlim(0, 60)


# ---------------------------------------------------------
# Chart 3: Scatter (保持不变)
# ---------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
sns.scatterplot(x='PluginDuration', y='EnergySupplied', data=df, alpha=0.2, color='#2ecc71', ax=ax3)

ax3.set_title('Fig 3: Duration vs. Energy', fontsize=16, fontweight='bold')
ax3.set_xlabel('Duration (Minutes)', fontsize=12)
ax3.set_ylabel('Energy (kWh)', fontsize=12)
ax3.set_xlim(0, 150)
ax3.set_ylim(0, 60)


# ---------------------------------------------------------
# Chart 4: Plugin Duration (Density)
# ---------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
# stat='density' 将纵坐标改为概率密度
sns.histplot(df['PluginDuration'], bins=range(0, 120, 2), kde=True, stat='density', color='#9b59b6', ax=ax4)

ax4.set_title('Fig 4: Plugin Duration Distribution', fontsize=16, fontweight='bold')
ax4.set_xlabel('Duration (Minutes)', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_xlim(0, 120)


print("Generating plots...")
plt.show()

# 验证统计数据
print(f"Max Duration in data: {df['PluginDuration'].max()}")
print(f"Mean Duration: {df['PluginDuration'].mean():.2f}")