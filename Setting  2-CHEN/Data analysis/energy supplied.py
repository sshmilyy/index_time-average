import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv('Raw_Data.csv')


# 2. 数据清洗 (针对 EnergySupplied)
# 强制转为数字，处理空值
df['EnergySupplied'] = pd.to_numeric(df['EnergySupplied'], errors='coerce')
df = df.dropna(subset=['EnergySupplied'])
# 过滤掉负数和极端异常值 (2017年单次快充很难超过100度电)
df = df[(df['EnergySupplied'] > 0) & (df['EnergySupplied'] <= 120)]

# 处理时间
df['FullStartTime'] = pd.to_datetime(df['StartDate'] + ' ' + df['StartTime'])
df['Hour'] = df['FullStartTime'].dt.round('h').dt.hour

# 3. 创建电量分类标签 (Energy Categories)
# 分组逻辑：小额补电 / 标准补电 / 大额补电 / 深度充电
bins = [0, 10, 20, 30, 50, 120]
labels = ['0-10 kWh', '10-20 kWh', '20-30 kWh', '30-50 kWh', '> 50 kWh']
df['EnergyCategory'] = pd.cut(df['EnergySupplied'], bins=bins, labels=labels)

# 4. 绘图设置
sns.set(style="whitegrid", context="notebook")
fig = plt.figure(figsize=(18, 8))
plt.subplots_adjust(wspace=0.3)

# --- 图1: 电量类别分布 (Bar Chart) ---
ax1 = fig.add_subplot(1, 2, 1)
# 计算各组数量
category_counts = df['EnergyCategory'].value_counts().sort_index()
# 绘制柱状图 - 使用橙色系
colors = ['#f39c12', '#e67e22', '#d35400', '#c0392b', '#8e44ad']
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors, ax=ax1)

# 标数值
for i, v in enumerate(category_counts.values):
    ax1.text(i, v + 100, str(v), ha='center', fontweight='bold', fontsize=12)

ax1.set_title('Fig 1: Count of Sessions by Energy Group', fontsize=14, fontweight='bold')
ax1.set_xlabel('Energy Supplied Group (kWh)', fontsize=12)
ax1.set_ylabel('Number of Sessions', fontsize=12)


print("Generating Energy Analysis Charts...")
plt.show()

# 打印详细数据
print("=== 详细分布数据 ===")
print(category_counts)
print("\n=== 占比分析 ===")
total = len(df)
for label, count in category_counts.items():
    print(f"{label}: {count/total*100:.2f}%")