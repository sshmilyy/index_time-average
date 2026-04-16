import pandas as pd

# 1. 读取数据
df = pd.read_csv('Raw_Data.csv')

# 2. 清洗数据
# 强制转为数字，处理非数字字符
df['EnergySupplied'] = pd.to_numeric(df['EnergySupplied'], errors='coerce')
df['PluginDuration'] = pd.to_numeric(df['PluginDuration'], errors='coerce')
df = df.dropna(subset=['EnergySupplied', 'PluginDuration']) # 删除空值

# 总样本数
total = len(df)


# 3. 统计特定阈值
count_0=len(df[df['PluginDuration'] == 0])
count_30 = len(df[df['PluginDuration'] > 0])-len(df[df['PluginDuration'] > 30])
count_60 = len(df[df['PluginDuration'] > 30])-len(df[df['PluginDuration'] > 60])
count_90 = len(df[df['PluginDuration'] > 60])-len(df[df['PluginDuration'] > 90])
count_120 = len(df[df['PluginDuration'] > 90])-len(df[df['PluginDuration'] > 120])
count_150 = len(df[df['PluginDuration'] > 120])-len(df[df['PluginDuration'] > 150])
count_180 = len(df[df['PluginDuration'] > 150])-len(df[df['PluginDuration'] > 180])
count_240 = len(df[df['PluginDuration'] > 180])-len(df[df['PluginDuration'] > 240])
count_320 = len(df[df['PluginDuration'] > 240])
# 4. 统计区间分布

# 5. 输出结果
print(f"=== 停留时长 (Plugin Duration) 分析结果 (总样本: {total}) ===")
print(f"0. 0 分钟: {count_0} 个 (占比 {count_0/total*100:.2f}%)")
print(f"1. 超过 0 少于 30 分钟: {count_30} 个 (占比 {count_30/total*100:.2f}%)")
print(f"2. 超过 30 少于 60 分钟 (1小时): {count_60} 个 (占比 {count_60/total*100:.2f}%)")
print(f"3. 超过 60 少于 90 分钟 (1.5小时): {count_90} 个 (占比 {count_90/total*100:.2f}%)")
print(f"4. 超过 90 少于 120 分钟 (2小时): {count_120} 个 (占比 {count_120/total*100:.2f}%)")
print(f"5. 超过 120 少于 150 分钟 (2.5小时): {count_150} 个 (占比 {count_150/total*100:.2f}%)")
print(f"6. 超过 150 少于 180 分钟 (3小时): {count_180} 个 (占比 {count_180/total*100:.2f}%)")
print(f"7. 超过 180 少于 240 分钟 (4小时): {count_240} 个 (占比 {count_240/total*100:.2f}%)")
print(f"8. 超过 240 分钟 (4小时): {count_320} 个 (占比 {count_320/total*100:.2f}%)")
print("\n=== 详细分布情况 ===")

