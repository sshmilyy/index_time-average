import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('Raw_Data.csv')

# 读取上传的文件

# 1. 合并日期和时间列，转换为 datetime 对象
df['FullStartTime'] = pd.to_datetime(df['StartDate'] + ' ' + df['StartTime'])

# 2. 四舍五入到最近的小时 ('H'表示 Hour)
# 例如: 5:01 -> 5:00, 5:59 -> 6:00
df['RoundedStartTime'] = df['FullStartTime'].dt.round('h')
df['t'] = df['RoundedStartTime'].dt.hour


df['PluginDuration'] = pd.to_numeric(df['PluginDuration'], errors='coerce')
df['EnergySupplied'] = pd.to_numeric(df['EnergySupplied'], errors='coerce')
df = df.dropna(subset=['PluginDuration', 'EnergySupplied'])
df = df[(df['PluginDuration'] > 0) & (df['EnergySupplied'] > 0)]
# C. 过滤掉大于 90 分钟的数据 (根据您的新要求)
original_count = len(df)
df = df[df['PluginDuration'] <= 120]
df['L'] = np.ceil(df['PluginDuration'] / 15).astype(int)

df = df[df['EnergySupplied'] <= 30]
df['R'] = np.ceil(df['EnergySupplied'] / 2).astype(int)

output_columns = ['t', 'PluginDuration', 'L', 'EnergySupplied', 'R']
final_count = len(df)

final_df = df[output_columns]

# ==========================================
# 4. 生成分布表 (Distribution Tables)
# ==========================================

# --- L 的分布 ---
l_counts = df['L'].value_counts().sort_index()
l_probs = df['L'].value_counts(normalize=True).sort_index()
print(f"l_probs: {l_probs}")
l_table = pd.DataFrame({
    'L (Slots)': l_counts.index,
    'Time Range': [f"{(i-1)*15}-{i*15} min" for i in l_counts.index],
    'Count': l_counts.values,
    'Probability': l_probs.values
})

# --- E 的分布 ---
r_counts = df['R'].value_counts().sort_index()
r_probs = df['R'].value_counts(normalize=True).sort_index()
print(f"r_probs: {r_probs}")
r_table = pd.DataFrame({
    'E (Units)': r_counts.index,
    'Energy Range': [f"{(i-1)*2}-{i*2} kWh" for i in r_counts.index],
    'Count': r_counts.values,
    'Probability': r_probs.values
})

# ==========================================
# 5. 输出具体数值 (供您直接复制使用)
# ==========================================
print("\n" + "="*40)
print("表 1: 停留时长分布 (Distribution of L)")
print("="*40)
print(l_table.to_string(index=False, formatters={'Probability': '{:.4f}'.format}))

print("\n" + "="*40)
print("表 2: 电量需求分布 (Distribution of E)")
print("="*40)
print(r_table.to_string(index=False, formatters={'Probability': '{:.4f}'.format}))
print(f"数据清洗完成: 原始 {original_count} 条 -> 过滤后 {final_count} 条")


print("\nL distribution array:")
print(l_probs.values)
print("\nR distribution array:")
print(r_probs.values)

print("\n" + "="*40)
print("表 1: 停留时长分布 (Distribution of L)")
print("="*40)
# 5. 保存为 CSV 文件
output_filename = 'processed_ev_data.csv'
final_df.to_csv(output_filename, index=False)
