import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = "../Performance_gap_ACN(14,6).xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 筛选数据：只保留 VEHICLE_GEN_PROB 在 0.3 到 0.9 之间的数据
df = df[(df['total_power'] >= 20) & (df['total_power'] <=40) & (df['penalty_weight'] == 0.8)].copy()

# 设置中文字体（可选，如果标签需要中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 提取横坐标和纵坐标数据
x_values = df['total_power']
gap_columns = df.columns[11:16]  # N到S列：index_gap 到 new_gap
labels = ['Index', 'Greedy', 'LLF', 'LRF', 'New']

# 创建折线图
plt.figure(figsize=(14, 8))

# 获取默认颜色循环
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i, col in enumerate(gap_columns):
    plt.plot(x_values, df[col] * 100, marker='o', label=labels[i], color=colors[i % len(colors)])

# 在 VEHICLE_GEN_PROB = 0.9 的位置添加水平虚线和数值
prob_09_data = df[df['total_power'] == 36]
if not prob_09_data.empty:
    for i, col in enumerate(gap_columns):
        gap_value = prob_09_data[col].iloc[0] * 100  # 获取gap值并转换为百分比
        plt.text(max(x_values) * 1.01, gap_value, f'{gap_value:.2f}', fontsize=12, va='center', color=colors[i % len(colors)])

# 设置图表标题和坐标轴标签（增大字体并更新标签）
plt.title('Varying Price & Penalty Weight = 0.8', fontsize=20)
plt.xlabel('Charging power', fontsize=18)
plt.ylabel('Performance Gap (%)', fontsize=18)
plt.xticks(x_values, fontsize=16)
plt.yticks(fontsize=16)

# 添加图例（增大字体）
plt.legend(title='Policy Types', loc='best', fontsize=16)

# 网格线
plt.grid(True, linestyle='--', alpha=0.8)

# 显示图表
plt.tight_layout()
plt.show()
