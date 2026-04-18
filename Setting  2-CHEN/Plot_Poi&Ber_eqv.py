import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path  # <--- 新增




# ==========================================
# 1. 读取已经跑好的 Excel 数据
# ==========================================
# 确保你的 Excel 文件名和之前代码导出的一致
EXCEL_DIR = Path("results_excel")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
file_path = EXCEL_DIR / "Poisson_Bernoulli_Equivalence.xlsx" # <--- 修改这里

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"❌ 找不到文件 {file_path}，请确认仿真代码已经跑完并生成了该表格！")
    exit()

# ==========================================
# 2. 论文级图表样式设置设置
# ==========================================
# 开启类似 LaTeX / 学术论文的字体风格 (如果报错可以把这行注释掉)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig, ax = plt.subplots(figsize=(10, 7))

# 为每个策略定制和参考图一样的标记(Marker)与颜色
# 参考图风格：红色的空心左三角、蓝色的星星、黑色的线等
styles = {
    'INDEX': {'color': 'red', 'marker': '<', 'label': "Whittle's index"},
    'LLF': {'color': 'blue', 'marker': '*', 'label': 'LLF'},
    'LRF': {'color': 'black', 'marker': 'o', 'label': 'LRF'},
    'GDY': {'color': '#00A000', 'marker': 's', 'label': 'GDY'},  # 深绿色
    'NEW': {'color': 'magenta', 'marker': 'd', 'label': 'NEW Policy'}
}

# ==========================================
# 3. 循环作图
# ==========================================
# 提取所有的策略和 N 值
policies = df['Policy'].unique()
n_values = df['System Scale (N)'].unique()

for policy in policies:
    # 筛选当前策略的数据，并确保按 N 排序
    subset = df[df['Policy'] == policy].sort_values(by='System Scale (N)')

    x = subset['System Scale (N)']
    # 如果你想画总收益(Total Reward)，可以用 y * x。这里默认画平均收益
    y_poi = subset['Poisson Reward']
    y_ber = subset['Bernoulli Reward']

    st = styles.get(policy, {'color': 'gray', 'marker': 'x', 'label': policy})

    # 画 Poisson 到达下的实线 (Solid line)
    ax.plot(x, y_poi,
            color=st['color'],
            marker=st['marker'],
            linestyle='-',
            linewidth=2,
            markersize=10,
            markerfacecolor='none',  # 空心标记 (关键！)
            markeredgewidth=1.5,
            label=f"{st['label']} w. Poisson arr.")

    # 画 Bernoulli 到达下的虚线 (Dashed line)
    ax.plot(x, y_ber,
            color=st['color'],
            marker=st['marker'],
            linestyle='--',
            linewidth=2,
            markersize=10,
            markerfacecolor='none',  # 空心标记 (关键！)
            markeredgewidth=1.5,
            label=f"{st['label']} w. Bernoulli arr.")

# ==========================================
# 4. 坐标轴、网格与图例美化
# ==========================================
# 设置网格，模仿 MATLAB 风格的灰色实线网格
ax.grid(True, linestyle='-', color='lightgray', alpha=0.8)

# 坐标轴刻度与标签
ax.set_xticks(n_values)  # 强制 x 轴只显示你测过的 N 值 (例如 10, 50, 100, 300)
ax.tick_params(axis='both', which='major', labelsize=12)

ax.set_xlabel('System Size (N)', fontsize=14, fontweight='bold')
# 注意：原图 Y 轴叫 Total Reward，根据你的需求修改此处文字
ax.set_ylabel('Total Reward', fontsize=14, fontweight='bold')

# 图例设置：放置在图表内部，带有黑色边框，紧凑排列
ax.legend(fontsize=11,
          loc='best',  # 自动寻找不遮挡线条的最佳位置
          frameon=True,
          edgecolor='black',
          framealpha=1.0)

plt.tight_layout()

# 将图片存入 PLOT_DIR
pdf_path = PLOT_DIR / "Arrival_Equivalence_Plot.pdf"
png_path = PLOT_DIR / "Arrival_Equivalence_Plot.png"

plt.savefig(pdf_path, format='pdf', dpi=300)
plt.savefig(png_path, format='png', dpi=300)

print(f"✅ 画图完成！图表已保存至: {PLOT_DIR} 文件夹中")
plt.show()