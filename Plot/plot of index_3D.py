import numpy as np
import matplotlib.pyplot as plt

# 1. 定义 40 个 (r, l) 坐标点
# 第 1 组：(0, 0), (0, 1), (0, 2), (0, 3) (共 4 个点)
r_0 = [0, 0, 0, 0]
l_0 = [0, 1, 2, 3]

# 第 2 组：r in [1, 3], l in [1, 12] (共 36 个点)
# R 坐标：[1]重复12次, [2]重复12次, [3]重复12次 (长度 36)
r_rest = np.repeat([1, 2, 3], 12).tolist()
# L 坐标：[1..12]重复3次 (长度 36)
l_rest = np.tile(np.arange(1, 13), 3).tolist()

# 结合 R 和 L 坐标轴
R_coords = np.array(r_0 + r_rest)
L_coords = np.array(l_0 + l_rest)

# 2. 加载数据
# 假设文件 'whittle_index_multimode_time_varying ePrice_penal=0.2.npy' 已被正确上传和访问
try:
    data = np.load('whittle_index_multimode_time_varying ePrice_penal=0.2.npy')
except FileNotFoundError:
    print("错误：文件未找到。请确保文件路径正确或已上传到当前环境。")
    # 如果加载失败，停止执行
    exit()

# 3. 提取 Z 轴数据 (固定 t 索引 2, action 索引 3)
# t 索引 2 对应 t=3, action 索引 3 对应 action=3
Z_values = data[:, 2, 3]

# 3. 绘制三维散点图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 使用 Whittle Index 值作为颜色映射
scatter = ax.scatter(R_coords, L_coords, Z_values, c=Z_values, cmap='viridis', s=50)

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Whittle Index Value (Z)')

# 设置轴标签
ax.set_xlabel('r (First State Component)')
ax.set_ylabel('l (Second State Component)')
ax.set_zlabel('Whittle Index W(r, l, t=3, action=3)')
ax.set_title('3D Scatter Plot of Whittle Index vs. State (r, l)')

# 设置刻度以清晰显示离散值
ax.set_xticks(np.unique(R_coords))
ax.set_yticks(np.unique(L_coords))
ax.view_init(elev=20, azim=45)
plt.show()