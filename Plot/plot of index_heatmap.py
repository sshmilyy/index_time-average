import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定义坐标轴和状态映射 (排除 r=0) ---
action_index = 3
T_len = 24
R_len = 3  # r: 1, 2, 3
L_len = 13  # l: 0, 1, ..., 12 (但 l=0 时 r=1, 2, 3 状态无效)

# Y 轴 (t 和 r 叠加) 的总长度: T_len * R_len = 24 * 3 = 72
Y_total_len = T_len * R_len

# --- 2. 加载数据并筛选 (只保留 r=1, 2, 3 的状态) ---
file_name = "whittle_index_multimode_time_varying ePrice_penal=0.2.npy"
try:
    data_raw = np.load(file_name)

    # 原始的 40 个状态索引 (r, l)
    r_coords_all = np.array([0, 0, 0, 0] + np.repeat([1, 2, 3], 12).tolist())
    l_coords_all = np.array([0, 1, 2, 3] + np.tile(np.arange(1, 13), 3).tolist())

    # 筛选出 r >= 1 的状态
    r_filter = (r_coords_all >= 1)

    # 筛选后的数据: 形状 (36, 24, 6)
    data_filtered = data_raw[r_filter, :, :]

    # 提取固定 action 3 的 W 值矩阵: 形状 (36, 24)
    W_values_filtered = data_filtered[:, :, action_index]

except Exception as e:
    print(f"错误: 无法加载或处理数据: {e}")
    exit()

# --- 3. 构造 72 x 13 的热图矩阵 ---
heatmap_matrix = np.full((Y_total_len, L_len), np.nan)

# R 实际值 (1, 2, 3)
r_values = np.array([1, 2, 3])
l_values = np.arange(L_len)  # 0 to 12

# 筛选后的 36 个状态索引 (r, l)
R_coords_filtered = r_coords_all[r_filter]
L_coords_filtered = l_coords_all[r_filter]

# 创建一个从 (r, l) 到 data_filtered 索引的查找表 (新索引 0 到 35)
state_to_idx = {(r, l): i for i, (r, l) in enumerate(zip(R_coords_filtered, L_coords_filtered))}

# 循环遍历所有 t 和 r 的组合，填充 72 x 13 矩阵
for t_idx in range(T_len):  # t_idx from 0 to 23
    for r_idx in range(R_len):  # r_idx from 0 to 2 (对应 r=1, 2, 3)
        r_val = r_values[r_idx]

        # 确定在 72 行矩阵中对应的行索引
        y_index = r_idx + t_idx * R_len

        # 遍历所有 l 的可能值 (0 到 12)
        for l_val in l_values:
            state = (r_val, l_val)
            l_col_index = l_val  # X 轴索引

            # 查找该 (r, l) 状态是否存在于筛选后的数据中
            if state in state_to_idx:
                data_idx = state_to_idx[state]
                # W_values_filtered 的第 data_idx 行, 第 t_idx 列
                W_value = W_values_filtered[data_idx, t_idx]
                heatmap_matrix[y_index, l_col_index] = W_value

# --- 4. 绘制热图 ---
fig, ax = plt.subplots(figsize=(10, 12))

im = ax.imshow(
    heatmap_matrix,
    cmap='Reds',
    aspect='auto',
    origin='lower'
)

# --- 5. 设置轴标签和刻度 ---
# X 轴 (l=0 到 l=12)
ax.set_xticks(l_values)
ax.set_xticklabels(l_values)
ax.set_xlabel('l', fontsize=12)

# 右侧 Y 轴: 标记主要分组 t=1, t=2, ...
t_ticks = [r_idx + t_idx * R_len + R_len / 2 - 0.5 for t_idx in range(T_len) for r_idx in [0]]
t_labels = [f't={t + 1}' for t in range(T_len)]
sec_ax = ax.secondary_yaxis('right')
sec_ax.set_yticks(t_ticks)
sec_ax.set_yticklabels(t_labels, fontsize=10, rotation=-90, va='center')
sec_ax.set_ylabel('Time t', rotation=270, labelpad=15, fontsize=12)

# 左侧 Y 轴: 标记次要分组 r=1, 2, 3
r_labels_final = []
r_ticks = []
# 只在 t=1, t=6, t=12, t=18, t=24 处显示 r 标签
for t in [0, 5, 11, 17, 23]:
    for r in range(R_len):
        r_ticks.append(t * R_len + r)
        r_labels_final.append(f'r={r_values[r]}')

ax.set_yticks(r_ticks)
ax.set_yticklabels(r_labels_final, fontsize=8)
ax.set_ylabel('r', rotation=90, labelpad=15, fontsize=12)

# 添加颜色条
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Whittle Index W')

ax.set_title(f'Composite Heatmap of W(r, l, t) vs l and stacked (t, r) (r=1, 2, 3 only, Action={action_index})',
             fontsize=14)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('whittle_index_composite_heatmap_stacked_tr_r_excl.png')
print("排除 r=0 后的复合热图已成功生成并保存为 'whittle_index_composite_heatmap_stacked_tr_r_excl.png'。")
plt.show()