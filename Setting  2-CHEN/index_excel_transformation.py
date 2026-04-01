import numpy as np
import pandas as pd

# 1. 读取你最新的 .npy 文件
file_path = 'index_cache_T24_penal0.2.npy'
data = np.load(file_path)

# 2. 设置输出的 Excel 文件名
output_file = 'output_24_tables.xlsx'

# 3. 创建 Excel 写入器
with pd.ExcelWriter(output_file) as writer:
    # 循环 24 次，每次取出一份 67x3 的数据
    for i in range(24):
        # 提取切片数据
        df_slice = pd.DataFrame(data[:, i, :])

        # 为了美观，给这 3 列加上表头
        df_slice.columns = ['Value_1', 'Value_2', 'Value_3']

        # 将其保存为一个新的 Sheet，名字叫 Table_1, Table_2 ... Table_24
        df_slice.to_excel(writer, sheet_name=f'Table_{i + 1}', index=False)

print(f"转换成功！已生成包含 24 个表格的 Excel 文件：{output_file}")