import numpy as np
import pandas as pd

# 1. 读取降维后的 2D .npy 文件
file_path = 'index_Stationary_Theoretical_penal=0.8_chen(10,6).npy'
data = np.load(file_path)

# 打印一下维度确认，应该是类似 (67, 3) 的二维形状
print(f"读取的数据维度为: {data.shape}")

# 2. 直接将这一个二维矩阵转为 DataFrame
df = pd.DataFrame(data)

# （可选）给列命名。这里自动根据动作数量生成列名，比如 Action_0, Action_1, Action_2
num_actions = data.shape[1]
df.columns = [f'Action_{k}' for k in range(num_actions)]

# 3. 保存为只有一个 Sheet 的 Excel 文件
output_file = 'Index_penal=0.8_chen_thm(10,6)6.xlsx'

# 直接导出，不需要 ExcelWriter 和 for 循环了
df.to_excel(output_file, index=False, sheet_name='Stationary_Index')

print(f"转换成功！已生成包含单一表格的 Excel 文件：{output_file}")