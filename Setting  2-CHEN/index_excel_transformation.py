import numpy as np
import pandas as pd
from pathlib import Path

EXCEL_DIR = Path("Results_excel")
EXCEL_DIR.mkdir(parents=True, exist_ok=True)

npy_dir = Path("Results_npy")
npy_dir.mkdir(parents=True, exist_ok=True)
# 1. 读取你最新的 .npy 文件
file_path = npy_dir / 'index_Xu_cache_period=24_pw=0.8_1610_Bernoulli.npy'
data = np.load(file_path)

# 2. 设置输出文件名
output_file = EXCEL_DIR / 'merged_output_24_tables.xlsx'

# 用于存储所有 DataFrame 的列表
all_tables = []

# 3. 循环处理数据
for i in range(24):
    # 提取切片数据 (67x3)
    df_slice = pd.DataFrame(data[:, i])

    # 设置表头，Table_n 作为前缀方便区分
    df_slice.columns = [f'T{i + 1}_V1']

    # 将当前表格加入列表
    all_tables.append(df_slice)

    # 如果不是最后一个表格，插入一个空列 DataFrame
    if i < 23:
        # 创建一个与原数据行数相同（67行）、宽度为1的空 DataFrame
        empty_col = pd.DataFrame({'': [''] * data.shape[0]})
        all_tables.append(empty_col)

# 4. 横向合并 (axis=1)
merged_df = pd.concat(all_tables, axis=1)

# 5. 保存到单个 Excel Sheet 中
merged_df.to_excel(output_file, index=False)

print(f"转换成功！已生成合并后的 Excel 文件：{output_file}")