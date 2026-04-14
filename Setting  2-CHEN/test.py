import pandas as pd

# 读取 Excel 文件
file_path = "output_24_tables_penal0.8test2_varying.xlsx"
xls = pd.ExcelFile(file_path)

# 获取所有工作表名称并排序（保证顺序为 Table_1 到 Table_24）
sheet_names = sorted(xls.sheet_names, key=lambda x: int(x.split('_')[1]))

# 存储每个工作表的数据框
dfs = []
max_rows = 0

# 第一遍遍历：读取数据并记录最大行数
for i, sheet in enumerate(sheet_names, start=1):
    df = pd.read_excel(xls, sheet_name=sheet, header=0)
    # 重命名列，添加表序号前缀
    df = df.rename(columns=lambda col: f"Table{i}_{col}")
    dfs.append(df)
    max_rows = max(max_rows, len(df))

# 第二遍处理：将每个 DataFrame 扩展到最大行数，缺失部分填 NaN
aligned_dfs = []
for df in dfs:
    if len(df) < max_rows:
        # 创建缺失行的 NaN DataFrame
        missing_rows = pd.DataFrame(index=range(max_rows - len(df)), columns=df.columns)
        df = pd.concat([df, missing_rows], ignore_index=True)
    aligned_dfs.append(df)

# 横向拼接
merged_df = pd.concat(aligned_dfs, axis=1)

# 保存结果
merged_df.to_excel("merged_tables_horizontal.xlsx", index=False)
print("合并完成，已保存为 merged_tables_horizontal.xlsx")