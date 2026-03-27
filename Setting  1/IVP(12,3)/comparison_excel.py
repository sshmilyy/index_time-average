import pandas as pd
import numpy as np


def compare_excel_files(file_path1, file_path2, tolerance=1e-3):
    print(f"--- 开始对比 ---")
    print(f"文件 1: {file_path1}")
    print(f"文件 2: {file_path2}")

    # 1. 读取文件 (根据之前的截图，数据似乎没有表头，所以用 header=None)
    # 如果你的 Excel 有表头，请改为 header=0
    try:
        # 尝试读取 Excel
        df1 = pd.read_excel(file_path1, header=None)
        df2 = pd.read_excel(file_path2, header=None)
    except:
        # 如果读取 Excel 失败，尝试读取 CSV (以防万一)
        df1 = pd.read_csv(file_path1, header=None)
        df2 = pd.read_csv(file_path2, header=None)

    # 2. 检查形状是否一致
    if df1.shape != df2.shape:
        print(f"\n[错误] 形状不一致!")
        print(f"文件 1 形状: {df1.shape}")
        print(f"文件 2 形状: {df2.shape}")
        return

    print(f"形状一致: {df1.shape}")

    # 3. 转换为 Numpy 数组进行数值计算
    arr1 = df1.to_numpy(dtype=float)
    arr2 = df2.to_numpy(dtype=float)

    # 处理可能的 NaN (空值)
    if np.isnan(arr1).any() or np.isnan(arr2).any():
        print("[警告] 数据中包含空值 (NaN)，已填充为 0 进行对比")
        arr1 = np.nan_to_num(arr1)
        arr2 = np.nan_to_num(arr2)

    # 4. 计算差异
    diff = arr1 - arr2
    abs_diff = np.abs(diff)

    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    # 统计超过容差的数量
    num_significant_diff = np.sum(abs_diff > tolerance)
    total_elements = arr1.size
    percent_diff = (num_significant_diff / total_elements) * 100

    print(f"\n--- 对比结果统计 ---")
    print(f"最大差异 (Max Error):  {max_diff:.8f}")
    print(f"平均差异 (Mean Error): {mean_diff:.8f}")
    print(f"差异 > {tolerance} 的数量: {num_significant_diff} / {total_elements} ({percent_diff:.2f}%)")

    # 5. 判定结论
    print(f"\n--- 结论 ---")
    if max_diff == 0:
        print("✅ 两个表格完全一致 (Perfect Match)")
    elif max_diff < tolerance:
        print(f"✅ 两个表格在容差 {tolerance} 内被视为一致。")
        print("   (微小的差异通常由 32位/64位 浮点精度或计算顺序导致，属正常现象)")
    else:
        print("⚠️ 两个表格存在显著差异！")

        # 找出差异最大的位置
        max_diff_idx = np.unravel_index(np.argmax(abs_diff, axis=None), abs_diff.shape)
        r, c = max_diff_idx
        print(f"   差异最大的位置: 行 {r}, 列 {c}")
        print(f"   文件1数值: {arr1[r, c]}")
        print(f"   文件2数值: {arr2[r, c]}")
        print(f"   差值: {diff[r, c]}")

    # 6. (可选) 导出差异矩阵
    if max_diff > 0:
        diff_df = pd.DataFrame(diff)
        output_filename = 'diff_report.csv'
        diff_df.to_csv(output_filename, index=False, header=False)
        print(f"\n详细差异矩阵已保存至: {output_filename}")


if __name__ == "__main__":
    # 替换为你实际的文件名
    file1 = 'result_0.2.xlsx'
    file2 = 'result_opt_0.2.xlsx'

    # 容差设为 1e-3 (0.001) 或 1e-4，取决于你对精度的要求
    compare_excel_files(file1, file2, tolerance=1e-5)