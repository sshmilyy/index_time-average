import numpy as np
import pandas as pd
import os

# ================= 配置区域 (只改这里) =================
# 输入文件名 (请确保该文件和脚本在同一个文件夹下)
INPUT_FILENAME = 'index_varying ePrice_penal=0.2_(12,5).npy'

# 输出文件名 (想存成什么名字)
OUTPUT_FILENAME = 'result.xlsx'


# ====================================================

def convert_npy_to_excel(input_file, output_file):
    print(f"🔄 正在读取文件: {input_file} ...")

    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 '{input_file}'")
        print("   -> 请确保 npy 文件和脚本在同一个文件夹，或者填写绝对路径。")
        return

    try:
        # 2. 加载数据
        data = np.load(input_file, allow_pickle=True)
        print(f"✅ 读取成功! 数据形状 (Shape): {data.shape}")

        # 3. 处理数据维度
        # 如果是 0维 (标量)
        if data.ndim == 0:
            df = pd.DataFrame([data])

        # 如果是 1维 (一行数据)
        elif data.ndim == 1:
            df = pd.DataFrame(data)

        # 如果是 2维 (标准表格)
        elif data.ndim == 2:
            df = pd.DataFrame(data)

        # 如果是 3维或更高 (例如图片或复杂数组)
        else:
            print(f"⚠️  警告: 数据是 {data.ndim} 维的，Excel 只能存 2 维。")
            print("   -> 正在自动将数据'展平'为 2 维 (Reshape)...")
            # 保留第一维度(行数)，将后面所有维度合并
            df = pd.DataFrame(data.reshape(data.shape[0], -1))

        # 4. 保存为 Excel
        print(f"💾 正在写入 Excel (这可能需要几秒钟)...")
        df.to_excel(output_file, index=False, header=False)  # header=False 表示不自动生成列名0,1,2

        print("-" * 30)
        print(f"🎉 成功! 文件已保存为: {output_file}")
        print(f"   -> 文件位置: {os.path.abspath(output_file)}")
        print("-" * 30)

    except Exception as e:
        print("\n❌ 发生未知错误:")
        print(e)
        print("\n可能的原因：")
        print("1. 缺少依赖库 (运行: pip install pandas openpyxl numpy)")
        print("2. npy 文件损坏")
        print("3. npy 里存的是字典而不是数组 (这种情况需要特殊处理)")


if __name__ == '__main__':
    convert_npy_to_excel(INPUT_FILENAME, OUTPUT_FILENAME)