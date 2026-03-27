def sort_3d_array(R, L, A):
    """
    对三维数组索引 (r, l, a) 进行排序

    参数:
    R: 第一维的最大值 (1-R)
    L: 第二维的最大值 (1-L)
    A: 第三维的最大值 (0-(A-1))

    返回:
    排序后的索引列表
    """
    # 生成所有可能的组合
    indices = []
    for r in range(1, R + 1):
        for l in range(1, L + 1):
            for a in range(A):
                indices.append((r, l, a))

    # 自定义排序函数
    def sort_key(item):
        r, l, a = item
        # 计算 index = r - 3*(l-1) - a
        index = r - 3 * (l - 1) - a
        # 返回排序键值：
        # 1. 首先按 -index 排序（从大到小）
        # 2. 然后按 l 排序（从小到大）
        # 3. 然后按 -r 排序（从大到小）
        # 4. 最后按 a 排序（从小到大）
        return (-index, l, -r, a)

    # 排序并返回结果
    sorted_indices = sorted(indices, key=sort_key)
    return sorted_indices


# 示例使用
if __name__ == "__main__":
    # 假设 R=3, L=2, A=2
    R, L, A = 10, 4, 3
    sorted_indices = sort_3d_array(R, L, A)

    print("原始所有组合:")
    for r in range(1, R + 1):
        for l in range(1, L + 1):
            for a in range(A):
                print(f"({r},{l},{a})", end=" ")
    print("\n")

    print("排序后的组合 (按 r-3*(l-1)-a 从大到小):")
    for i, (r, l, a) in enumerate(sorted_indices):
        index = r - 3 * (l - 1) - a
        print(f"{i + 1}. ({r},{l},{a}) - index={index}")

    print("\n排序规则解释:")
    print("1. 首先按 index = r-3*(l-1)-a 从大到小")
    print("2. index相同时，第二维(l)小的优先")
    print("3. l相同时，第一维(r)大的优先")
    print("4. r相同时，第三维(a)小的优先")