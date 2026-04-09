import numpy as np
import pandas as pd
import itertools
from parameter_setting_CHEN106 import *
import time
# --- 1. 惩罚函数 F 与参数配置 ---
   # 假设单位电价 p=1.0 (您可以根据实际 Proposition 1 调整)
penalty_weight = 0.8
T=period
def f(x):
        return penalty_weight * (max(0, x)**2)

def delta_f(x):
    if x <= 0: return 0
    return f(x) - f(x-1)

# 车辆到达逻辑：假设如果槽位空，每步有 lambda 的概率来新车

def get_arrival_prob(r, l):
    """从 parameter_setting 的分布中获取概率"""
    if r in r_dist and l in l_dist:
        ri = r_dist.index(r)
        li = l_dist.index(l)
        return r_p[ri] * l_p[li]
    return 0

# --- 2. MDP 核心逻辑 ---
def get_reward(s, a, nu, t):
    """计算 t 时刻补贴后的收益"""
    P_COST = get_time_varying_p0(t)
    r, l = s
    if r == 0 or l == 0:
        return 0
    imm_reward = (alpha - P_COST) * min(a, r)
    penalty = f(r - min(a, r)) if l == 1 else 0
    return imm_reward - penalty - nu * min(a, r)


#按照L=1之后就会有车正常的进入
def get_transitions(s, a, t):
    """状态转移概率"""
    r, l = s
    prob_arrival = get_time_varying_prob(t)
    if l > 1:
        # 还没到期，需求减少，期限减1
        next_s = (max(0, r - a), l - 1)
        return [(S_TO_IDX[next_s], 1.0)]
    else:
        # L=1 或 (0,0): 车辆离开，可能来新车
        trans = []
        trans.append((S_TO_IDX[(0, 0)], 1.0 - prob_arrival)) # 无新车
        for rs in r_dist:
            for ls in l_dist:
                prob = prob_arrival * get_arrival_prob(rs, ls)
                if prob > 0:
                    trans.append((S_TO_IDX[(rs, ls)], prob))
        return trans

# --- 3. 相对价值迭代 (RVI) ---
def solve_rvi(nu):
    tol = 1e-4
    h = np.zeros((T, NUM_STATES))
    # 时变模型可能需要更多迭代次数才能收敛，建议把 50 改大一点，比如 200
    for _ in range(200):
        h_new = np.zeros((T, NUM_STATES))

        # 1. 优先计算一整个周期的所有状态价值
        for t in reversed(range(T)):
            t_next = (t + 1) % T
            for s_idx, s in enumerate(S_SPACE):
                q_vals = []
                for a in range(MAX_CHARGE + 1):
                    r_imm = get_reward(s, a, nu, t)
                    # 注意：future_v 使用的是上一轮完整的 h 矩阵
                    future_v = sum(p * h[t_next, ns] for ns, p in get_transitions(s, a, t))
                    q_vals.append(r_imm + future_v)
                h_new[t, s_idx] = max(q_vals)

        # 2. 在整个 T 周期计算完成后，再做相对价值标准化
        rho = h_new[0, 2]  # 选择 (t=0, s_idx=2) 作为基准状态
        h_new -= rho

        # 3. 在外层循环进行收敛判断
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new.copy()
            break
        h = h_new.copy()

    return h

# --- 4. 二分查找与理论验证 ---
def find_index(s, k, t):
    """找到使得 Q(s, k) = Q(s, k+1) 的 nu"""
    low, high = -10, 20.0
    for _ in range(100):
        mid = (low + high) / 2
        h = solve_rvi(mid)
        # 计算 Q 值
        def get_q(action, t_curr):
            t_next = (t_curr + 1) % T
            r_imm = get_reward(s, action, mid, t_curr)
            future_v = sum(p * h[t_next, ns] for ns, p in get_transitions(s, action, t_curr))
            return r_imm + future_v
        if get_q(k+1,t) > get_q(k,t):
            low = mid
        else: high = mid
    return (low + high) / 2


def find_index_minimized(s, k,t):
    low, high = -10, 20.0
    epsilon = 1e-8

    for _ in range(100):
        mid = (low + high) / 2
        h = solve_rvi(mid)
        def q_val(action, t_curr):
            t_next = (t_curr + 1) % T
            r_imm = get_reward(s, action, mid, t_curr)
            future_v = sum(p * h[t_next, ns] for ns, p in get_transitions(s, action, t_curr))
            return r_imm + future_v
        # 目标：寻找最小的 nu，使得 q_val(k) >= q_val(k+1)
        # 如果当前 mid 已经满足了让 k 优于或等于 k+1 的条件
        # 我们尝试去更小的区间找，看是否有更小的 nu 也能满足
        if q_val(k,t) >= q_val(k + 1,t)-epsilon:
            high = mid
        else:
            low = mid
    return high

def get_theoretical_index(r, l, i,t):
    """Proposition 1 的理论公式 """
    W = MAX_CHARGE
    p = get_time_varying_p0(t)
    if r == 0: return 0
    if r <= (l-1)*W:
        return -10
    else:
        # (L-1)W < R <= LW 或 R > LW 情况
        if i <= max(0, r - (l-1)*W - 1):
            return (alpha - p) + delta_f(r - (l-1)*W - i)
        else:
            return -10

# --- 5. 执行并导出 ---
# --- 5. 执行并导出 ---
print("开始计算 Index...")
start_time = time.time()

# 初始化用于导出的 3D NumPy 数组: 形状为 (状态总数, 周期长度T, 动作数量)
# 根据你的测试代码，动作数量等于 MAX_CHARGE
index_table = np.zeros((NUM_STATES, T, MAX_CHARGE))

data = []

# 调换一下循环顺序，先遍历状态，再遍历时间，这样视觉上进度更清晰
for s_idx, s in enumerate(S_SPACE):
    r, l = s
    if r == 0:
        continue  # R=0时，Index默认就是0，矩阵初始化已经是0了，直接跳过

    for t in range(T):
        for k in range(MAX_CHARGE):
            num_idx = find_index_minimized(s, k, t)
            theo_idx = get_theoretical_index(r, l, k, t)

            # 将算出的数值 Index 写入 3D 矩阵
            index_table[s_idx, t, k] = num_idx

            data.append({
                'time': t,
                'Demand_R': r,
                'Deadline_L': l,
                'Action_Transition': f"{k}",
                'Numerical_Index': round(num_idx, 2),
                'Theoretical_Index': round(theo_idx, 2),
                'Abs_Error': round(abs(num_idx - theo_idx), 2)
            })

    # 打印进度 (因为计算很慢，方便你观察)
    if s_idx % 10 == 0:
        print(f"已处理 {s_idx}/{NUM_STATES} 个状态...")

# 导出 Excel 方便人工查看
df = pd.DataFrame(data)
df.to_excel("Whittle_Index_Result_varying.xlsx", index=False)

# ====== 新增：导出 .npy 文件 ======
npy_filename = f"index_Poisson_varying_penal={penalty_weight}.npy"
np.save(npy_filename, index_table)
# ==================================

print(f"Total time used is {time.time() - start_time:.4f}s\n")
print(f"计算完成！Excel 已保存，Numpy 矩阵已保存为: {npy_filename}")