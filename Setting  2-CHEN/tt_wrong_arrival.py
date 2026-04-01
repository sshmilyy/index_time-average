import numpy as np
import pandas as pd
import itertools
from parameter_setting_CHEN106 import *
import time
# --- 1. 惩罚函数 F 与参数配置 ---
P_COST = 1.0   # 假设单位电价 p=1.0 (您可以根据实际 Proposition 1 调整)
BETA = 0.8     # 假设惩罚系数，f(x) = beta * x^2
def f(x):
    return BETA * (max(0, x)**2)

def delta_f(x):
    if x <= 0: return 0
    return f(x) - f(x-1)

# 车辆到达逻辑：假设如果槽位空，每步有 lambda 的概率来新车
LAMBDA_ARR = 0.5

def get_arrival_prob(r, l):
    """从 parameter_setting 的分布中获取概率"""
    if r in r_dist and l in l_dist:
        ri = r_dist.index(r)
        li = l_dist.index(l)
        return r_p[ri] * l_p[li]
    return 0

# --- 2. MDP 核心逻辑 ---
def get_reward(s, a, nu):
    """计算补贴后的收益 R_w - nu * a"""
    r, l = s
    if r == 0 or l == 0:
        return 0
    # 基础收益 (alpha - p) * min(a, r)
    imm_reward = (alpha - P_COST) * min(a, r)
    # L=1 时的到期惩罚
    penalty = f(r - min(a,r)) if l == 1 else 0
    return imm_reward - penalty - nu * a

# --- 修改后的转移函数：解耦未来车辆 ---
def get_transitions_decoupled(s, a):
    r, l = s
    if l > 1:
        # 当前车辆未离开，继续追踪
        next_s = (max(0, r - a), l - 1)
        return [(S_TO_IDX[next_s], 1.0)]
    elif l == 1:
        return [(S_TO_IDX[(0, 0)], 1.0)]
    else:
        trans = []
        trans.append((S_TO_IDX[(0, 0)], 1.0 - LAMBDA_ARR)) # 无新车
        for rs in r_dist:
            for ls in l_dist:
                prob = LAMBDA_ARR * get_arrival_prob(rs, ls)
                if prob > 0:
                    trans.append((S_TO_IDX[(rs, ls)], prob))
        return trans

# --车辆到L=1的时候，会强制进入（0，0）
def solve_rvi_decoupled(nu):
    tol = 1e-4
    h = np.zeros(NUM_STATES)
    for _ in range(100):
        h_new = np.zeros(NUM_STATES)
        for s_idx, s in enumerate(S_SPACE):
            q_vals = []
            for a in range(MAX_CHARGE + 1):
                r_imm = get_reward(s, a, nu)
                future_v = sum(p * h[ns] for ns, p in get_transitions_decoupled(s, a))
                q_vals.append(r_imm + future_v)
            h_new[s_idx] = max(q_vals)
        rho = h_new[11]
        h_new -= rho # 标准化防止数值爆炸
        if np.max(np.abs(h_new - h)) < tol: break
        h = h_new
    return h


# --- 4. 二分查找与理论验证 ---
def find_index_minimized(s, k):
    low, high = -20, 20.0
    epsilon = 1e-9  # 引入微小偏差处理数值噪声
    for _ in range(100):
        mid = (low + high) / 2
        h = solve_rvi_decoupled(mid)

        def q_val(a):
            return get_reward(s, a, mid) + sum(p * h[ns] for ns, p in get_transitions_decoupled(s, a))

        # 目标：寻找最小的 nu，使得 q_val(k) >= q_val(k+1)
        # 如果当前 mid 已经满足了让 k 优于或等于 k+1 的条件
        # 我们尝试去更小的区间找，看是否有更小的 nu 也能满足
        if q_val(k) >= q_val(k + 1) - epsilon:
            high = mid
        else:
            low = mid

    return high



def get_theoretical_index(r, l, i):
    """Proposition 1 的理论公式 """
    W = MAX_CHARGE
    p = P_COST
    if r == 0: return 0
    if r <= (l-1)*W:
        return alpha - p
    else:
        # (L-1)W < R <= LW 或 R > LW 情况
        if i <= max(0, r - (l-1)*W - 1):
            return (alpha - p) + delta_f(r - (l-1)*W - i)
        else:
            return alpha - p

# --- 5. 执行并导出 ---
print("开始计算 Index...")
start_time=time.time()
data = []
for s in S_SPACE:
    r, l = s
    if r == 0: continue
    for k in range(MAX_CHARGE):
        num_idx = find_index_minimized(s, k)
        theo_idx = get_theoretical_index(r, l, k)
        data.append({
            'Demand_R': r,
            'Deadline_L': l,
            'Action_Transition': f"{k}",
            'Numerical_Index': round(num_idx, 2),
            'Theoretical_Index': round(theo_idx, 2),
            'Abs_Error': round(abs(num_idx - theo_idx), 2)
        })

df = pd.DataFrame(data)
df.to_excel("Whittle_Index_wrong_const.xlsx", index=False)
print(f"Total time used is {time.time() - start_time:.4f}s\n")
print("计算完成，结果已保存至 Whittle_Index_wrong_const.xlsx")
