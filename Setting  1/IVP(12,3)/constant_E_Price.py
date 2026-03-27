import numpy as np
import itertools
import sys
import os
import time
from tqdm import tqdm
# 增加递归深度以处理可能深度很大的 PVI 迭代
sys.setrecursionlimit(2000)

# --- 1. 模型参数及分布 (基于 simulation(12,3)_diff_v.py) ---

# R/L 分布 - 用于转移概率
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12]
r_p = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09]
l_dist = [1, 2, 3]
l_p= [0.3,0.4,0.3]

# 全局参数
T_PERIOD = 12 # 周期长度
MAX_R = max(r_dist)
MAX_L = max(l_dist)
MAX_CHARGE = 6
ALPHA = 3  # 单位充电收益
BETA = 0.95  # 折现因子
PENALTY_WEIGHT = 0.6  # 惩罚权重

# --- 2. 有效状态空间 S (R=0, L=0) 或 (R>0, L>0) ---
R_ACTIVE_SPACE = list(range(MAX_R + 1))
L_ACTIVE_SPACE = list(range(1, MAX_L + 1))
S_ACTIVE = list(itertools.product(R_ACTIVE_SPACE, L_ACTIVE_SPACE))
S_SPACE = [(0, 0)] + S_ACTIVE
S_TO_IDX = {s: i for i, s in enumerate(S_SPACE)}
NUM_STATES = len(S_SPACE)
A_SPACE = list(range(MAX_CHARGE + 1))  # 动作/模式


# --- 3. 辅助函数和模型函数 ---

def f(x, penalty_weight=PENALTY_WEIGHT):
    return (x ** 2) * penalty_weight


def get_time_varying_price(t):
    """p(t) - 周期性电价函数 (占位符)"""
    t_in_period = t % T_PERIOD
    if t_in_period <= 6:
        return 1.0
    elif t_in_period <= 12:
        return 1.0 - (0.2 * (t_in_period - 6) / 6.0)
    elif t_in_period <= 20:
        return 0.9 + (0.4 * (t_in_period - 12) / 8.0)
    else:
        return 1.2 - (0.2 * (t_in_period - 20) / 4.0)


def get_time_varying_prob(t, period=24, min_prob=0.1, max_prob=0.9):  # v1
    """P(Arrival | t) (来自您的代码)"""
    t_in_period = t % period
    half_period = period / 2
    prob_range = max_prob - min_prob
    if t_in_period <= half_period:
        return min_prob + prob_range * (t_in_period / half_period)
    else:
        return min_prob + prob_range * ((period - t_in_period) / half_period)


def R_original_func(s, a, t):
    R, L = s
    p_t = get_time_varying_price(t)
    if L > 0:
        charge_benefit = min(R, a) * (ALPHA - p_t)
        penalty = 0
        if L == 1:
            penalty = f(max(0, R - a))
        return charge_benefit - penalty
    else:
        return 0.0


def P_transition_func(s_prime, s, a, t):
    """状态转移概率 P(s' | s, a, t)"""
    R, L = s
    R_prime, L_prime = s_prime

    if L > 0:  # 活跃作业
        if L == 1:
            R_next, L_next = (0, 0)  # 截止，作业离开
        else:  # L >= 2
            R_next = max(R - a, 0)
            L_next = L - 1

        return 1.0 if (R_prime, L_prime) == (R_next, L_next) else 0.0

    else:  # L = 0 (必须是 s = (0, 0))
        prob_arrival = get_time_varying_prob(t)

        if R_prime == 0 and L_prime == 0:
            return 1.0 - prob_arrival  # 无新车到达

        elif R_prime in r_dist and L_prime in l_dist:
            r_idx = r_dist.index(R_prime)
            prob_R = r_p[r_idx]
            l_idx = l_dist.index(L_prime)
            prob_L = l_p[l_idx]
            return prob_arrival * prob_R * prob_L

        return 0.0


# --- 4. 周期性值迭代 PVI ---

def periodic_value_iteration(nu, V_current, T, S, A, beta, epsilon=1e-5, max_iter=2000):
    s_to_idx = S_TO_IDX
    num_states = len(S)
    V_current = np.copy(V_current)

    for k in range(max_iter):
        V_next = np.copy(V_current)
        max_diff = 0
        for t in reversed(range(T)):
            t_plus_1 = (t + 1) % T
            for s_idx, s in enumerate(S):
                max_q_value = -np.inf
                for a in A:
                    R_Original = R_original_func(s, a, t)
                    R_tilde = R_Original - nu * a
                    expected_future_value = 0

                    for s_prime in S:
                        s_prime_idx = s_to_idx[s_prime]
                        prob = P_transition_func(s_prime, s, a, t)
                        expected_future_value += prob * V_current[s_prime_idx, t_plus_1]
                    Q_value = R_tilde + beta * expected_future_value
                    max_q_value = max(max_q_value, Q_value)
                V_next[s_idx, t] = max_q_value
                max_diff = max(max_diff, abs(V_next[s_idx, t] - V_current[s_idx, t]))
        if max_diff < epsilon:
            return V_next
        V_current = V_next
    return V_current

# 5. Q-Value 计算
def get_Q_value(s, t, a, nu, V_nu, T, S, beta):
    t_plus_1 = (t + 1) % T
    R_Original = R_original_func(s, a, t)
    R_tilde = R_Original - nu * a

    expected_future_value = 0
    for s_prime in S:
        s_prime_idx = S_TO_IDX[s_prime]
        prob = P_transition_func(s_prime, s, a, t)
        expected_future_value += prob * V_nu[s_prime_idx, t_plus_1]

    return R_tilde + beta * expected_future_value


# 6. 索引二分查找
def calculate_index(s, t, mode_i, T, S, A, beta, nu_L, nu_R, index_tol=1e-4,
                    pvi_tol=1e-4):
    if s == (0, 0): return 0.0

    # 我们比较 mode_i 和 mode_i + 1
    i = mode_i
    i_plus_1 = mode_i + 1

    # 简单的边界检查
    if i_plus_1 >= len(A):
        return -999.0  # 无效的比较

    V_nu = np.zeros((NUM_STATES, T))

    nu_low = nu_L
    nu_high = nu_R

    # --- 边界预检查 (强制非负) ---
    # 检查 nu = 0 时的情况
    V_nu_0 = periodic_value_iteration(0.0, V_nu, T, S, A, beta, pvi_tol)
    Q0_0 = get_Q_value(s, t, i, 0.0, V_nu_0, T, S, beta)
    Q1_0 = get_Q_value(s, t, i_plus_1, 0.0, V_nu_0, T, S, beta)

    # 如果 Q(i) > Q(i+1) 在 nu=0 时成立，说明即使免费，多充这1度电也不划算。
    if Q0_0 > Q1_0:
        return 0.0

        # --- 二分查找 ---
    # 现在我们知道 nu=0 时 Q(i) < Q(i+1)，所以索引一定是正数。
    # 我们在 [0, nu_R] 之间搜索
    nu_low = 0.0

    while (nu_high - nu_low) > index_tol:
        nu_mid = (nu_low + nu_high) / 2

        V_nu_mid = periodic_value_iteration(nu_mid, V_nu, T, S, A, beta, pvi_tol)
        V_nu = V_nu_mid  # warm start

        Q0 = get_Q_value(s, t, i, nu_mid, V_nu_mid, T, S, beta)
        Q1 = get_Q_value(s, t, i_plus_1, nu_mid, V_nu_mid, T, S, beta)

        Delta_Q = Q0 - Q1

        if Delta_Q > 0:
            # Q(i) > Q(i+1)，说明 nu 太大，导致 i+1 惩罚过重
            nu_high = nu_mid
        else:
            # Q(i) < Q(i+1)，说明 nu 太小，i+1 仍然太划算
            nu_low = nu_mid

    return nu_high


# --- 7. 主函数：生成索引查找表 ---

def generate_index_lookup_table():
    print("--- 启动周期性 Whittle 索引查找表生成 ---")
    print(f"状态空间大小: {NUM_STATES} x 时间周期: {T_PERIOD} = {NUM_STATES * T_PERIOD} 个索引")

    # 维度升级：[状态, 时间, 动作]
    # index_table[s, t, k] 存储从模式 k 切换到 k+1 的索引
    index_table = np.zeros((NUM_STATES, T_PERIOD, MAX_CHARGE))

    # 设置合理的索引搜索范围 (根据您的回报函数，[-5, 5] 通常足够，但可能需要调整)
    NU_MIN_GUESS = 0
    NU_MAX_GUESS = 20

    for t in range(T_PERIOD):
        # 为每个时间步的状态计算添加进度条
        for s_idx, s in tqdm(enumerate(S_SPACE), total=NUM_STATES, desc=f"Calculating for t={t}"):
            if s != (0, 0):
                # 循环计算每个增量模式的索引
                for k in range(MAX_CHARGE):
                    nu = calculate_index(s, t, k, T_PERIOD, S_SPACE, A_SPACE, BETA,
                                         NU_MIN_GUESS, NU_MAX_GUESS)
                    index_table[s_idx, t, k] = nu
            else:
                index_table[s_idx, t, :] = 0.0
    return index_table


if __name__ == '__main__':
    FILE_NAME = 'whittle_index_multimode_time_varying_test.npy'

    if os.path.exists(FILE_NAME):
        print(f"--- 从缓存文件 '{FILE_NAME}' 加载索引 ---")
        TABLE = np.load(FILE_NAME)
    else:
        print(f"--- 缓存文件 '{FILE_NAME}' 不存在，开始计算 ---")
        TABLE = generate_index_lookup_table()
        print(f"\n--- 计算完成，保存到 '{FILE_NAME}' ---")
        np.save(FILE_NAME, TABLE)

    # --- 验证 s=(3, 2) 的不同模式索引 ---
    for r in r_dist:
        for l in l_dist:
            s_test = (r,l)
            s_idx = S_TO_IDX[s_test]
            for t in range(T_PERIOD):
                price = get_time_varying_price(t)
                # 使用 t % T_PERIOD 确保索引在周期内，避免再次出现 IndexError
                t_periodic = t % T_PERIOD

                idx_0_1 = TABLE[s_idx, t_periodic, 0]
                idx_1_2 = TABLE[s_idx, t_periodic, 1]
                idx_2_3 = TABLE[s_idx, t_periodic, 2]
                idx_3_4 = TABLE[s_idx, t_periodic, 3]
                idx_4_5 = TABLE[s_idx, t_periodic, 4]

                print(f"index({s_test}, t={t}, Mode=0->1)={idx_0_1:.2f}")
                print(f"index({s_test}, t={t}, Mode=1->2)={idx_1_2:.2f}")
                print(f"index({s_test}, t={t}, Mode=2->3)={idx_2_3:.2f}")
                print(f"index({s_test}, t={t}, Mode=3->4)={idx_3_4:.2f}")
                print(f"index({s_test}, t={t}, Mode=4->5)={idx_4_5:.2f}")


