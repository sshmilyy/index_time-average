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
l_dist = [1, 2, 3,4,5]
l_p= [0.2,0.2,0.2,0.2,0.2]

# 全局参数
period = 24  # 周期长度
max_r = max(r_dist)
max_l = max(l_dist)
MAX_CHARGE = max_r/2
alpha = 3  # 单位充电收益
BETA = 0.95  # 折现因子
penalty_weight = 0.6  # 惩罚权重

# --- 2. 有效状态空间 S (R=0, L=0) 或 (R>0, L>0) ---
R_ACTIVE_SPACE = list(range(max_r + 1))
L_ACTIVE_SPACE = list(range(1, max_l + 1))
S_ACTIVE = list(itertools.product(R_ACTIVE_SPACE, L_ACTIVE_SPACE))
S_SPACE = [(0, 0)] + S_ACTIVE
S_TO_IDX = {s: i for i, s in enumerate(S_SPACE)}
NUM_STATES = len(S_SPACE)
A_SPACE = list(range(int(MAX_CHARGE) + 1))  # 动作/模式


# --- 3. 辅助函数和模型函数 ---



def precompute_matrices(T, S_space, A_space):
    num_states = len(S_space)
    num_actions = len(A_space)

    # P[t, a, s, s_next]
    P_mat = np.zeros((T, num_actions, num_states, num_states))
    # R[t, s, a]
    R_mat = np.zeros((T, num_states, num_actions))

    # 将动作数组化，方便广播
    actions_arr = np.array(A_space)

    print("Pre-computing P and R matrices...")
    for t in range(T):
        for s_idx, s in enumerate(S_space):
            for a_idx, a in enumerate(A_space):
                # 1. 计算 Reward (不包含 nu * a，这部分在 PVI 中动态减)
                R_mat[t, s_idx, a_idx] = reward_function(s, a, t)

                # 2. 计算 Transition Probability
                # 这里为了加速，不循环 s_prime，而是根据逻辑直接填值
                # 你的逻辑中，大部分转移是确定性的，除了 (0,0)
                R_val, L_val = s

                if L_val > 0:
                    # 确定性转移逻辑
                    if L_val == 1:
                        next_s = (0, 0)
                    else:
                        next_R = max(R_val - a, 0)
                        next_L = L_val - 1
                        next_s = (next_R, next_L)

                    if next_s in S_TO_IDX:
                        next_idx = S_TO_IDX[next_s]
                        P_mat[t, a_idx, s_idx, next_idx] = 1.0

                else:  # s = (0,0)
                    prob_arrival = get_time_varying_prob(t)
                    # Case 1: 无新车
                    next_idx_0 = S_TO_IDX[(0, 0)]
                    P_mat[t, a_idx, s_idx, next_idx_0] += (1.0 - prob_arrival)

                    # Case 2: 新车到达 (分布概率)
                    # 这部分稍微有点慢，但只做一次
                    for r_idx, r_val in enumerate(r_dist):
                        for l_idx, l_val in enumerate(l_dist):
                            prob = prob_arrival * r_p[r_idx] * l_p[l_idx]
                            next_s = (r_val, l_val)
                            if next_s in S_TO_IDX:
                                next_idx = S_TO_IDX[next_s]
                                P_mat[t, a_idx, s_idx, next_idx] += prob

    return P_mat, R_mat, actions_arr

def f(x):
    return (x ** 2) * penalty_weight

def get_time_varying_p0(t):
    t_in_period = t % period
    if t_in_period <= 6:
        return 0.67
    elif t_in_period <= 22:
        return 1.33
    elif t_in_period <= 23:
        return 0.67

def get_time_varying_prob(t, period=24, min_prob=0.4, max_prob=0.6):  # v1
    """P(Arrival | t) (来自您的代码)"""
    t_in_period = t % period
    half_period = period / 2
    prob_range = max_prob - min_prob
    if t_in_period <= half_period:
        return min_prob + prob_range * (t_in_period / half_period)
    else:
        return min_prob + prob_range * ((period - t_in_period) / half_period)


def reward_function(s, a, t):
    R, L = s
    p_t = get_time_varying_p0(t)
    if L > 0:
        charge_benefit = min(R, a) * (alpha - p_t)
        penalty = 0
        if L == 1:
            penalty = f(max(0, R - a))
        return charge_benefit - penalty
    else:
        return 0.0
def R_original_func(current_state, action,t):
    current_p=get_time_varying_p0(t)
    rewards = [
        (min(r, a) * (alpha - current_p) - (f(max(0, r - a)) if l == 1 else 0))
        for r, l, a in zip(current_state[::2],
                           current_state[1::2],
                           action)
    ]
    return sum(rewards)

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
                    R_Original = reward_function(s, a, t)
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
    R_Original = reward_function(s, a, t)
    R_tilde = R_Original - nu * a

    expected_future_value = 0
    for s_prime in S:
        s_prime_idx = S_TO_IDX[s_prime]
        prob = P_transition_func(s_prime, s, a, t)
        expected_future_value += prob * V_nu[s_prime_idx, t_plus_1]

    return R_tilde + beta * expected_future_value


def periodic_value_iteration_vectorized(nu, V_current, T, beta, P_mat, R_mat, actions_arr, epsilon=1e-5, max_iter=2000):
    """
    P_mat: (T, A, S, S')
    R_mat: (T, S, A)
    V_current: (S, T)
    """
    num_states = R_mat.shape[1]

    # 预计算 R_tilde = R(s,a,t) - nu * a
    # 广播机制: (T, S, A) - (A,) -> (T, S, A)
    R_tilde_all = R_mat - nu * actions_arr[None, None, :]

    for k in range(max_iter):
        V_next = np.copy(V_current)
        max_diff = 0

        for t in reversed(range(T)):
            t_plus_1 = (t + 1) % T
            V_future = V_current[:, t_plus_1]  # Shape: (S,)

            # 核心矩阵运算：计算期望未来价值
            # P_mat[t]: (A, S, S')
            # V_future: (S')
            # dot -> (A, S) -> 转置为 (S, A) 以匹配 R_tilde
            expected_future = np.dot(P_mat[t], V_future).T

            # Q(s,a) = R_tilde + beta * E[V]
            Q_values = R_tilde_all[t] + beta * expected_future  # Shape: (S, A)

            # Bellman Optimality
            V_next[:, t] = np.max(Q_values, axis=1)  # Shape: (S,)

        diff = np.max(np.abs(V_next - V_current))
        if diff < epsilon:
            return V_next
        V_current = V_next

    return V_current


# --- 优化后的 get_Q_value ---
def get_Q_value_vectorized(s_idx, t, a_idx, nu, V_nu, T, beta, P_mat, R_mat, actions_arr):
    t_plus_1 = (t + 1) % T

    # 直接查表
    R_val = R_mat[t, s_idx, a_idx]
    R_tilde = R_val - nu * actions_arr[a_idx]

    # 向量点积计算期望
    # P_mat[t, a_idx, s_idx, :] 是一个 (S') 的向量
    # V_nu[:, t_plus_1] 是一个 (S') 的向量
    prob_vec = P_mat[t, a_idx, s_idx, :]
    v_next_vec = V_nu[:, t_plus_1]

    expected_future_value = np.dot(prob_vec, v_next_vec)

    return R_tilde + beta * expected_future_value

# 6. 索引二分查找
def calculate_index_opt(s_idx, t, mode_i, T, beta, nu_L, nu_R, P_mat, R_mat, actions_arr, index_tol=1e-4):
    # s 传入的是 idx
    if s_idx == 0: return 0.0  # (0,0) state

    i = mode_i
    i_plus_1 = mode_i + 1
    if i_plus_1 >= len(actions_arr): return -999.0

    num_states = R_mat.shape[1]
    V_nu = np.zeros((num_states, T))  # Initialize

    # 边界检查 nu=0
    V_nu_0 = periodic_value_iteration_vectorized(0.0, V_nu, T, beta, P_mat, R_mat, actions_arr)
    Q0_0 = get_Q_value_vectorized(s_idx, t, i, 0.0, V_nu_0, T, beta, P_mat, R_mat, actions_arr)
    Q1_0 = get_Q_value_vectorized(s_idx, t, i_plus_1, 0.0, V_nu_0, T, beta, P_mat, R_mat, actions_arr)

    if Q0_0 > Q1_0: return 0.0

    nu_low = 0.0
    nu_high = nu_R

    while (nu_high - nu_low) > index_tol:
        nu_mid = (nu_low + nu_high) / 2

        # 使用向量化 PVI
        V_nu_mid = periodic_value_iteration_vectorized(nu_mid, V_nu, T, beta, P_mat, R_mat, actions_arr)
        V_nu = V_nu_mid  # Warm start

        Q0 = get_Q_value_vectorized(s_idx, t, i, nu_mid, V_nu_mid, T, beta, P_mat, R_mat, actions_arr)
        Q1 = get_Q_value_vectorized(s_idx, t, i_plus_1, nu_mid, V_nu_mid, T, beta, P_mat, R_mat, actions_arr)

        if Q0 - Q1 > 0:
            nu_high = nu_mid
        else:
            nu_low = nu_mid

    return nu_high


# --- 7. 主函数：生成索引查找表 ---

def generate_index_lookup_table():
    print("--- 启动周期性 Whittle 索引查找表生成 ---")
    print(f"状态空间大小: {NUM_STATES} x 时间周期: {period} = {NUM_STATES * period} 个索引")

    # 维度升级：[状态, 时间, 动作]
    # index_table[s, t, k] 存储从模式 k 切换到 k+1 的索引
    index_table = np.zeros((int(NUM_STATES), period, int(MAX_CHARGE)))

    # 设置合理的索引搜索范围 (根据您的回报函数，[-5, 5] 通常足够，但可能需要调整)
    NU_MIN_GUESS = 0
    NU_MAX_GUESS = 20
    P_mat, R_mat, actions_arr = precompute_matrices(period, S_SPACE, A_SPACE)
    for t in range(period):
        # 为每个时间步的状态计算添加进度条
        for s_idx, s in tqdm(enumerate(S_SPACE), total=NUM_STATES, desc=f"Calculating for t={t}"):
            if s != (0, 0):
                # 循环计算每个增量模式的索引
                for k in range(int(MAX_CHARGE)):
                    nu = calculate_index_opt(s_idx, t, k, period, BETA,
                                         NU_MIN_GUESS, NU_MAX_GUESS, P_mat, R_mat, actions_arr)
                    index_table[s_idx, t, k] = nu
            else:
                index_table[s_idx, t, :] = 0.0
    return index_table


if __name__ == '__main__':
    FILE_NAME = 'index_varying ePrice_penal=0.6_(12,5).npy'

    if os.path.exists(FILE_NAME):
        print(f"--- 从缓存文件 '{FILE_NAME}' 加载索引 ---")
        TABLE = np.load(FILE_NAME)
    else:
        print(f"--- 缓存文件 '{FILE_NAME}' 不存在，开始计算 ---")
        TABLE = generate_index_lookup_table()
        print(f"\n--- 计算完成，保存到 '{FILE_NAME}' ---")
        np.save(FILE_NAME, TABLE)




