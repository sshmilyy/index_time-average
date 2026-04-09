import numpy as np
import itertools
import sys
import os
import time
from tqdm import tqdm
from numba import njit, prange  # 引入 Numba
from parameter_setting_CHEN106 import MAX_CHARGE, r_dist, l_dist, r_p, l_p, max_r, max_l, S_TO_IDX, alpha, S_SPACE, \
    A_SPACE, period, NUM_STATES, get_time_varying_p0, get_time_varying_prob

penalty_weight = 0.8


def f(x):
    return (x ** 2) * penalty_weight


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


def precompute_matrices_stationary(S_space, A_space):
    """预计算平稳环境下的状态转移矩阵和收益矩阵"""
    num_states = len(S_space)
    num_actions = len(A_space)

    P_mat = np.zeros((num_actions, num_states, num_states))
    R_mat = np.zeros((num_states, num_actions))
    actual_A_mat = np.zeros((num_states, num_actions))

    # 获取恒定的电价和到达率 (平稳环境)
    p_constant = get_time_varying_p0(0)
    arrival_prob = get_time_varying_prob(0)

    # ==========================
    # 替换后的正确读取概率逻辑
    # ==========================
    arrival_dist = np.zeros(num_states)
    for i in range(len(r_dist)):
        r_val = r_dist[i]
        prob_r = r_p[i]

        for j in range(len(l_dist)):
            l_val = l_dist[j]
            prob_l = l_p[j]

            if (r_val, l_val) in S_TO_IDX:
                arrival_dist[S_TO_IDX[(r_val, l_val)]] = prob_r * prob_l
    # ==========================

    for s_idx, s in enumerate(S_space):
        R_val, L_val = s
        for a_idx, a in enumerate(A_space):

            # 1. 实际有效充电量
            actual_a = min(R_val, a) if L_val > 0 else 0
            actual_A_mat[s_idx, a_idx] = actual_a

            # 2. 基础 Reward
            if L_val > 0:
                charge_benefit = actual_a * (alpha - p_constant)
                penalty = f(max(0, R_val - a)) if L_val == 1 else 0
                R_mat[s_idx, a_idx] = charge_benefit - penalty
            else:
                R_mat[s_idx, a_idx] = 0.0

            # 3. 状态转移概率
            if L_val > 1:
                next_R = max(0, R_val - a)
                next_L = L_val - 1
                next_s_idx = S_TO_IDX.get((next_R, next_L), S_TO_IDX[(0, 0)])
                P_mat[a_idx, s_idx, next_s_idx] = 1.0
            else:
                empty_s_idx = S_TO_IDX[(0, 0)]
                for next_s_idx, prob in enumerate(arrival_dist):
                    if prob > 0:
                        # 只有这里的 prob 是真正的概率，乘出来才不会大于 1
                        P_mat[a_idx, s_idx, next_s_idx] += arrival_prob * prob
                P_mat[a_idx, s_idx, empty_s_idx] += (1.0 - arrival_prob)

    return P_mat, R_mat, actual_A_mat

# --- 4. Numba 核心加速区 ---
@njit(fastmath=True)
def run_pvi_numba(nu, T, P_mat, R_mat, actions_arr, V_init, actual_A_mat):
    num_states = R_mat.shape[1]
    num_actions = R_mat.shape[2]

    V = V_init.copy()
    V_new = np.zeros_like(V)

    epsilon = 1e-10
    max_iter = 50000

    R_tilde = np.zeros_like(R_mat)
    for t in range(T):
        for s in range(num_states):
            for a in range(num_actions):
                # 【核心修改 2】：使用 actual_A_mat 扣除惩罚
                R_tilde[t, s, a] = R_mat[t, s, a] - nu * actual_A_mat[s, a]

    for k in range(max_iter):
        diff = 0.0
        # Gauss-Seidel 倒推
        for t in range(T - 1, -1, -1):
            for s in range(num_states):
                max_q = -1e20
                for a in range(num_actions):
                    ev_next = 0.0
                    for s_next in range(num_states):
                        prob = P_mat[t, a, s, s_next]
                        if prob > 0:
                            if t == T - 1:
                                ev_next += prob * V[s_next, 0]
                            else:
                                ev_next += prob * V_new[s_next, t + 1]

                    q_val = R_tilde[t, s, a] + ev_next
                    if q_val > max_q:
                        max_q = q_val

                V_new[s, t] = max_q

        # ==================== 真正的 RVI ====================
        ref_val = V_new[0, 0]

        for t in range(T):
            for s in range(num_states):
                V_new[s, t] = V_new[s, t] - ref_val
                d = abs(V_new[s, t] - V[s, t])
                if d > diff:
                    diff = d

        for t in range(T):
            for s in range(num_states):
                V[s, t] = V_new[s, t]

        if diff < epsilon:
            break

    return V


@njit(fastmath=True)
def get_q_diff_numba(s_idx, t, mode_k, nu, V_converged, T, P_mat, R_mat, actions_arr, actual_A_mat):
    t_next = (t + 1) % T
    num_states = R_mat.shape[1]

    # 动作 k
    a1 = mode_k
    # 【核心修改 3】：使用 actual_A_mat 扣除惩罚
    r1 = R_mat[t, s_idx, a1] - nu * actual_A_mat[s_idx, a1]
    ev1 = 0.0
    for sn in range(num_states):
        p = P_mat[t, a1, s_idx, sn]
        if p > 0:
            ev1 += p * V_converged[sn, t_next]
    q1 = r1 + ev1

    # 动作 k + 1
    a2 = mode_k + 1
    # 【核心修改 3】：使用 actual_A_mat 扣除惩罚
    r2 = R_mat[t, s_idx, a2] - nu * actual_A_mat[s_idx, a2]
    ev2 = 0.0
    for sn in range(num_states):
        p = P_mat[t, a2, s_idx, sn]
        if p > 0:
            ev2 += p * V_converged[sn, t_next]
    q2 = r2 + ev2

    # diff = Q(k) - Q(k+1)
    return q1 - q2


@njit(parallel=True)
def compute_index_table_parallel(T, NUM_STATES, MAX_CHARGE, P_mat, R_mat, actions_arr, actual_A_mat, S_arr):
    index_table = np.zeros((NUM_STATES, T, int(MAX_CHARGE)))

    # 扩大搜索范围以应对极端状态和可能为负数的 Index
    nu_min = 0.0
    nu_max = 50.0

    for s_idx in prange(NUM_STATES):
        V_cache = np.zeros((NUM_STATES, T))

        # 获取当前状态对应的 R_val 和 L_val
        R_val = S_arr[s_idx, 0]
        L_val = S_arr[s_idx, 1]

        for t in range(T):
            # 车辆不在时无需计算
            if L_val == 0:
                continue

            for k in range(int(MAX_CHARGE)):
                # 【核心修改 4】：直接过滤掉毫无意义的越界动作
                if k >= R_val:
                    index_table[s_idx, t, k] = 0.0
                    continue

                if k + 1 >= len(actions_arr):
                    index_table[s_idx, t, k] = -999.0
                    continue

                low = nu_min
                high = nu_max
                ans = high

                # 二分循环
                for _ in range(80):
                    mid = (low + high) / 2.0
                    V_mid = run_pvi_numba(mid, T, P_mat, R_mat, actions_arr, V_cache, actual_A_mat)

                    # 避免并行中的内存指派问题，使用切片赋值
                    V_cache[:] = V_mid

                    # 这里的 diff 是 Q(k) - Q(k+1)
                    diff = get_q_diff_numba(s_idx, t, k, mid, V_mid, T, P_mat, R_mat, actions_arr, actual_A_mat)

                    # 【核心修改 5】：精细的早停机制和正确的收缩方向
                    if abs(diff) < 1e-6:
                        ans = mid
                        break
                    elif diff > 0:
                        # Q(k) > Q(k+1)，说明惩罚力度过大，去左半区间找更小的惩罚
                        high = mid
                        ans = mid
                    else:
                        # Q(k) < Q(k+1)，说明惩罚力度过小，去右半区间找更大的惩罚
                        low = mid

                index_table[s_idx, t, k] = ans

    return index_table


# --- 5. 主流程 ---
if __name__ == '__main__':
    FILE_NAME = f"index_Poisson_Fast_penal={penalty_weight}_chen({max_r},{max_l}).npy"

    start_time = time.time()

    # 获取包含 actual_A_mat 的矩阵
    P_mat, R_mat, actual_A_mat = precompute_matrices_stationary(S_SPACE, A_SPACE)
    print(f"Matrix precomputation done in {time.time() - start_time:.2f}s")

    print(f"Starting optimized Whittle Index calculation for {NUM_STATES * period * int(MAX_CHARGE)} indices...")
    print("Compiling Numba functions (this takes a few seconds)...")

    # 将 S_SPACE 转换为 numpy 数组以供 Numba prange 并行调用
    S_arr = np.array(S_SPACE, dtype=np.int32)

    t0 = time.time()
    # 传入实际动作矩阵 actual_A_mat 和 状态数组 S_arr
    TABLE = compute_index_table_parallel(period, NUM_STATES, MAX_CHARGE, P_mat, R_mat, actions_arr, actual_A_mat, S_arr)
    t1 = time.time()

    print(f"\n--- Calculation Completed in {t1 - t0:.2f} seconds ---")

    np.save(FILE_NAME, TABLE)
    print(f"Saved to {FILE_NAME}")