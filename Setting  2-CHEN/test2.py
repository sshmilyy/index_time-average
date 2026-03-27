import numpy as np
import time
from numba import njit, prange

# ---------------------------------------------------------
# 1. 导入环境参数 (请确保从您的 parameter_setting 文件中正确导入)
# ---------------------------------------------------------
from parameter_setting_CHEN106 import (
    MAX_CHARGE,
    max_r,
r_p,l_p,
    max_l,
    S_TO_IDX,
    alpha,
    S_SPACE,
    A_SPACE,
    NUM_STATES,
    get_time_varying_p0,
    get_time_varying_prob,
    r_dist,
    l_dist
)

penalty_weight = 0.8


def f(x):
    """到达 Deadline 时的未充满惩罚函数"""
    return (x ** 2) * penalty_weight


# ---------------------------------------------------------
# 2. 降维后的环境矩阵预计算 (Stationary T=1)
# ---------------------------------------------------------
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
                next_R = 0
                next_L = 0
                next_s_idx = S_TO_IDX.get((next_R, next_L), S_TO_IDX[(0, 0)])
                P_mat[a_idx, s_idx, next_s_idx] = 1.0
                # empty_s_idx = S_TO_IDX[(0, 0)]
                # for next_s_idx, prob in enumerate(arrival_dist):
                #     if prob > 0:
                #         # 只有这里的 prob 是真正的概率，乘出来才不会大于 1
                #         P_mat[a_idx, s_idx, next_s_idx] += arrival_prob * prob
                # P_mat[a_idx, s_idx, empty_s_idx] += (1.0 - arrival_prob)

    return P_mat, R_mat, actual_A_mat

# ---------------------------------------------------------
# 3. Numba 加速的数值迭代求解器 (Numerical)
# ---------------------------------------------------------
@njit
def run_rvi_stationary_numba(nu, P_mat, R_mat, actions_arr, actual_A_mat):
    num_states = P_mat.shape[1]
    V = np.zeros(num_states)
    V_new = np.zeros(num_states)

    R_tilde = np.zeros((num_states, len(actions_arr)))
    for s in range(num_states):
        for a in range(len(actions_arr)):
            R_tilde[s, a] = R_mat[s, a] - nu * actual_A_mat[s, a]

    diff = 1.0
    epsilon = 1e-6
    max_iter = 50000
    iters = 0

    while diff > epsilon and iters < max_iter:
        for s in range(num_states):
            max_val = -1e9
            for a_idx in range(len(actions_arr)):
                ev_next = 0.0
                for next_s in range(num_states):
                    prob = P_mat[a_idx, s, next_s]
                    if prob > 0:
                        ev_next += prob * V[next_s]

                q_val = R_tilde[s, a_idx] + ev_next
                if q_val > max_val:
                    max_val = q_val

            V_new[s] = max_val

        ref_val = V_new[0]
        diff = 0.0
        for s in range(num_states):
            V_new[s] -= ref_val
            d = abs(V_new[s] - V[s])
            if d > diff:
                diff = d

        for s in range(num_states):
            V[s] = V_new[s]

        iters += 1

    return V


@njit
def get_q_diff_stationary_numba(s, k, nu, P_mat, R_mat, actions_arr, actual_A_mat, V):
    num_states = P_mat.shape[1]
    Q_vals = np.zeros(2)
    test_actions = np.array([k, k + 1])

    for idx, a_idx in enumerate(test_actions):
        r_mod = R_mat[s, a_idx] - nu * actual_A_mat[s, a_idx]
        ev_next = 0.0
        for next_s in range(num_states):
            prob = P_mat[a_idx, s, next_s]
            if prob > 0:
                ev_next += prob * V[next_s]
        Q_vals[idx] = r_mod + ev_next

    return Q_vals[0] - Q_vals[1]


@njit(parallel=True)
def compute_index_table_numerical_parallel(num_states, max_charge, P_mat, R_mat, actions_arr, actual_A_mat, S_arr):
    index_table = np.zeros((num_states, int(max_charge)))

    for s_idx in prange(num_states):
        R_val = S_arr[s_idx, 0]
        L_val = S_arr[s_idx, 1]

        if L_val == 0 or R_val == 0:
            for k in range(int(max_charge)):
                index_table[s_idx, k] = 0.0
            continue

        for k in range(int(max_charge)):
            if k >= R_val:
                index_table[s_idx, k] = 0.0
                continue
            if k + 1 >= len(actions_arr):
                index_table[s_idx, k] = -999.0
                continue

            low, high = -10.0, 50.0
            ans = 0.0

            for _ in range(80):
                mid = (low + high) / 2.0
                V_converged = run_rvi_stationary_numba(mid, P_mat, R_mat, actions_arr, actual_A_mat)
                diff = get_q_diff_stationary_numba(s_idx, k, mid, P_mat, R_mat, actions_arr, actual_A_mat, V_converged)

                if abs(diff) < 1e-6:
                    ans = mid
                    break
                elif diff > 0:
                    high = mid
                    ans = mid
                else:
                    low = mid

            index_table[s_idx, k] = ans

    return index_table


# ---------------------------------------------------------
# 4. 理论推导的闭式解 (Theoretical)
# ---------------------------------------------------------
def compute_theoretical_index_table_stationary(S_space, max_charge, A_space):
    index_table = np.zeros((len(S_space), int(max_charge)))
    W = int(max_charge)
    p_constant = get_time_varying_p0(0)

    def delta_F(x):
        if x <= 0:
            return 0.0
        return ((x ** 2) * penalty_weight) - (((x - 1) ** 2) * penalty_weight)

    for s_idx, s in enumerate(S_space):
        R_val, L_val = s
        if L_val == 0 or R_val == 0:
            continue

        for k in range(W):
            i = k
            if i >= R_val:
                index_table[s_idx, k] = 0.0
                continue
            if i + 1 >= len(A_space):
                index_table[s_idx, k] = -999.0
                continue

            X = R_val - (L_val - 1) * W

            if X > 0:
                if i < X:
                    ans = alpha - p_constant + delta_F(X - i)
                else:
                    ans = alpha - p_constant
            else:
                ans = alpha - p_constant

            index_table[s_idx, k] = ans

    return index_table

