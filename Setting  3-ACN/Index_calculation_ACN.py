import numpy as np
import itertools
import sys
import os
import time
from numba import njit, prange  # 引入 Numba
from Parameter_Setting_ACN import MAX_CHARGE, r_dist, l_dist, r_p, l_p, beta, S_TO_IDX, alpha, S_SPACE, \
    A_SPACE, period, NUM_STATES, get_time_varying_p0, get_time_varying_prob,max_r,max_l

# --- 3. 辅助函数 (Python端预计算) ---
penalty_weight = 0.6


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


def precompute_matrices(T, S_space, A_space):
    # (保持不变，这部分只运行一次，耗时很短)
    num_states = len(S_space)
    num_actions = len(A_space)
    p_mat = np.zeros((T, num_actions, num_states, num_states))
    r_mat = np.zeros((T, num_states, num_actions))
    actions_arr = np.array(A_space, dtype=np.float64)

    print("Pre-computing P and R matrices...")
    for t in range(T):
        for s_idx, s in enumerate(S_space):
            for a_idx, a in enumerate(A_space):
                r_mat[t, s_idx, a_idx] = reward_function(s, a, t)
                R_val, L_val = s
                if L_val > 0:
                    if L_val == 1:
                        next_s = (0, 0)
                    else:
                        next_R = max(R_val - a, 0)
                        next_L = L_val - 1
                        next_s = (next_R, next_L)
                    if next_s in S_TO_IDX:
                        next_idx = S_TO_IDX[next_s]
                        p_mat[t, a_idx, s_idx, next_idx] = 1.0
                else:
                    lambda_t = get_time_varying_prob(t)
                    prob_no_arrival = np.exp(-lambda_t)

                    next_idx_0 = S_TO_IDX[(0, 0)]
                    p_mat[t, a_idx, s_idx, next_idx_0] += prob_no_arrival
                    for r_idx, r_val in enumerate(r_dist):
                        for l_idx, l_val in enumerate(l_dist):
                            prob = (1-prob_no_arrival) * r_p[r_idx] * l_p[l_idx]
                            next_s = (r_val, l_val)
                            if next_s in S_TO_IDX:
                                next_idx = S_TO_IDX[next_s]
                                p_mat[t, a_idx, s_idx, next_idx] += prob
    return p_mat, r_mat, actions_arr


# --- 4. Numba 核心加速区 ---
# 将原本 Python 的 PVI 和 Binary Search 全部移入 Numba 环境

@njit(fastmath=True)
def run_pvi_numba(nu, T, beta, P_mat, R_mat, actions_arr, V_init):
    """
    Numba 加速版的周期性值迭代
    """
    num_states = R_mat.shape[1]
    num_actions = R_mat.shape[2]

    # 初始化 V
    V = V_init.copy()
    V_new = np.zeros_like(V)

    epsilon = 1e-4
    max_iter = 2000

    # 预计算 R_tilde: (T, S, A)
    # R_mat[t, s, a] - nu * a
    R_tilde = np.zeros_like(R_mat)
    for t in range(T):
        for s in range(num_states):
            for a in range(num_actions):
                R_tilde[t, s, a] = R_mat[t, s, a] - nu * actions_arr[a]

    for k in range(max_iter):
        diff = 0.0
        # 逆序时间更新
        for t in range(T - 1, -1, -1):
            t_next = (t + 1) % T

            # 对每个状态 s 计算最佳动作价值
            for s in range(num_states):
                max_q = -1e20  # Negative infinity

                # 遍历动作 a 寻找 max Q
                for a in range(num_actions):
                    # 计算期望未来价值 E[V_next]
                    # dot(P[t, a, s, :], V[:, t_next])
                    ev_next = 0.0
                    for s_next in range(num_states):
                        prob = P_mat[t, a, s, s_next]
                        if prob > 0:
                            ev_next += prob * V[s_next, t_next]

                    q_val = R_tilde[t, s, a] + beta * ev_next
                    if q_val > max_q:
                        max_q = q_val

                V_new[s, t] = max_q

        # 检查收敛
        for t in range(T):
            for s in range(num_states):
                d = abs(V_new[s, t] - V[s, t])
                if d > diff:
                    diff = d

        # 更新 V
        for t in range(T):
            for s in range(num_states):
                V[s, t] = V_new[s, t]

        if diff < epsilon:
            break

    return V


@njit(fastmath=True)
def get_q_diff_numba(s_idx, t, mode_k, nu, V_converged, T, beta, P_mat, R_mat, actions_arr):
    """
    计算 Q(s, mode_k) - Q(s, mode_k+1)
    """
    t_next = (t + 1) % T
    num_states = R_mat.shape[1]

    # 动作 k
    a1 = mode_k
    r1 = R_mat[t, s_idx, a1] - nu * actions_arr[a1]
    ev1 = 0.0
    for sn in range(num_states):
        p = P_mat[t, a1, s_idx, sn]
        if p > 0:
            ev1 += p * V_converged[sn, t_next]
    q1 = r1 + beta * ev1

    # 动作 k + 1
    a2 = mode_k + 1
    r2 = R_mat[t, s_idx, a2] - nu * actions_arr[a2]
    ev2 = 0.0
    for sn in range(num_states):
        p = P_mat[t, a2, s_idx, sn]
        if p > 0:
            ev2 += p * V_converged[sn, t_next]
    q2 = r2 + beta * ev2

    return q1 - q2


@njit(parallel=True)
def compute_index_table_parallel(T, NUM_STATES, MAX_CHARGE, P_mat, R_mat, actions_arr, beta):
    """
    并行计算所有状态的索引
    """
    # 结果容器
    index_table = np.zeros((NUM_STATES, T, int(MAX_CHARGE)))

    # 参数范围
    nu_min = 0.0
    nu_max = 20.0
    tol = 1e-4

    # 扁平化循环以便并行: 状态 * 时间
    # 我们不对 MAX_CHARGE 并行，因为同一状态的二分查找通常可以复用 V（虽然这里为了简单每次重算，但并行外层足够了）
    # prange 使得这个循环在多核 CPU 上并行执行
    for s_idx in prange(NUM_STATES):
        # 初始化 V 缓存 (每个线程一份)
        V_cache = np.zeros((NUM_STATES, T))

        for t in range(T):
            if s_idx == 0:
                # (0,0) 状态跳过，保持 0
                continue

            for k in range(int(MAX_CHARGE)):
                # 检查 k+1 是否越界
                if k + 1 >= len(actions_arr):
                    index_table[s_idx, t, k] = -999.0
                    continue

                # --- 二分查找开始 ---
                low = nu_min
                high = nu_max

                # 预检 nu=0
                V_0 = run_pvi_numba(0.0, T, beta, P_mat, R_mat, actions_arr, V_cache)
                diff_0 = get_q_diff_numba(s_idx, t, k, 0.0, V_0, T, beta, P_mat, R_mat, actions_arr)

                if diff_0 > 0:
                    # 如果 nu=0 时 Q(k) > Q(k+1)，说明即使没有补贴也更喜欢 k，
                    # 按照 Whittle 索引定义，此时 index <= 0
                    index_table[s_idx, t, k] = 0.0
                    continue

                ans = high

                # 二分循环
                # 这里为了速度，我们假设函数是单调的。
                # 实际上可以只跑 15-20 次迭代即可达到精度
                for _ in range(20):
                    mid = (low + high) / 2.0
                    # Warm start: 使用上一次的 V_cache 作为起点 (V_0 或 上次迭代结果)
                    V_mid = run_pvi_numba(mid, T, beta, P_mat, R_mat, actions_arr, V_cache)
                    # 更新 cache 以加速下一次 PVI
                    V_cache = V_mid

                    diff = get_q_diff_numba(s_idx, t, k, mid, V_mid, T, beta, P_mat, R_mat, actions_arr)

                    if diff > 0:
                        # Q(k) > Q(k+1)，说明补贴 mid 足够大，让 k 变得比 k+1 有吸引力
                        # 我们需要更小的补贴来找到平衡点？
                        # 等等，通常 Whittle Index 定义是：使得动作 k 和 k+1 无差别的补贴 nu
                        # Q_nu(k) = Q_nu(k+1)
                        # 如果 diff > 0 (Q_k > Q_k+1)，说明当前 nu 下 k 优于 k+1。
                        # 通常 nu 是"充电的代价"还是"补贴"？
                        # 你的代码中: Reward = U - nu * a. nu 越大，a 越大惩罚越大。
                        # 所以 nu 很大时，我们倾向于小的 action (k)。
                        # nu 很小时，我们倾向于大的 action (k+1)。
                        # 如果 diff > 0 (Q_k > Q_k+1)，说明小动作 k 优于大动作 k+1。这说明 nu 太大了（惩罚太重）。
                        # 我们需要减小 nu。
                        high = mid
                        ans = mid
                    else:
                        # Q_k < Q_k+1，大动作更好，说明 nu 太小（惩罚不够）。
                        low = mid

                index_table[s_idx, t, k] = ans

    return index_table


# --- 5. 主流程 ---

if __name__ == '__main__':
    FILE_NAME = f"Index_penal={penalty_weight}_ACN({max_r},{max_l})max=6.npy"

    start_time = time.time()

    # 1. 预计算矩阵 (Python)
    P_mat, R_mat, actions_arr = precompute_matrices(period, S_SPACE, A_SPACE)
    print(f"Matrix precomputation done in {time.time() - start_time:.2f}s")

    # 2. 调用 Numba 并行计算
    print(f"Starting optimized Whittle Index calculation for {NUM_STATES * period * int(MAX_CHARGE)} indices...")
    print("Compiling Numba functions (this takes a few seconds)...")

    # 这里的第一次调用会触发 JIT 编译
    t0 = time.time()
    TABLE = compute_index_table_parallel(period, NUM_STATES, MAX_CHARGE, P_mat, R_mat, actions_arr, beta)
    t1 = time.time()

    print(f"\n--- Calculation Completed in {t1 - t0:.2f} seconds ---")

    # 保存
    np.save(FILE_NAME, TABLE)
    print(f"Saved to {FILE_NAME}")