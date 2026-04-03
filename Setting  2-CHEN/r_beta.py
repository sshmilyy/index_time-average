import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

# 从你的参数文件导入依赖
from parameter_setting_CHEN106 import  S_SPACE, A_SPACE, period, max_r, max_l, \
    get_time_varying_p0, alpha, S_TO_IDX, get_time_varying_prob, r_dist, l_dist, r_p, l_p

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
    # (保持原样，省略细节以节省空间，你的原版代码直接用即可)
    num_states = len(S_space)
    num_actions = len(A_space)
    P_mat = np.zeros((T, num_actions, num_states, num_states))
    R_mat = np.zeros((T, num_states, num_actions))
    actions_arr = np.array(A_space, dtype=np.float64)

    print("Pre-computing P and R matrices...")
    for t in range(T):
        for s_idx, s in enumerate(S_space):
            for a_idx, a in enumerate(A_space):
                R_mat[t, s_idx, a_idx] = reward_function(s, a, t)
                R_val, L_val = s
                if L_val > 1:
                    next_R = max(R_val - a, 0)
                    next_L = L_val - 1
                    next_s = (next_R, next_L)
                    if next_s in S_TO_IDX:
                        next_idx = S_TO_IDX[next_s]
                        P_mat[t, a_idx, s_idx, next_idx] = 1.0
                else:
                    prob_arrival = get_time_varying_prob(t)
                    next_idx_0 = S_TO_IDX[(0, 0)]
                    P_mat[t, a_idx, s_idx, next_idx_0] += (1.0 - prob_arrival)
                    for r_idx, r_val in enumerate(r_dist):
                        for l_idx, l_val in enumerate(l_dist):
                            prob = prob_arrival * r_p[r_idx] * l_p[l_idx]
                            next_s = (r_val, l_val)
                            if next_s in S_TO_IDX:
                                next_idx = S_TO_IDX[next_s]
                                P_mat[t, a_idx, s_idx, next_idx] += prob
    return P_mat, R_mat, actions_arr


# --- 1. 原版受限 LP (Pr1问题) ---
def solve_single_bandit_relaxation(P_mat, R_mat, actions_arr, avg_power_per_charger):
    T = P_mat.shape[0]
    num_actions = P_mat.shape[1]
    num_states = P_mat.shape[2]

    model = gp.Model("Single_Bandit_Relaxation")
    model.setParam("OutputFlag", 0)
    x = model.addVars(T, num_states, num_actions, vtype=GRB.CONTINUOUS, lb=0.0, name="x")

    obj_expr = gp.quicksum(
        x[t, s, a] * R_mat[t, s, a] for t in range(T) for s in range(num_states) for a in range(num_actions)) / T
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # 约束 1: 流转平衡
    for t in range(T):
        t_prev = (t - 1) % T
        for s in range(num_states):
            prob_s_at_t = gp.quicksum(x[t, s, a] for a in range(num_actions))
            prob_flow_in = gp.quicksum(
                x[t_prev, s_prime, a_prime] * P_mat[t_prev, a_prime, s_prime, s]
                for s_prime in range(num_states) for a_prime in range(num_actions)
                if P_mat[t_prev, a_prime, s_prime, s] > 0
            )
            model.addConstr(prob_s_at_t == prob_flow_in, name=f"flow_{t}_{s}")

    # 约束 2: 归一化
    model.addConstr(gp.quicksum(x[0, s, a] for s in range(num_states) for a in range(num_actions)) == 1.0,
                    name="normalization")

    # 约束 3: 功率上限
    avg_power_expr = gp.quicksum(
        x[t, s, a] * actions_arr[a] for t in range(T) for s in range(num_states) for a in range(num_actions)) / T
    capacity_constr = model.addConstr(avg_power_expr <= avg_power_per_charger, name="capacity_constraint")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        optimal_reward = model.ObjVal
        actual_power = avg_power_expr.getValue()
        # 【关键修改】：提取功率约束的对偶变量（即最佳影子价格 β*）
        beta_star = abs(capacity_constr.Pi)
        return optimal_reward, actual_power, beta_star
    else:
        raise ValueError("LP failed!")


# --- 2. 新增：无约束的 W(β) 问题求解器 ---
def solve_W_beta_problem(P_mat, R_mat, actions_arr, beta_star):
    """
    求解无约束的 W(β) 问题。
    此时没有平均功率约束了，但是执行动作 a 会在目标函数中扣除 beta_star * a 的费用。
    """
    T = P_mat.shape[0]
    num_actions = P_mat.shape[1]
    num_states = P_mat.shape[2]

    model = gp.Model("W_Beta_Problem")
    model.setParam("OutputFlag", 0)
    y = model.addVars(T, num_states, num_actions, vtype=GRB.CONTINUOUS, lb=0.0, name="y")

    # 【核心区别】：目标函数变成了 R - β*a
    obj_expr = gp.quicksum(
        y[t, s, a] * (R_mat[t, s, a] - beta_star * actions_arr[a])
        for t in range(T) for s in range(num_states) for a in range(num_actions)
    ) / T
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # 约束 1: 流转平衡 (同上)
    for t in range(T):
        t_prev = (t - 1) % T
        for s in range(num_states):
            prob_s_at_t = gp.quicksum(y[t, s, a] for a in range(num_actions))
            prob_flow_in = gp.quicksum(
                y[t_prev, s_prime, a_prime] * P_mat[t_prev, a_prime, s_prime, s]
                for s_prime in range(num_states) for a_prime in range(num_actions)
                if P_mat[t_prev, a_prime, s_prime, s] > 0
            )
            model.addConstr(prob_s_at_t == prob_flow_in, name=f"flow_{t}_{s}")

    # 约束 2: 归一化 (同上)
    model.addConstr(gp.quicksum(y[0, s, a] for s in range(num_states) for a in range(num_actions)) == 1.0,
                    name="normalization")

    # ！！注意：没有功率约束了 ！！

    model.optimize()

    if model.status == GRB.OPTIMAL:
        # 还原出这个策略下，原始真实的 Reward 和 真实的功率消耗
        original_reward = sum(
            y[t, s, a].X * R_mat[t, s, a] for t in range(T) for s in range(num_states) for a in range(num_actions)) / T
        actual_power = sum(
            y[t, s, a].X * actions_arr[a] for t in range(T) for s in range(num_states) for a in range(num_actions)) / T
        return original_reward, actual_power
    else:
        raise ValueError("W(beta) LP failed!")


if __name__ == "__main__":
    P_mat, R_mat, actions_arr = precompute_matrices(period, S_SPACE, A_SPACE)

    # 取一个中间的功率配比进行详细对比
    for power_ratio in [0.3,0.4]:
        for N in [1,120,180,300,500,1000]:
            total_power = round(N * max_r / max_l * power_ratio)
            avg_power_per_charger = total_power / N

            print("\n" + "=" * 60)
            print(f"TESTING WITH POWER RATIO: {power_ratio} (Limit: {avg_power_per_charger:.3f},N={N})")
            print("=" * 60)

            # 1. 求解 Pr1(β) - 带有功率限制的原始 LP
            lp_reward, lp_power, beta_star = solve_single_bandit_relaxation(
                P_mat, R_mat, actions_arr, avg_power_per_charger
            )

            print(f"[1] Original Constrained LP (Pr1)")
            print(f"    -> Reward Limit: {lp_reward:.4f}")
            print(f"    -> Power Used  : {lp_power:.4f}")
            print(f"    -> Shadow Price (β*): {beta_star:.4f}")
            print("-" * 60)

            # 2. 求解 W(β) - 使用 β* 作为罚款的无约束 Index Policy
            w_beta_reward, w_beta_power = solve_W_beta_problem(
                P_mat, R_mat, actions_arr, beta_star
            )

            print(f"[2] Unconstrained W(β*) Problem (Index Policy acting on β*)")
            print(f"    -> Original Reward: {w_beta_reward:.4f}")
            print(f"    -> Power Used     : {w_beta_power:.4f}")
            # 把 W(beta) 因为功率没用对而产生的“差价”补回来
            math_check = w_beta_reward + beta_star * (avg_power_per_charger - w_beta_power)
            print(f"Math Check (Strong Duality): {math_check:.4f} (Should exactly equal LP Reward: {lp_reward:.4f})")
            print("=" * 60)