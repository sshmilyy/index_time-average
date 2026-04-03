import numpy as np
import gurobipy as gp
from gurobipy import GRB

# 从你的文件导入必要的环境参数和仿真函数
from parameter_setting_CHEN106 import MAX_CHARGE, S_SPACE, A_SPACE, period, max_r, max_l, T
# 导入你上面写的真实仿真函数
from Performance_Evaluation_CHEN import generate_arrival_sequence_poi, run_experiments, f, delta_f
from r_beta import precompute_matrices
# 导入我们之前写的 LP 预计算和求解函数 (假设存在 r_beta.py 中)

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


if __name__ == "__main__":
    # 1. 设定参数
    power_ratio = 0.4
    penalty_weight = 0.6

    # 2. 加载你之前算好的 Index 表 (请确保路径和名字对应)
    file_name = f"index_cache_T24_penal{penalty_weight}.npy"
    print(f"Loading index file: {file_name}")
    INDEX_TABLE = np.load(file_name)

    # 3. 预计算 LP 需要的矩阵
    P_mat, R_mat, actions_arr = precompute_matrices(period, S_SPACE, A_SPACE)

    print("\n" + "=" * 70)
    print(f"VERIFYING ASYMPTOTIC OPTIMALITY (Power Ratio: {power_ratio})")
    print("=" * 70)

    # 我们测试不同的系统规模 N
    N_list = [200]

    for N in N_list:
        # 重写 N 相关的全局参数（因为原先 N 可能是全局常量，这里动态修改以适应测试）
        import Performance_Evaluation_CHEN

        Performance_Evaluation_CHEN.N = N  # 动态修改模块内的 N

        # 计算总功率和单桩限制
        total_power = round(N * max_r / max_l * power_ratio)
        avg_power_per_charger = total_power / N

        # --- A. 计算理论上的绝对上限 (LP Bound) ---
        lp_reward, lp_power, beta_star = solve_single_bandit_relaxation(
            P_mat, R_mat, actions_arr, avg_power_per_charger
        )

        # --- B. 运行真实的 Monte Carlo 多智能体仿真 ---
        NUM_SIMULATIONS = 50  # 为了平滑随机性，跑50次取平均
        EVAL_WINDOW = 200  # 取最后200步作为稳态
        sim_rewards_per_charger = []

        flat_state = tuple([0, 0] * N)
        for _ in range(NUM_SIMULATIONS):
            # 1. 生成 N 个桩的随机到达序列
            arrival_seq = generate_arrival_sequence_poi()
            # 2. 跑你写的 Index Policy 仿真
            avg_total_reward, _, _, _ = run_experiments(
                'index', arrival_seq, flat_state, total_power, table=INDEX_TABLE, eval_window=EVAL_WINDOW
            )
            # 3. 记录**平均分摊到每个桩**的收益，以便和单机 LP 对比
            sim_rewards_per_charger.append(avg_total_reward / N)

        mean_sim_reward = np.mean(sim_rewards_per_charger)

        # --- C. 打印对比结果 ---
        gap = abs(lp_reward - mean_sim_reward) / lp_reward * 100
        print(
            f"N = {N:<4} | Total Power = {total_power:<4} | LP Bound: {lp_reward:.4f} | Sim Index Reward: {mean_sim_reward:.4f} | Gap: {gap:.2f}%")