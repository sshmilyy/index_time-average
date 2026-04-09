# r_beta.py
import gurobipy as gp
from gurobipy import GRB
import parameter_setting_CHEN106_1 as ps


def solve_single_bandit_relaxation(P_mat, R_mat, avg_power_per_charger):
    """
    求解 Pr1(β) - 带有功率限制的原始 LP
    """
    T = P_mat.shape[0]
    num_actions = P_mat.shape[1]
    num_states = P_mat.shape[2]

    model = gp.Model("Single_Bandit_Relaxation")
    model.setParam("OutputFlag", 0)  # 关闭求解器输出以保持控制台整洁

    # 决策变量：x[t, s, a]
    x = model.addVars(T, num_states, num_actions, vtype=GRB.CONTINUOUS, lb=0.0, name="x")

    # 1. 目标函数：最大化平均收益
    obj_expr = gp.quicksum(
        x[t, s, a] * R_mat[t, s, a]
        for t in range(T) for s in range(num_states) for a in range(num_actions)
    ) / T
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # 2. 约束条件 A: 稳态流转平衡 (Flow Balance)
    for t in range(T):
        t_prev = (t - 1) % T
        for s in range(num_states):
            prob_s_at_t = gp.quicksum(x[t, s, a] for a in range(num_actions))
            prob_flow_in = gp.quicksum(
                x[t_prev, s_prime, a_prime] * P_mat[t_prev, a_prime, s_prime, s]
                for s_prime in range(num_states) for a_prime in range(num_actions)
            )
            model.addConstr(prob_s_at_t == prob_flow_in, name=f"flow_{t}_{s}")

    # 3. 约束条件 B: 概率归一化 (Normalization)
    # 只需要在 t=0 时约束总和为 1，加上流转平衡，后续自然为 1
    model.addConstr(
        gp.quicksum(x[0, s, a] for s in range(num_states) for a in range(num_actions)) == 1.0,
        name="normalization"
    )

    # 4. 约束条件 C: 充电功率上限 (Capacity Constraint)
    # 【注意看这里！用 ps.A_SPACE[a] 完美替代了原来的 actions_arr[a]】
    avg_power_expr = gp.quicksum(
        x[t, s, a] * ps.A_SPACE[a]
        for t in range(T) for s in range(num_states) for a in range(num_actions)
    ) / T

    capacity_constr = model.addConstr(
        avg_power_expr <= avg_power_per_charger,
        name="capacity_constraint"
    )

    # 求解模型
    model.optimize()

    if model.status == GRB.OPTIMAL:
        optimal_reward = model.ObjVal
        actual_power = avg_power_expr.getValue()
        # 提取影子价格 (拉格朗日乘子 beta*)
        beta_star = abs(capacity_constr.Pi)

        return optimal_reward, actual_power, beta_star
    else:
        print("求解失败！模型可能不可行。")
        return None, None, None


# ==========================================
# 独立测试模块
# ==========================================
# ==========================================
# 独立测试模块：测试 LP 松弛上限与影子价格
# ==========================================
if __name__ == "__main__":
    import time
    from charging_env import ChargingEnv

    print("✅ 开始独立测试 r_beta_1.py (单臂拉格朗日松弛 LP求解)...")

    # 1. 设定测试参数
    test_N = 10
    test_power_ratio = 0.6
    test_penalty = 0.8
    test_T = 960  # 使用标准的 500 步或者周期 24，取决于你的 Env 设置

    env = ChargingEnv(N=test_N, power_ratio=test_power_ratio, penalty_weight=test_penalty, T=test_T)

    # 2. 获取预计算矩阵
    start_time = time.time()
    P_mat, R_mat = env.precompute_matrices()
    start_solve = time.time()
    lp_reward, actual_power, beta_star = solve_single_bandit_relaxation(
        P_mat, R_mat, env.avg_power
    )
    solve_time = time.time() - start_solve

    # 4. 打印学术风的输出结果
    print("\n" + "=" * 50)
    print(f"\n⚙️测试环境：N={test_N}, Power_Ratio={test_power_ratio}, Penalty={test_penalty}")
    if lp_reward is not None:
        print(f"🎉 求解成功！(Gurobi 耗时: {solve_time:.3f}s)")
        print("-" * 50)
        print(f"🏆 LP 理论稳态上限收益 (Optimal Reward): {lp_reward:.4f}")
        print(f"⚡ 实际分配的平均功率   (Actual Power)  : {actual_power:.4f} (硬限制为 <= {env.avg_power:.4f})")
        print(f"🔑 影子价格 / Beta* (Lagrange Mult): {beta_star:.4f}")
    else:
        print("❌ 求解失败，Gurobi 返回不可行 (Infeasible)。")
        print("   -> 请检查参数设置，或者确认预计算矩阵流转概率是否和为 1。")
    print("=" * 50 + "\n")