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
if __name__ == "__main__":
    from charging_env import ChargingEnv
    from Index_calculation import WhittleSolver
    print("✅ 测试: WhittleSolver")

    test_env = ChargingEnv(N=10, power_ratio=0.4, penalty_weight=0.8, T=50)
    solver = WhittleSolver(test_env)

    print(f"正在为 T={test_env.T}, Penalty={test_env.penalty_weight} 计算 Index 表...")
    # table = solver.solve() # 取消注释以运行你的求解代码
    print("求解器初始化正常，测试完成！")