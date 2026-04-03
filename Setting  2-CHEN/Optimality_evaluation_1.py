import numpy as np
import pandas as pd
import parameter_setting_CHEN106_1 as ps
from charging_env import ChargingEnv
from r_beta_1 import solve_single_bandit_relaxation
from Index_calculation import WhittleSolver
from Performance_Evaluation_CHEN_1 import generate_arrival_sequence_poi, run_experiments

if __name__ == "__main__":
    print("🚀 开始运行全面渐进最优性测试 (Optimality Evaluation)")

    # 1. 设定测试网格
    N_list = [10, 50, 100]
    penalty_weights_to_test = [0.4]
    power_ratios_to_test = [0.5]

    all_results = []

    for pw in penalty_weights_to_test:
        for pr in power_ratios_to_test:
            for N in N_list:
                print(f"\n{'=' * 50}")
                print(f"⚙️ 当前设定: N={N}, Penalty={pw}, Power_Ratio={pr}")

                # 2. 动态生成环境对象
                env = ChargingEnv(N=N, power_ratio=pr, penalty_weight=pw, T=500)

                # 3. 求解 LP 理论上限
                P_mat, R_mat = env.precompute_matrices()
                lp_reward, _, beta_star = solve_single_bandit_relaxation(
                    P_mat, R_mat, env.avg_power)

                # 4. 求解/加载 Whittle Index
                index_file = f"index_cache_T{env.T}_pw={env.penalty_weight}.npy"
                solver = WhittleSolver(env)
                try:
                    INDEX_TABLE = np.load(index_file)
                    print(f"已加载已有 Index 表: {index_file}")
                except FileNotFoundError:
                    print("计算新的 Index 表中...")
                    INDEX_TABLE = solver.get_index_table() # 填好 solver1.py 后取消注释
                    np.save(index_file, INDEX_TABLE)
                    INDEX_TABLE = None  # 临时占位

                # 5. 运行仿真验证
                NUM_SIMULATIONS = 50
                sim_rewards = []
                for _ in range(NUM_SIMULATIONS):
                    arr_seq = generate_arrival_sequence_poi(env)
                    init_s = tuple([0, 0] * env.N)

                    sim_total, _, _, _ = run_experiments(
                        'index', arr_seq, init_s, env, table=INDEX_TABLE
                    )
                    sim_rewards.append(sim_total / env.N)  # 折算单桩收益

                mean_sim_reward = np.mean(sim_rewards)
                gap = abs(lp_reward - mean_sim_reward) / lp_reward * 100 if lp_reward else 0

                print(f"✅ 结果 -> LP Bound: {lp_reward} | Sim: {mean_sim_reward} | Gap: {gap:.2f}%")

                all_results.append({
                    "N": N, "Penalty": pw, "PowerRatio": pr,
                    "LP_Bound": lp_reward, "Sim_Reward": mean_sim_reward, "Gap(%)": gap
                })

    # 6. 保存为 CSV 给论文画图用
    df = pd.DataFrame(all_results)
    df.to_csv("Asymptotic_Optimality_Results.csv", index=False)
    print("\n🎉 所有实验跑完！结果已存入 Asymptotic_Optimality_Results.csv")