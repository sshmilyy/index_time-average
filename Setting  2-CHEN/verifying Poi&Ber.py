# verify_arrivals_multi_policy.py
import numpy as np
import time
import parameter_setting_CHEN106_1 as ps
from charging_env import ChargingEnv
from Performance_Evaluation_CHEN_1 import run_experiments
from Index_calculation import WhittleSolver  # 引入 Index 求解器
import pandas as pd  # 引入 pandas 处理表格和导出 Excel

# ==========================================
# 1. 到达生成器 (保持不变)
# ==========================================
def generate_bernoulli_arrivals(env):
    arrivals = []
    for t in range(env.T):
        prob = ps.get_time_varying_prob(t)
        arrivals_at_t = []
        for _ in range(env.N):
            if np.random.rand() < prob:
                r = np.random.choice(ps.r_dist, p=ps.r_p)
                l = np.random.choice(ps.l_dist, p=ps.l_p)
                arrivals_at_t.append((r, l))
            else:
                arrivals_at_t.append(None)
        arrivals.append(arrivals_at_t)
    return arrivals


def generate_poisson_arrivals(env):
    arrivals = []
    for t in range(env.T):
        prob = ps.get_time_varying_prob(t)
        expected_lambda = env.N * prob
        total_cars = np.random.poisson(expected_lambda)
        actual_cars = min(total_cars, env.N)

        arrivals_at_t = [None] * env.N
        if actual_cars > 0:
            chosen_indices = np.random.choice(env.N, actual_cars, replace=False)
            for idx in chosen_indices:
                r = np.random.choice(ps.r_dist, p=ps.r_p)
                l = np.random.choice(ps.l_dist, p=ps.l_p)
                arrivals_at_t[idx] = (r, l)

        arrivals.append(arrivals_at_t)
    return arrivals


# ==========================================
# 2. 多策略核心验证模块
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting asymptotic equivalence verification of Poisson and Bernoulli arrivals...")

    # Simulation Parameters
    N_list = [10, 40,70, 100, 200, 300]
    POLICIES = ['gdy', 'llf', 'lrf', 'new', 'index']

    TEST_POWER_RATIO = 0.8
    TEST_PENALTY = 0.8
    EVAL_WINDOW = 120
    NUM_SIMULATIONS = 20
    '''
    for N in N_list:
        print(f"\n" + "=" * 80)
        print(f"⚙️ System Size: N = {N}")
        print("=" * 80)

        # 1. 创建环境
        env = ChargingEnv(N=N, power_ratio=TEST_POWER_RATIO, penalty_weight=TEST_PENALTY)
        init_s = tuple([0, 0] * env.N)

        # 2. 准备 Whittle Index 表 (如果策略池里有 'index')
        INDEX_TABLE = None
        if 'index' in POLICIES:
            solver = WhittleSolver(env)
            INDEX_TABLE = solver.get_index_table()  # 自动计算或读取缓存

        print(f"{'Policy':<12} | {'Bernoulli avg Reward':<20} | {'Poisson avg Reward':<20} | {'Difference':<12}| {'Gap':<10}")
        print("-" * 85)

        # 3. 遍历每一个策略进行对比
        for alg in POLICIES:
            bernoulli_rewards = []
            poisson_rewards = []

            for _ in range(NUM_SIMULATIONS):
                # --- 测试 A: Bernoulli ---
                arr_seq_bern = generate_bernoulli_arrivals(env)
                kwarg_table = {'table': INDEX_TABLE} if alg == 'index' else {}

                reward_bern, _, _, _ = run_experiments(
                    algorithm=alg, arrival_seq=arr_seq_bern,
                    initial=init_s, env=env, eval_window=EVAL_WINDOW,
                    **kwarg_table  # 如果是 index 策略，自动传入 table
                )
                bernoulli_rewards.append(reward_bern / env.N)

                # --- 测试 B: Poisson ---
                arr_seq_pois = generate_poisson_arrivals(env)
                reward_pois, _, _, _ = run_experiments(
                    algorithm=alg, arrival_seq=arr_seq_pois,
                    initial=init_s, env=env, eval_window=EVAL_WINDOW,
                    **kwarg_table
                )
                poisson_rewards.append(reward_pois / env.N)

            # 4. 计算当前策略在两种到达模式下的差异
            mean_bern = np.mean(bernoulli_rewards)
            mean_pois = np.mean(poisson_rewards)
            diff = abs(mean_bern - mean_pois)
            gap = abs(mean_bern - mean_pois) / mean_bern * 100 if mean_bern != 0 else 0

            print(f"{alg.upper():<12} | {mean_bern:<20.4f} | {mean_pois:<20.4f} | {diff:<11.4f} | {gap:.2f}%")
    '''

    # --- NEW: Pre-compute Index Tables for all N to save massive time ---
    index_tables_cache = {}
    if 'index' in POLICIES:
        print("\n⏳ Pre-computing Whittle Index tables for all N (This happens only once)...")
        for N in N_list:
            temp_env = ChargingEnv(N=N, power_ratio=TEST_POWER_RATIO, penalty_weight=TEST_PENALTY)
            solver = WhittleSolver(temp_env)
            index_tables_cache[N] = solver.get_index_table(                                                                                                                                       4)
            print(f"   [OK] Index table for N={N} ready.")

    # List to collect all experiment records for Excel
    all_records = []

    # --- Phase 1: Loop over Policies first, then N ---
    for alg in POLICIES:
        alg_upper = alg.upper()
        print(f"\n" + "=" * 80)
        print(f"⏳ Running simulations for Policy: {alg_upper} ...")
        start_time_alg = time.time()

        alg_records = []  # Temporarily hold records for the current policy

        for N in N_list:
            env = ChargingEnv(N=N, power_ratio=TEST_POWER_RATIO, penalty_weight=TEST_PENALTY)
            init_s = tuple([0, 0] * env.N)

            # Fetch the pre-computed table from cache if needed
            kwarg_table = {'table': index_tables_cache[N]} if alg == 'index' else {}

            bernoulli_rewards = []
            poisson_rewards = []

            for _ in range(NUM_SIMULATIONS):
                # Bernoulli Test
                arr_seq_bern = generate_bernoulli_arrivals(env)
                reward_bern, _, _, _ = run_experiments(
                    algorithm=alg, arrival_seq=arr_seq_bern,
                    initial=init_s, env=env, eval_window=EVAL_WINDOW, **kwarg_table
                )
                bernoulli_rewards.append(reward_bern / env.N)

                # Poisson Test
                arr_seq_pois = generate_poisson_arrivals(env)
                reward_pois, _, _, _ = run_experiments(
                    algorithm=alg, arrival_seq=arr_seq_pois,
                    initial=init_s, env=env, eval_window=EVAL_WINDOW, **kwarg_table
                )
                poisson_rewards.append(reward_pois / env.N)

            mean_bern = np.mean(bernoulli_rewards)
            mean_pois = np.mean(poisson_rewards)

            # Calculate gap based on this policy's own reward
            abs_gap = abs(mean_bern - mean_pois)
            local_max_reward = max(abs(mean_bern), abs(mean_pois))
            # Add a small epsilon (1e-4) to prevent division by zero for policies with ~0 reward
            rel_gap = (abs_gap / local_max_reward) * 100

            record = {
                'Policy': alg_upper,
                'N': N,
                'Bernoulli_Reward': mean_bern,
                'Poisson_Reward': mean_pois,
                'Abs_Gap': abs_gap,
                'Rel_Gap_pct': rel_gap
            }
            alg_records.append(record)
            all_records.append(record)

        # --- Print results immediately after this POLICY is finished ---
        print(f"✅ Results for Policy = {alg_upper} (Time taken: {time.time() - start_time_alg:.1f}s):")
        print(f"{'N':<8} | {'Bernoulli Reward':<18} | {'Poisson Reward':<18} | {'Abs Gap':<15} | {'Rel Gap (%)':<10}")
        print("-" * 80)

        for rec in alg_records:
            print(
                f"{int(rec['N']):<8} | {rec['Bernoulli_Reward']:<18.4f} | {rec['Poisson_Reward']:<18.4f} | {rec['Abs_Gap']:<15.4f} | {rec['Rel_Gap_pct']:.2f}%")

    # ==========================================
    # 3. Data Processing & Excel Export
    # ==========================================
    df = pd.DataFrame(all_records)

    # --- Export to Excel ---
    excel_filename = "Poisson_Bernoulli_Equivalence.xlsx"

    df_excel = df.rename(columns={
        'Policy': 'Policy',
        'N': 'System Scale (N)',
        'Bernoulli_Reward': 'Bernoulli Reward',
        'Poisson_Reward': 'Poisson Reward',
        'Abs_Gap': 'Absolute Gap',
        'Rel_Gap_pct': 'Relative Gap (%)'
    })

    # Sort for final Excel readability
    df_excel = df_excel.sort_values(by=['Policy', 'System Scale (N)'])

    df_excel.to_excel(excel_filename, index=False)
    print("\n" + "=" * 80)
    print(f"🎉 All done! Data successfully saved to: {excel_filename}")
    print("=" * 80)