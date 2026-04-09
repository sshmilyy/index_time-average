# verify_arrivals_multi_policy.py
import numpy as np
import time
import parameter_setting_CHEN106_1 as ps
from charging_env import ChargingEnv
from Performance_Evaluation_CHEN_1 import run_experiments
from Index_calculation import WhittleSolver  # 引入 Index 求解器


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
    print("🚀 开始多策略下的 Poisson 与 Bernoulli 渐进等价性验证...")

    # 设定测试的 N 和策略池
    N_list = [10,50, 100, 300]  # 建议先用这三个测，跑通了再加 500，因为多策略仿真较耗时
    POLICIES = ['gdy', 'llf', 'lrf', 'new', 'index']

    # 仿真参数
    TEST_POWER_RATIO = 0.6
    TEST_PENALTY = 0.8
    EVAL_WINDOW = 120
    NUM_SIMULATIONS = 50  # 每个策略测 10 次取平均

    for N in N_list:
        print(f"\n" + "=" * 80)
        print(f"⚙️ 正在测试系统规模: N = {N}")
        print("=" * 80)

        # 1. 创建环境
        env = ChargingEnv(N=N, power_ratio=TEST_POWER_RATIO, penalty_weight=TEST_PENALTY)
        init_s = tuple([0, 0] * env.N)

        # 2. 准备 Whittle Index 表 (如果策略池里有 'index')
        INDEX_TABLE = None
        if 'index' in POLICIES:
            solver = WhittleSolver(env)
            INDEX_TABLE = solver.get_index_table()  # 自动计算或读取缓存

        print(f"{'策略 (Policy)':<15} | {'Bernoulli 均收益':<20} | {'Poisson 均收益':<20} | {'差距 (Gap)':<10}")
        print("-" * 80)

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
            gap = abs(mean_bern - mean_pois) / mean_bern * 100 if mean_bern != 0 else 0

            print(f"{alg.upper():<15} | {mean_bern:<20.4f} | {mean_pois:<20.4f} | {gap:.2f}%")