import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pandas import DataFrame
from scipy.stats import t
import matplotlib.pyplot as plt
from pathlib import Path
import parameter_setting_CHEN106_1 as ps  # 【改动1】：统一从 ps 获取分布常量
EXCEL_DIR = Path("Results_excel")
EXCEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 辅助函数：从 env 获取参数
# ==========================================
def f(x, env):
    return (x ** 2) * env.penalty_weight


# Performance_Evaluation_CHEN_1.py

def generate_arrival_sequence_poi(env):
    #Bernoulli
    arrivals = []
    for t in range(env.T):
        prob = ps.get_time_varying_prob(t)
        arrivals_at_t = []

        for _ in range(env.N):
            # np.random.rand() 生成 0~1 的随机数，小于 prob 代表有车来
            if np.random.rand() < prob:
                r = np.random.choice(ps.r_dist, p=ps.r_p)
                l = np.random.choice(ps.l_dist, p=ps.l_p)
                arrivals_at_t.append((r, l))
            else:
                arrivals_at_t.append(None)  # 这个桩当前时刻没有车来
        arrivals.append(arrivals_at_t)
    return arrivals
'''
def generate_arrival_sequence_poi(env):
    #Poisson (接入 env)
    arrivals = []
    for t in range(env.T):
        prob = ps.get_time_varying_prob(t)
        lam = env.N * prob * 0.5  # 使用 env.N
        num_arrivals = np.random.poisson(lam)

        arrivals_at_t = []
        for _ in range(num_arrivals):
            r = np.random.choice(ps.r_dist, p=ps.r_p)
            l = np.random.choice(ps.l_dist, p=ps.l_p)
            arrivals_at_t.append((r, l))
        arrivals.append(arrivals_at_t)
    return arrivals
'''

'''
def transition_probability_simu(current_state, action, current_t, arrival_seq, env):
    """状态转移 (接入 env)"""
    if len(current_state) != 2 * env.N:
        raise ValueError("The length of the state vector does not match the number of actions.")
    new_state = list(current_state)

    # --- Step 1: Process Charging & Decay for existing cars ---
    for i in range(env.N):
        r_idx = 2 * i
        l_idx = 2 * i + 1

        if new_state[r_idx] < action[i]:
            raise ValueError(f"Action {action[i]} exceeds the remaining charge {new_state[r_idx]} at index {i}")

        if new_state[l_idx] > 0:
            new_state[r_idx] = max(new_state[r_idx] - action[i], 0)

        new_state[l_idx] = max(new_state[l_idx] - 1, 0)

    # --- Step 2: Handle New Arrivals ---
    for i in range(env.N):
        new_state[2 * i] = 0 if new_state[2 * i + 1] == 0 else new_state[2 * i]

    empty_charger_indices = [i for i in range(env.N) if new_state[2 * i + 1] == 0]
    arriving_cars = arrival_seq[current_t]
    num_to_fill = min(len(empty_charger_indices), len(arriving_cars))

    for k in range(num_to_fill):
        target_idx = empty_charger_indices[k]
        car_params = arriving_cars[k]
        new_state[2 * target_idx] = car_params[0]
        new_state[2 * target_idx + 1] = car_params[1]

    return tuple(new_state)

'''

def transition_probability_simu(current_state, action, current_t, arrival_seq, env):
    """
    【最终修正版】：严格匹配独立臂 MDP 的状态转移矩阵
    """
    if len(current_state) != 2 * env.N:
        raise ValueError("The length of the state vector does not match.")
    new_state = list(current_state)

    # --- Step 1: 处理现有车辆的充电与时长衰减 ---
    for i in range(env.N):
        r_idx = 2 * i
        l_idx = 2 * i + 1

        if new_state[r_idx] < action[i]:
            raise ValueError(f"Action {action[i]} exceeds remaining charge.")

        if new_state[l_idx] > 0:
            new_state[r_idx] = max(new_state[r_idx] - action[i], 0)
            new_state[l_idx] = max(new_state[l_idx] - 1, 0)

    # --- Step 2: 处理新车到达 (完全解耦) ---
    arriving_cars_at_t = arrival_seq[current_t]

    for i in range(env.N):
        r_idx = 2 * i
        l_idx = 2 * i + 1

        # 只有当该充电桩本身完全空闲 (L=0) 时，才能检查专属它的车是否来了
        if new_state[l_idx] == 0:
            car = arriving_cars_at_t[i]  # 取出它的专属到达情况
            if car is not None:
                new_state[r_idx] = car[0]
                new_state[l_idx] = car[1]
            else:
                new_state[r_idx] = 0  # 没有车来，确保清空旧的 R，回归纯净的 (0,0) 状态

    return tuple(new_state)

def reward_function(current_state, action, t, env):
    """奖励函数 (接入 env)"""
    current_p = ps.get_time_varying_p0(t)
    rewards = [
        (min(r, a) * (ps.alpha - current_p) - (f(max(0, r - a), env) if l == 1 else 0))
        for r, l, a in zip(current_state[::2], current_state[1::2], action)
    ]
    return sum(rewards)


# ==========================================
# 各类Policy分配逻辑 (核心算法 100% 保持不变)
# ==========================================

# 1. NEW Policy
def new_policy(state, env):
    action = [0] * env.N
    available_power = env.total_power
    chargers = [(state[2 * i], state[2 * i + 1], i) for i in range(env.N)]
    active_chargers = [c for c in chargers if c[1] > 0]
    sorted_chargers = sorted(active_chargers, key=lambda x: (x[0] - ps.MAX_CHARGE * x[1]))
    for r, l, new in sorted_chargers:
        alloc = min(r, ps.MAX_CHARGE, available_power)
        action[new] = alloc
        available_power -= alloc
        if available_power <= 0: break
    return tuple(action)


def simulate_new(initial_state, arrival_seq, env, eval_window=100):
    step_rewards, states, actions = [], [initial_state], []
    current_state = initial_state
    for t in range(env.T):
        action = new_policy(current_state, env)
        actions.append(action)
        step_rewards.append(reward_function(current_state, action, t, env))
        current_state = transition_probability_simu(current_state, action, t, arrival_seq, env)
        states.append(current_state)
    eval_window = min(eval_window, env.T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards


# 2. LLF Policy
def llf_policy(state, env):
    action = [0] * env.N
    available_power = env.total_power
    chargers = [(state[2 * i], state[2 * i + 1], i) for i in range(env.N)]
    active_chargers = [c for c in chargers if c[1] > 0]
    sorted_chargers = sorted(active_chargers, key=lambda x: (x[0]))
    for r, l, llf in sorted_chargers:
        alloc = min(r, ps.MAX_CHARGE, available_power)
        action[llf] = alloc
        available_power -= alloc
        if available_power <= 0: break
    return tuple(action)


def simulate_llf(initial_state, arrival_seq, env, eval_window=100):
    step_rewards, states, actions = [], [initial_state], []
    current_state = initial_state
    for t in range(env.T):
        action = llf_policy(current_state, env)
        actions.append(action)
        step_rewards.append(reward_function(current_state, action, t, env))
        current_state = transition_probability_simu(current_state, action, t, arrival_seq, env)
        states.append(current_state)
    eval_window = min(eval_window, env.T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards


# 3. Greedy Policy
def gdy_policy(state, env):
    available_power = env.total_power
    allocation = [0] * env.N
    active_chargers = []
    for i in range(env.N):
        r, l = state[2 * i], state[2 * i + 1]
        if l > 0 and r > 0:
            active_chargers.append((i, r))
    active_chargers.sort(key=lambda x: -x[1])
    for idx, demand in active_chargers:
        if available_power <= 0: break
        alloc = min(demand, ps.MAX_CHARGE, available_power)
        allocation[idx] = alloc
        available_power -= alloc
    return tuple(allocation)


def simulate_gdy(initial_state, arrival_seq, env, eval_window=100):
    step_rewards, states, actions = [], [initial_state], []
    current_state = initial_state
    for t in range(env.T):
        action = gdy_policy(current_state, env)
        actions.append(action)
        step_rewards.append(reward_function(current_state, action, t, env))
        current_state = transition_probability_simu(current_state, action, t, arrival_seq, env)
        states.append(current_state)
    eval_window = min(eval_window, env.T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards


# 4. LRF Policy
def lrf_policy(state, env):
    action = [0] * env.N
    available_power = env.total_power
    chargers = [(state[2 * i], state[2 * i + 1], i) for i in range(env.N)]
    active_chargers = [c for c in chargers if c[1] > 0]
    sorted_chargers = sorted(active_chargers, key=lambda x: (x[1], -x[0]))
    for r, l, lrf in sorted_chargers:
        alloc = min(r, ps.MAX_CHARGE, available_power)
        action[lrf] = alloc
        available_power -= alloc
        if available_power <= 0: break
    return tuple(action)


def simulate_lrf(initial_state, arrival_seq, env, eval_window=100):
    step_rewards, states, actions = [], [initial_state], []
    current_state = initial_state
    for t in range(env.T):
        action = lrf_policy(current_state, env)
        actions.append(action)
        step_rewards.append(reward_function(current_state, action, t, env))
        current_state = transition_probability_simu(current_state, action, t, arrival_seq, env)
        states.append(current_state)
    eval_window = min(eval_window, env.T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards


# 5. Index Policy
def index_algorithm(state, action, t, table):
    periodic = t % 24
    s_idx = ps.S_TO_IDX[state]
    return table[s_idx, int(periodic), int(action)]


def greedy_charging_strategy(current_state, t, env, table):
    final_action = [0] * env.N
    remaining_power = env.total_power
    while remaining_power > 0:
        candidates = []
        for i in range(env.N):
            r, l = current_state[2 * i], current_state[2 * i + 1]
            current_a = final_action[i]
            if current_a < min(ps.MAX_CHARGE, r):
                current_index = index_algorithm((r, l), current_a, t, table)
                candidates.append((current_index, i))
        if not candidates: break
        selected = max(candidates, key=lambda x: x[0])
        final_action[selected[1]] += 1
        remaining_power -= 1
    return tuple(final_action)


def simulate_idx(initial_state, arrival_seq, env, table, eval_window=100):
    step_rewards, states, actions = [], [initial_state], []
    current_state = initial_state
    for t in range(env.T):
        action = greedy_charging_strategy(current_state, t, env, table)
        actions.append(action)
        step_rewards.append(reward_function(current_state, action, t, env))
        current_state = transition_probability_simu(current_state, action, t, arrival_seq, env)
        states.append(current_state)
    eval_window = min(eval_window, env.T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards

# 6. Index_Xu Policy
def index_Xu_policy(state, env, index_table_Xu, current_t):
    periodic_t = current_t % ps.period
    action = [0] * env.N
    available_power = env.total_power

    # 抽取每个充电桩的 (r, l, i)
    chargers = [(state[2 * i], state[2 * i + 1], i) for i in range(env.N)]
    active_chargers = [c for c in chargers if c[1] > 0 and c[0] > 0]

    candidates = []
    for r, l, idx in active_chargers:
        s = (r, l)
        if s in ps.S_TO_IDX:
            s_idx = ps.S_TO_IDX[s]
            # 获取对应的二元动作指数
            idx_val = index_table_Xu[s_idx, periodic_t]
            candidates.append((idx_val, r, l, idx))

    # 按照 Index 从大到小排序，贪心分配
    candidates.sort(key=lambda x: x[0], reverse=True)

    for idx_val, r, l, idx in candidates:
        if available_power <= 0:
            break
        # Xu Policy 认为只要被激活，就给最高功率，或者给满足需求的最高功率
        alloc = min(r, ps.MAX_CHARGE, available_power)
        action[idx] = alloc
        available_power -= alloc

    return tuple(action)


def simulate_idx_Xu(initial_state, arrival_seq, env, index_table_Xu, eval_window=100):
    step_rewards, states, actions = [], [initial_state], []
    current_state = initial_state

    for t in range(env.T):
        action = index_Xu_policy(current_state, env, index_table_Xu, t)
        actions.append(action)
        step_rewards.append(reward_function(current_state, action, t, env))
        current_state = transition_probability_simu(current_state, action, t, arrival_seq, env)
        states.append(current_state)

    eval_window = min(eval_window, env.T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards



# 7. Clairvoyant (CVT) Policy
def process_sequence(arrival_seq, env):
    T_len = len(arrival_seq)
    dict_r, dict_l = {}, {}
    slot_free_time = [0] * env.N
    for t in range(T_len):
        arrivals_at_t = arrival_seq[t]
        arrival_idx = 0
        for n in range(env.N):
            is_slot_free = (t >= slot_free_time[n])
            if is_slot_free and arrival_idx < len(arrivals_at_t):
                '''r_val, l_val = arrivals_at_t[arrival_idx]'''
                car = arrivals_at_t[arrival_idx]
                if car is None:
                    continue  # 如果这个桩没有车来，直接跳过，处理下一个
                r_val, l_val = car
                dict_r[(t, n)] = r_val
                dict_l[(t, n)] = l_val
                if l_val > 0:
                    slot_free_time[n] = t + l_val
                arrival_idx += 1
            else:
                dict_r[(t, n)] = 0
                dict_l[(t, n)] = 0
    return dict_r, dict_l


def cvt_cts_policy(arrival_seq, env, eval_window=100):
    dict1, dict2 = process_sequence(arrival_seq, env)
    r = {(n + 1, t + 1): v for (t, n), v in dict1.items()}
    l = {(n + 1, t + 1): v for (t, n), v in dict2.items()}
    gamma = {(n, t): env.penalty_weight for n in range(1, env.N + 1) for t in range(1, env.T + 1)}

    model = gp.Model()
    w = model.addVars(range(1, env.N + 1), range(1, env.T + 1), vtype=GRB.CONTINUOUS, lb=0, ub=ps.MAX_CHARGE, name="w")
    obj = gp.LinExpr()

    for t in range(1, env.T + 1):
        current_p = ps.get_time_varying_p0(t - 1)
        for n in range(1, env.N + 1):
            obj += (ps.alpha - current_p) * w[n, t]
            if r.get((n, t), 0) > 0:
                penalty = gamma[n, t] * (
                            r[n, t] - gp.quicksum(w[n, s] for s in range(t, min(t + l[n, t], env.T + 1)))) ** 2
                obj -= penalty
    model.setObjective(obj, GRB.MAXIMIZE)

    for t in range(1, env.T + 1):
        model.addConstr(gp.quicksum(w[n, t] for n in range(1, env.N + 1)) <= env.total_power, name=f"c4_{t}")

    for n in range(1, env.N + 1):
        for t in range(1, env.T + 1):
            if r.get((n, t), 0) > 0:
                end_time = min(t + l[(n, t)], env.T + 1)
                model.addConstr(
                    r[n, t] - gp.quicksum(w[n, s] for s in range(t, end_time)) >= 0, name=f"c5_{n}_{t}")

    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 0.001)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None, 0.0

    sol = model.getAttr('X', w)
    eval_window = min(eval_window, env.T)
    start_t = env.T - eval_window + 1
    steady_reward = 0

    for t in range(start_t, env.T + 1):
        current_p = ps.get_time_varying_p0(t - 1)
        for n in range(1, env.N + 1):
            steady_reward += (ps.alpha - current_p) * sol[n, t]
            if r.get((n, t), 0) > 0:
                penalty = gamma[n, t] * (r[n, t] - sum(sol[n, s] for s in range(t, min(t + l[(n, t)], env.T + 1))
                )) ** 2
                steady_reward -= penalty

    avg_steady_reward = steady_reward / eval_window
    return sol, avg_steady_reward


# ==========================================
# 统一入口 (完美对接 Optimality_evaluation.py)
# ==========================================
def run_experiments(algorithm, arrival_seq, initial, env, table=None, eval_window=100):
    if algorithm == "index":
        return simulate_idx(initial, arrival_seq, env, table, eval_window)
    elif algorithm == "index_Xu":
        return simulate_idx_Xu(initial, arrival_seq, env, table, eval_window)
    else:
        simulate_func = {
            'new': simulate_new,
            'gdy': simulate_gdy,
            'llf': simulate_llf,
            'lrf': simulate_lrf
        }[algorithm]
        return simulate_func(initial, arrival_seq, env, eval_window)


# ==========================================
# 独立测试模块 (用于单独检验仿真逻辑和 Gurobi)
# ==========================================
# ==========================================
# 独立测试模块 (包含 Whittle Index 完整测试)
# ==========================================
if __name__ == "__main__":
    from charging_env import ChargingEnv
    from Index_calculation import WhittleSolver, WhittleSolverXu
    import time
    from r_beta_1 import solve_single_bandit_relaxation
    import pandas as pd
    print("✅ Testing Performance_Evaluation ...")

    # 1. 创建测试环境
    for penalty_weight in [0.2,0.4,0.6,0.8]:
        test_env = ChargingEnv(N=20, power_ratio=0.6, penalty_weight=penalty_weight)
        test_env1 = ChargingEnv(N=20, power_ratio=0.6, penalty_weight=penalty_weight, T=24)

        # 2. 生成泊松到达序列
        print("\n1️⃣ arrival process...")
        arr_seq = generate_arrival_sequence_poi(test_env)
        init_s = tuple([0, 0] * test_env.N)
        eval_win = 96

        # 3. 准备 Whittle Index 表
        print("\n2️⃣ preparing for Whittle Index Policy Form...")
        solver = WhittleSolver(test_env)
        INDEX_TABLE = solver.get_index_table()
        print("✔️ Index Form Done！")

        print("\n2️⃣ preparing for Whittle Index_Xu Policy Form...")
        solver_Xu = WhittleSolverXu(test_env)
        INDEX_TABLE_XU = solver_Xu.get_index_table_Xu()
        print("✔️ Index_Xu Form Done！")

        # 创建结果列表
        results = []

        # 4. Test all the Policy
        print("\n3️⃣ Testing All the Policy...")

        # 5. Test CVT (Gurobi)
        start_time = time.time()
        sol, cvt_total_reward = cvt_cts_policy(arr_seq, test_env, eval_window=eval_win)
        cvt_time = time.time() - start_time

        print("=" * 50)


        # 测试其他策略
        algorithms_to_test = ['llf', 'new', 'lrf', 'gdy', 'index_Xu', 'index']

        for alg in algorithms_to_test:
            start_time = time.time()

            if alg == 'index':
                reward, _, _, _ = run_experiments(
                    algorithm=alg,
                    arrival_seq=arr_seq,
                    initial=init_s,
                    env=test_env,
                    table=INDEX_TABLE,
                    eval_window=eval_win
                )
            elif alg == 'index_Xu':
                reward, _, _, _ = run_experiments(
                    algorithm=alg,
                    arrival_seq=arr_seq,
                    initial=init_s,
                    env=test_env,
                    table=INDEX_TABLE_XU,
                    eval_window=eval_win
                )
            else:
                reward, _, _, _ = run_experiments(
                    algorithm=alg,
                    arrival_seq=arr_seq,
                    initial=init_s,
                    env=test_env,
                    eval_window=eval_win
                )

            single_charger_reward = reward / test_env.N
            cost_time = time.time() - start_time

            results.append([alg.upper(), single_charger_reward, cost_time])
            print(f"✔️ Policy [{alg.upper():<9}] -> Average reward: {single_charger_reward:.4f} ( {cost_time:.3f}s)")

        if sol is not None:
            single_cvt_reward = cvt_total_reward / test_env.N
            results.append(['CVT', single_cvt_reward, cvt_time])
            print(f"✔️ Policy [CVT      ] -> Average reward: {single_cvt_reward:.4f} ( {cvt_time:.3f}s)")
        else:
            results.append(['CVT', 'Infeasible', cvt_time])
            print("❌ CVT Infeasible！")

        # 6. Test LP bound
        start_time = time.time()
        P_mat, R_mat = test_env1.precompute_matrices()
        lp_reward, actual_power, beta_star = solve_single_bandit_relaxation(
            P_mat, R_mat, test_env.avg_power
        )
        lp_time = time.time() - start_time

        if lp_reward is not None:
            results.append(['LP_Bound', lp_reward, lp_time])
            print(f"✔️ Policy [LP       ] -> Average reward: {lp_reward:.4f} ( {lp_time:.3f}s)")
        else:
            results.append(['LP_Bound', 'Infeasible', lp_time])
            print("❌ Gurobi Infeasible。")
        print("=" * 50 + "\n")

        # 保存到Excel
        df = pd.DataFrame(results, columns=['Policy', 'Average Reward per Charger', 'Computation Time (s)'])
        file_name = f'Performance_evaluation_penalty={test_env.penalty_weight}_ratio={test_env.power_ratio}.xlsx'
        save_path = EXCEL_DIR / file_name
        df.to_excel(save_path, index=False)
        print(f"📊 Results saved to: {save_path}")
