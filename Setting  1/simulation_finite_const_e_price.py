import random
import numpy as np
import itertools
import sys
from scipy.stats import ttest_ind, t
import gurobipy as gp
from gurobipy import GRB
from pandas import DataFrame
import re
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
'''
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12]
r_p = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09]
r_dist = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42]
r_p = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]

r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12, 13, 14, 15, 16, 17, 18, 19,20]
r_p = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12, 13, 14, 15, 16, 17, 18, 19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
r_p = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,0.025,
       0.025, 0.025, 0.025, 0.025, 0.025,0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 
       0.025, 0.025, 0.025, 0.025,0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 
       0.025, 0.025, 0.025,0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,] 


l_dist = [1, 2, 3]
l_p= [0.3,0.4,0.3]
l_dist = [1, 2, 3, 4, 5]
l_p= [0.2,0.2,0.2,0.2,0.2]
l_dist = [1, 2, 3, 4, 5, 6, 7]
l_p= [0.14,0.14,0.14,0.14,0.14,0.15,0.15]
l_dist = [1, 2, 3, 4, 5, 6, 7,8]
l_p= [0.1,0.1,0.1,0.1,0.15,0.15,0.15, 0.15]
l_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9]
l_p = [0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]
l_dist = [1, 2, 3, 4, 5,6,7,8,9,10]
l_p= [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
l_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12, 13, 14,15]
l_p = [0.07, 0.07, 0.07, 0.07, 0.07,0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06]
'''
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12]
r_p = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09]
l_dist = [1, 2, 3]
l_p= [0.3,0.4,0.3]
DISCOUNT_FACTOR = 0.9

# Global function

def generate_arrival_sequence():
    return [
        [  # 每个时间步的到达情况
            (np.random.choice(r_dist, p=r_p), np.random.choice(l_dist, p=l_p))
            if random.random() < VEHICLE_GEN_PROB else None
            for _ in range(N)
        ]
        for _ in range(T)
    ]


def transition_probability_simu(current_state, action, current_t, arrival_seq):
    """确定性状态转移版本"""
    if len(current_state) != 2 * N:
        raise ValueError("The length of the state vector does not match the number of actions.")
    new_state = list(current_state)

    # 第一步：处理所有充电桩的衰减
    for i in range(N):
        r = 2 * i
        l = 2 * i + 1
        if new_state[r] < action[i]:
            raise ValueError("Action exceeds the remaining charge of the car.")
        # 只有剩余时间>0时才衰减请求功率
        if new_state[l] > 0:
            new_state[r] = max(new_state[r] - action[i], 0)
        # 统一减少剩余时间
        new_state[l] = max(new_state[l] - 1, 0)

    # 第二步：处理新车生成
    for i in range(N):
        l = 2 * i + 1

        if new_state[l] == 0:
            arrival = arrival_seq[current_t][i]  # 使用预生成数据
            if arrival is not None:
                new_state[2 * i], new_state[l] = arrival
            else:
                new_state[2 * i] = new_state[l] = 0

    return tuple(new_state)


def reward_function(current_state, action):
    rewards = [
        (min(r, a) * (alpha - p_0) - (f(max(0, r - a)) if l == 1 else 0))
        for r, l, a in zip(current_state[::2],
                           current_state[1::2],
                           action)
    ]
    return sum(rewards)


def run_experiments(algorithm, arrival_seq, initial, power):  # 调整参数顺序
    reward, _, _ = {
        'new': simulate_new,
        'index': simulate_idx,
        'index_xu': simulate_idx_xu,
        'gdy': simulate_gdy,
        'llf': simulate_llf,
        'lrf': simulate_lrf
    }[algorithm](T, initial, power, arrival_seq)
    return reward


def t_test(algorithm1_rewards, algorithm2_rewards):
    t_stat, p_value = ttest_ind(algorithm1_rewards, algorithm2_rewards,
                                alternative='less',  # 检验algorithm1是否小于algorithm2
                                equal_var=False)  # 使用Welch's t-test
    print(f"Algorithm 1: {sum(algorithm1_rewards) / NUM}, Algorithm 2: {sum(algorithm2_rewards) / NUM}")
    print(f"T-test结果:")
    print(f"t统计量 = {t_stat:.6f}")
    print(f"p值 = {p_value:.6f}")
    if p_value < 0.05:
        print(f"统计显著：algorithm 1 显著小于 algorithm 2 (p < 0.05)")
    else:
        print("未达到统计显著性：无法证明两个algorithm优劣性")


# new_policy: 以(R-L*max_charge)为index
def new_policy(state, power):
    N = len(state) // 2
    action = [0] * N
    available_power = power
    chargers = [(state[2 * i], state[2 * i + 1], i) for i in range(N)]  # 解包为(r, l, 充电桩索引)

    # 过滤有效充电请求（l>0）
    active_chargers = [c for c in chargers if c[1] > 0]

    # 按r升序、l升序排序（r越小优先级越高，r相同则l越小越优先）
    sorted_chargers = sorted(active_chargers, key=lambda x: (x[0] - MAX_CHARGE * x[1]))
    for r, l, new in sorted_chargers:
        # 计算可分配功率
        alloc = min(r, MAX_CHARGE, available_power)
        action[new] = alloc
        available_power -= alloc

        if available_power <= 0:
            break

    return tuple(action)


def simulate_new(T, initial_state, power, arrival_seq):
    total_reward = 0
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = new_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action)
        total_reward += reward * (DISCOUNT_FACTOR ** t)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# clairvoyant policy
def process_sequence(original_seq):
    T_len = len(original_seq)
    if T_len == 0:
        return {}, {}  # 返回空字典

    N_len = len(original_seq[0]) if T_len > 0 else 0

    # 新增字典初始化
    dict_r = {}
    dict_l = {}

    # 处理时间序列（原地修改）
    for t in range(T_len):
        for n in range(N_len):
            current_event = original_seq[t][n]
            if current_event is not None and current_event[1] > 0:
                l = current_event[1]
                for k in range(1, l):
                    prev_t = t + k
                    if prev_t < T_len:
                        original_seq[prev_t][n] = None

    # 新增字典生成逻辑
    for t in range(T_len):
        for n in range(N_len):
            event = original_seq[t][n]
            dict_r[(t, n)] = event[0] if event else 0
            dict_l[(t, n)] = event[1] if event else 0

    return dict_r, dict_l


def cvt_cts_policy(arrival_seq):
    dict1, dict2 = process_sequence(arrival_seq)
    r = {(n + 1, t + 1): v for (t, n), v in dict1.items()}
    l = {(n + 1, t + 1): v for (t, n), v in dict2.items()}

    # 辅助变量 β[n,t]
    beta = {(n, t): penalty_weight for n in range(1, N + 1) for t in range(1, T + 1)}

    # 初始化模型
    model = gp.Model()

    # 决策变量：w[n,t] ∈ [0, w_0] ∩ ℕ
    w = model.addVars(range(1, N + 1), range(1, T + 1), vtype=GRB.CONTINUOUS, lb=0, ub=MAX_CHARGE, name="w")

    # 目标函数
    model.setObjective(
        gp.quicksum(
            (DISCOUNT_FACTOR ** (t - 1)) * gp.quicksum(
                (alpha - p_0) * w[n, t] - beta[n, t] * (r[n, t] - gp.quicksum(
                    w[n, s] for s in range(t, min(t + l[n, t], T + 1))
                )) ** 2
                for n in range(1, N + 1)
            )
            for t in range(1, T + 1)
        ),
        GRB.MAXIMIZE
    )

    # 约束 (4): ∑_n w[n,t] ≤ W
    for t in range(1, T + 1):
        model.addConstr(gp.quicksum(w[n, t] for n in range(1, N + 1)) <= total_power, name=f"c4_{t}")

    # 约束 (5): r[n,t] - ∑_{s=t}^{t+l[n,t]-1} w[n,s] ≥ 0
    for n in range(1, N + 1):
        for t in range(1, T + 1):
            model.addConstr(
                r[n, t] - gp.quicksum(w[n, s] for s in range(t, min(t + l[n, t], T + 1))) >= 0,
                name=f"c5_{n}_{t}"
            )

    # 求解
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 0)  # 设置 0.5% gap
    model.optimize()
    sol = model.getAttr('X', w)
    return sol, model.objVal


# LLF:
def llf_policy(state, power):
    N = len(state) // 2
    action = [0] * N
    available_power = power
    chargers = [(state[2 * i], state[2 * i + 1], i) for i in range(N)]  # 解包为(r, l, 充电桩索引)

    # 过滤有效充电请求（l>0）
    active_chargers = [c for c in chargers if c[1] > 0]

    # 按r升序、l升序排序（r越小优先级越高，r相同则l越小越优先）
    sorted_chargers = sorted(active_chargers, key=lambda x: (x[0]))
    for r, l, llf in sorted_chargers:
        # 计算可分配功率
        alloc = min(r, MAX_CHARGE, available_power)
        action[llf] = alloc
        available_power -= alloc

        if available_power <= 0:
            break

    return tuple(action)


def simulate_llf(T, initial_state, power, arrival_seq):
    total_reward = 0
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = llf_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action)
        total_reward += reward * (DISCOUNT_FACTOR ** t)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# gdy

def gdy_policy(state, power):
    """通用N充电桩贪心策略"""
    available_power = power
    N = len(state) // 2  # 自动识别充电桩数量
    allocation = [0] * N
    # 收集所有需要充电的桩 (剩余时间 > 0)
    active_chargers = []
    for i in range(N):
        r = state[2 * i]
        l = state[2 * i + 1]
        if l > 0 and r > 0:
            active_chargers.append((i, r))  # 保存索引和请求功率

    # 按请求功率降序排序
    active_chargers.sort(key=lambda x: -x[1])

    # 按优先级分配功率
    for idx, demand in active_chargers:
        if available_power <= 0:
            break
        # 计算可分配功率
        alloc = min(demand, MAX_CHARGE, available_power)
        allocation[idx] = alloc
        available_power -= alloc

    return tuple(allocation)


def simulate_gdy(T, initial_state, power, arrival_seq):
    total_reward = 0
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = gdy_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action)
        total_reward += reward * (DISCOUNT_FACTOR ** t)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# LRF policy:先给最小remaining job 的充电，同样的话考虑给最小lifetime的充
def lrf_policy(state, power):
    N = len(state) // 2
    action = [0] * N
    available_power = power

    chargers = [(state[2 * i], state[2 * i + 1], i) for i in range(N)]  # 解包为(r, l, 充电桩索引)

    # 过滤有效充电请求（l>0）
    active_chargers = [c for c in chargers if c[1] > 0]

    # 按r升序、l升序排序（r越小优先级越高，r相同则l越小越优先）
    sorted_chargers = sorted(active_chargers, key=lambda x: (x[1], -x[0]))
    for r, l, lrf in sorted_chargers:
        # 计算可分配功率
        alloc = min(r, MAX_CHARGE, available_power)
        action[lrf] = alloc
        available_power -= alloc

        if available_power <= 0:
            break

    return tuple(action)


def simulate_lrf(T, initial_state, power, arrival_seq):
    total_reward = 0
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = lrf_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action)
        total_reward += reward * (DISCOUNT_FACTOR ** t)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# Index policy_XU

def idx_xu(r: int, l: int) -> float:
    # R = 0
    if r == 0:
        return 0

    # R <= (L-1) * MAX_CHARGE
    elif r <= (l - 1) * MAX_CHARGE:
        return alpha - p_0
    else:
        return alpha - p_0 + (DISCOUNT_FACTOR**(l-1))* delta_f(r - (l - 1) * MAX_CHARGE)


def idx_xu_order(state, power):
    """基于index_xu的简化策略"""
    N = len(state) // 2
    action = [0] * N
    available_power = power
    chargers = []

    # 收集所有充电桩的index值
    for i in range(N):
        r = state[2 * i]
        l = state[2 * i + 1]
        chargers.append((idx_xu(r, l), i))  # (index值, 充电桩索引)

    # 按index值降序排序
    sorted_chargers = sorted(chargers, key=lambda x: -x[0])

    for idx_value, i in sorted_chargers:
        max_possible = min(state[2 * i], MAX_CHARGE)
        alloc = min(max_possible, available_power)
        action[i] = alloc
        available_power -= alloc

        if available_power <= 0:
            break

    return tuple(action)


def simulate_idx_xu(T, initial_state, power, arrival_seq):
    total_reward = 0
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = idx_xu_order(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action)
        total_reward += reward * (DISCOUNT_FACTOR ** t)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# Index policy

def index(r: int, l: int, t: int, i: int) -> float:
    # R < (L-1)MAX_CHARGE
    if r <= (l - 1) * MAX_CHARGE:
        return alpha - p_0

    # R > LW
    if r > l * MAX_CHARGE:
        if 0 <= i <= MAX_CHARGE - 1:
            return alpha - p_0 +(DISCOUNT_FACTOR**(l-1))* delta_f(r - (l - 1) * MAX_CHARGE - i)
        else:
            return 0

    # (L-1)W < R ≤ LW
    max_i = r - (l - 1) * MAX_CHARGE - 1
    if 0 <= i <= max_i:
        return alpha - p_0 +(DISCOUNT_FACTOR**(l-1))* delta_f(r - (l - 1) * MAX_CHARGE - i)
    else:
        return alpha - p_0


def greedy_charging_strategy(current_state, t, power):
    N = len(current_state) // 2
    final_action = [0] * N
    remaining_power = power

    while remaining_power > 0:
        # 收集所有可分配充电桩的指数
        candidates = []
        for i in range(N):
            r = current_state[2 * i]
            l = current_state[2 * i + 1]
            current_a = final_action[i]

            # 检查是否还能继续分配
            if current_a < min(MAX_CHARGE, r):
                current_index = index(r, l, t, current_a)
                candidates.append((current_index, i))

        if not candidates:
            break

        # 选择指数最大的充电桩
        selected = max(candidates, key=lambda x: x[0])
        final_action[selected[1]] += 1
        remaining_power -= 1

    return tuple(final_action)


def simulate_idx(T, initial_state, power, arrival_seq):
    total_reward = 0
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = greedy_charging_strategy(current_state, t, power)
        actions.append(action)

        reward = reward_function(current_state, action)
        total_reward += reward * (DISCOUNT_FACTOR ** t)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


if __name__ == "__main__":
    results = []
    charger_states = []
    alpha = 3  # 单位充电收益
    p_0 = 1  # 固定电价
    T = 500
    max_r = max(r_dist)
    max_l = max(l_dist)
    MAX_CHARGE = max_r / 2  # 充电桩最大容量 W
    N = 10
    for penalty_weight in [0.6]:
        for VEHICLE_GEN_PROB in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95, 1]:
            for power_ratio in [0.6]:
                avg_l = sum(l_dist) / len(l_dist)
                avg_r = sum(r_dist) / len(r_dist)
                total_power = round(N * max_r / max_l * power_ratio)  # 总可用充电功率

                f = lambda x: (x ** 2) * penalty_weight
                delta_f = lambda x: f(x) - f(x - 1)
                print(
                    f"power_ratio={total_power}, penalty_weight = {penalty_weight}, total_power = {total_power}, VEHICLE_GEN_PROB = {VEHICLE_GEN_PROB}\n")
                flat_state = tuple([1, 1] * N)
                NUM = 50
                index_results = []
                index_xu_results = []  # 新增结果收集列表
                gdy_results = []
                llf_results = []
                lrf_results = []
                new_results = []
                cvt_results = []

                index_gap_results = []
                index_xu_gap_results = []  # 新增结果收集列表
                gdy_gap_results = []
                llf_gap_results = []
                lrf_gap_results = []
                new_gap_results = []
                cvt_gap_results = []
                for exp_id in range(NUM):
                    current_arrival_seq = generate_arrival_sequence()
                    _, cvt_reward = cvt_cts_policy(current_arrival_seq)
                    index_xu_reward = run_experiments('index_xu', current_arrival_seq, flat_state, total_power)
                    index_reward = run_experiments('index', current_arrival_seq, flat_state, total_power)
                    gdy_reward = run_experiments('gdy', current_arrival_seq, flat_state, total_power)
                    llf_reward = run_experiments('llf', current_arrival_seq, flat_state, total_power)
                    lrf_reward = run_experiments('lrf', current_arrival_seq, flat_state, total_power)
                    new_reward = run_experiments('new', current_arrival_seq, flat_state, total_power)
                    index_gap = abs(cvt_reward - index_reward) / abs(cvt_reward)
                    index_xu_gap = abs(cvt_reward - index_xu_reward) / abs(cvt_reward)
                    new_gap = abs(cvt_reward - new_reward) / abs(cvt_reward)
                    lrf_gap = abs(cvt_reward - lrf_reward) / abs(cvt_reward)
                    gdy_gap = abs(cvt_reward - gdy_reward) / abs(cvt_reward)
                    llf_gap = abs(cvt_reward - llf_reward) / abs(cvt_reward)
                    # 将结果添加到列表
                    cvt_results.append(cvt_reward)  # 添加这一行
                    index_results.append(index_reward)
                    index_xu_results.append(index_xu_reward)
                    gdy_results.append(gdy_reward)
                    llf_results.append(llf_reward)
                    lrf_results.append(lrf_reward)
                    new_results.append(new_reward)
                    # 将gap结果添加到列表
                    index_gap_results.append(index_gap)
                    index_xu_gap_results.append(index_xu_gap)
                    gdy_gap_results.append(gdy_gap)
                    llf_gap_results.append(llf_gap)
                    lrf_gap_results.append(lrf_gap)
                    new_gap_results.append(new_gap)

                # clairvoyant
                cvt_mean = np.mean(cvt_results)
                cvt_std = np.std(cvt_results, ddof=1)
                sem = cvt_std / np.sqrt(NUM)
                conf_int = t.interval(0.95, df=NUM - 1, loc=cvt_mean, scale=sem)
                # index
                index_mean = np.mean(index_results)
                index_std = np.std(index_results, ddof=1)
                sem = cvt_std / np.sqrt(NUM)
                idx_ci = t.interval(0.95, df=NUM - 1, loc=index_mean, scale=sem)
                # index xu
                index_xu_mean = np.mean(index_xu_results)
                index_xu_std = np.std(index_xu_results, ddof=1)
                sem = cvt_std / np.sqrt(NUM)
                idx_xu_ci = t.interval(0.95, df=NUM - 1, loc=index_xu_mean, scale=sem)
                # gdy
                gdy_mean = np.mean(gdy_results)
                gdy_std = np.std(gdy_results, ddof=1)
                sem = cvt_std / np.sqrt(NUM)
                gdy_ci = t.interval(0.95, df=NUM - 1, loc=gdy_mean, scale=sem)
                # llf
                llf_mean = np.mean(llf_results)
                llf_std = np.std(llf_results, ddof=1)
                sem = cvt_std / np.sqrt(NUM)
                llf_ci = t.interval(0.95, df=NUM - 1, loc=llf_mean, scale=sem)
                # lrf
                lrf_mean = np.mean(lrf_results)
                lrf_std = np.std(lrf_results, ddof=1)
                sem = cvt_std / np.sqrt(NUM)
                lrf_ci = t.interval(0.95, df=NUM - 1, loc=lrf_mean, scale=sem)
                # new
                new_mean = np.mean(new_results)
                new_std = np.std(new_results, ddof=1)
                sem = cvt_std / np.sqrt(NUM)
                new_ci = t.interval(0.95, df=NUM - 1, loc=new_mean, scale=sem)

                # index gap
                diff_IDX_mean = np.mean(index_gap_results)
                diff_IDX_std = np.std(index_gap_results, ddof=1)
                sem_IDX = diff_IDX_std / np.sqrt(NUM)
                CI_IDX = t.interval(0.95, df=NUM - 1, loc=diff_IDX_mean, scale=sem_IDX)
                # index_xu gap
                diff_XU_mean = np.mean(index_xu_gap_results)
                diff_XU_std = np.std(index_xu_gap_results, ddof=1)
                sem_XU = diff_XU_std / np.sqrt(NUM)
                CI_XU = t.interval(0.95, df=NUM - 1, loc=diff_XU_mean, scale=sem_XU)

                # greedy gap
                diff_GDY_mean = np.mean(gdy_gap_results)
                diff_GDY_std = np.std(gdy_gap_results, ddof=1)
                sem_GDY = diff_GDY_std / np.sqrt(NUM)
                CI_GDY = t.interval(0.95, df=NUM - 1, loc=diff_GDY_mean, scale=sem_GDY)

                # lrf gap
                diff_LRF_mean = np.mean(lrf_gap_results)
                diff_LRF_std = np.std(lrf_gap_results, ddof=1)
                sem_LRF = diff_LRF_std / np.sqrt(NUM)
                CI_LRF = t.interval(0.95, df=NUM - 1, loc=diff_LRF_mean, scale=sem_LRF)

                # llf gap
                diff_LLF_mean = np.mean(llf_gap_results)
                diff_LLF_std = np.std(llf_gap_results, ddof=1)
                sem_LLF = diff_LLF_std / np.sqrt(NUM)
                CI_LLF = t.interval(0.95, df=NUM - 1, loc=diff_LLF_mean, scale=sem_LLF)

                # NEW gap
                diff_NEW_mean = np.mean(new_gap_results)
                diff_NEW_std = np.std(new_gap_results, ddof=1)
                sem_NEW = diff_NEW_std / np.sqrt(NUM)
                CI_NEW = t.interval(0.95, df=NUM - 1, loc=diff_NEW_mean, scale=sem_NEW)
                row = {
                    "power_ratio": power_ratio,
                    "penalty_weight": penalty_weight,
                    "total_power": total_power,
                    "VEHICLE_GEN_PROB": VEHICLE_GEN_PROB,
                    "blank ": '',
                    "cvt_mean": cvt_mean,
                    "index_mean": index_mean,
                    "index_xu_mean": index_xu_mean,
                    "gdy_mean": gdy_mean,
                    "llf_mean": llf_mean,
                    "lrf_mean": lrf_mean,
                    "new_mean": new_mean,
                    "blank": '',
                    "index_gap": diff_IDX_mean,
                    "index_xu_gap": diff_XU_mean,
                    "gdy_gap": diff_GDY_mean,
                    "llf_gap": diff_LLF_mean,
                    "lrf_gap": diff_LRF_mean,
                    "new_gap": diff_NEW_mean,
                    "blank  ": '',
                    "index_gap_p": CI_IDX,
                    "index_xu_gap_p": CI_XU,
                    "gdy_gap_p": CI_GDY,
                    "llf_gap_p": CI_LLF,
                    "lrf_gap_p": CI_LRF,
                    "new_gap_p": CI_NEW,
                }
                results.append(row)
                df = DataFrame(results)
    df.to_excel("simulation_results(12,3)infinite_p=0.6_ratio=0.6prob_from_0.1to1.xlsx", index=False)
