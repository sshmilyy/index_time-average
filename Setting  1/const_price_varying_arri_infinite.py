import random
import numpy as np
import itertools
import sys
from scipy.stats import ttest_ind, t
import gurobipy as gp
from gurobipy import GRB
from pandas import DataFrame
import re
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12]
r_p = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09]
l_dist = [1, 2, 3]
l_p= [0.3,0.4,0.3]

# Global function
def get_time_varying_p0(t):
    t_in_period = t % T_PERIOD
    if t_in_period <= 6:
        return 0.67
    elif t_in_period <= 22:
        return 1.33
    elif t_in_period <= 23:
        return 0.67
def get_time_varying_prob(t, period=24, min_prob=0.4, max_prob=0.6):#v1
    t_in_period = t % period
    half_period = period / 2
    prob_range = max_prob - min_prob
    if t_in_period <= half_period:
        return min_prob + prob_range * (t_in_period / half_period)
    else:
        return min_prob + prob_range * ((period - t_in_period) / half_period)


def generate_arrival_sequence_poi():
    arrivals = []
    for t in range(T):
        # 1. Calculate Lambda for Poisson
        # We assume lambda scales with the previous probability to keep load similar
        # lambda = N * prob.
        prob = get_time_varying_prob(t)
        lam = N * prob

        # 2. Determine number of arrivals (Poisson)
        num_arrivals = np.random.poisson(lam)

        arrivals_at_t = []
        for _ in range(num_arrivals):
            # 3. Assign R and L state
            # Using random.choice to respect the distributions defined in your global scope.
            # If you strictly want uniform integers within a range (e.g., 1 to 12),
            # you can change this to: r = np.random.randint(min(r_dist), max(r_dist)+1)
            r = np.random.choice(r_dist, p=r_p)
            l = np.random.choice(l_dist, p=l_p)
            arrivals_at_t.append((r, l))

        arrivals.append(arrivals_at_t)
    return arrivals


def transition_probability_simu(current_state, action, current_t, arrival_seq):

    if len(current_state) != 2 * N:
        raise ValueError("The length of the state vector does not match the number of actions.")
    new_state = list(current_state)

    # --- Step 1: Process Charging & Decay for existing cars ---
    for i in range(N):
        r_idx = 2 * i
        l_idx = 2 * i + 1

        # Safety check for validity
        if new_state[r_idx] < action[i]:
            # In simulation, if algo allocates more than R, we just cap it or error.
            # Here strictly raising error as per original code.
            raise ValueError(f"Action {action[i]} exceeds the remaining charge {new_state[r_idx]} at index {i}")

        if new_state[l_idx] > 0:
            new_state[r_idx] = max(new_state[r_idx] - action[i], 0)

        new_state[l_idx] = max(new_state[l_idx] - 1, 0)

    # --- Step 2: Handle New Arrivals (Poisson Stream) ---

    # Identify which chargers are currently empty (L=0)
    for i in range(N):
        new_state[2*i] =0 if new_state[2*i+1]==0 else new_state[2*i]
    empty_charger_indices = [i for i in range(N) if new_state[2 * i + 1] == 0]

    # Get the pool of cars arriving at this specific time t
    arriving_cars = arrival_seq[current_t]  # List of (r, l) tuples

    # Determine how many cars can actually enter
    # Logic: min(Available Spots, Number of Arrivals)
    num_to_fill = min(len(empty_charger_indices), len(arriving_cars))

    # Assign cars to empty spots
    for k in range(num_to_fill):
        target_idx = empty_charger_indices[k]
        car_params = arriving_cars[k]

        new_state[2 * target_idx] = car_params[0]  # R
        new_state[2 * target_idx + 1] = car_params[1]  # L

    # Note: Any remaining cars in arriving_cars (if len > num_to_fill) are ignored.
    # This effectively simulates "rest leave".

    return tuple(new_state)


def reward_function(current_state, action,t):
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

        reward = reward_function(current_state, action,t)
        total_reward +=  beta**(t) * reward

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# clairvoyant policy
def process_sequence(arrival_seq):
    """
    输入: arrival_seq (list of lists): generate_arrival_sequence_poi 的输出
    输出: dict_r, dict_l (keys: (t, n), values: R or L)
    逻辑: 模拟实时停车过程。如果有空位，新来的车就停进去；否则车流失。
    """
    T_len = len(arrival_seq)

    # 初始化输出字典
    dict_r = {}
    dict_l = {}

    # 追踪每个充电桩何时变为空闲 (初始化为0，表示T=0时都是空的)
    # slot_free_time[n] = x 表示桩 n 在时刻 x 之后（含x）才是空的
    slot_free_time = [0] * N

    for t in range(T_len):
        arrivals_at_t = arrival_seq[t]  # 获取当前时刻到达的车队 [(r1, l1), (r2, l2)...]
        arrival_idx = 0  # 指向车队中下一辆待停车的车

        # 遍历所有充电桩，寻找空位
        for n in range(N):
            # 1. 判断桩 n 在当前时刻 t 是否空闲
            is_slot_free = (t >= slot_free_time[n])

            # 2. 如果桩空闲 且 还有新车在等待
            if is_slot_free and arrival_idx < len(arrivals_at_t):
                # 取出一辆车
                r_val, l_val = arrivals_at_t[arrival_idx]

                # 记录参数：时刻 t，桩 n，接纳了新车 R, L
                dict_r[(t, n)] = r_val
                dict_l[(t, n)] = l_val

                # 更新该桩的被占用状态
                # 假设车停 L 个时段。如果 t=0, l=1，它占用 t=0，t=1 时变为空闲
                if l_val > 0:
                    slot_free_time[n] = t + l_val

                # 指向下一辆车
                arrival_idx += 1

            else:
                # 两种情况会进这里：
                # A. 桩 n 还没充完电 (忙碌)
                # B. 桩 n 是空的，但是当前时刻没有新车来了 (或车都停完了)
                dict_r[(t, n)] = 0
                dict_l[(t, n)] = 0

        # 注意：如果 arrival_idx < len(arrivals_at_t)，说明车比桩多，
        # 多出来的车在这里会被直接丢弃 (Lost)，符合通常的 loss system 假设。

    return dict_r, dict_l


def cvt_cts_policy(arrival_seq):
    dict1, dict2 = process_sequence(arrival_seq)
    r = {(n + 1, t + 1): v for (t, n), v in dict1.items()}
    l = {(n + 1, t + 1): v for (t, n), v in dict2.items()}

    # 辅助变量 β[n,t]
    gamma = {(n, t): penalty_weight for n in range(1, N + 1) for t in range(1, T + 1)}

    # 初始化模型
    model = gp.Model()

    # 决策变量：w[n,t] ∈ [0, w_0] ∩ ℕ
    w = model.addVars(range(1, N + 1), range(1, T + 1), vtype=GRB.CONTINUOUS, lb=0, ub=MAX_CHARGE, name="w")

    # 目标函数
    # 构建目标函数表达式
    obj = gp.LinExpr()

    for t in range(1, T + 1):
        # 获取模拟环境同款的实时电价 (注意 t-1 对应 python 的 0-index)
        #current_p = get_time_varying_prob(t - 1, period=24)
        current_p = p_0
        for n in range(1, N + 1):
            # 1. 收益项：基于当前时刻 t 结算
            # 修正：用 current_p 替换 p_0
            obj += (beta ** (t - 1)) * (alpha - current_p) * w[n, t]

            # 2. 惩罚项：基于离开时刻结算
            # 只有当该 (n, t) 确实有车到达时才计算
            if r.get((n, t), 0) > 0:
                duration = l[(n, t)]
                arrival_idx = t - 1
                departure_idx = arrival_idx + duration - 1

                # 修正：折现因子使用离开时间 (departure_idx)
                # 惩罚项 = gamma * (R - sum(w))^2
                penalty = gamma[n, t] * (r[n, t] - gp.quicksum(
                    w[n, s] for s in range(t, min(t + l[n, t], T + 1))
                )) ** 2

                obj -= (beta ** departure_idx) * penalty

    model.setObjective(obj, GRB.MAXIMIZE)

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

        reward = reward_function(current_state, action,t)
        total_reward +=  beta**(t) * reward

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

        reward = reward_function(current_state, action,t)
        total_reward +=  beta**(t) * reward

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

        reward = reward_function(current_state, action,t)
        total_reward += beta**(t)  * reward

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# Index policy_XU

def idx_xu(r: int, l: int,t) -> float:
    # R = 0
    if r == 0:
        return 0

    # R <= (L-1) * MAX_CHARGE
    elif r <= (l - 1) * MAX_CHARGE:
        #return alpha - get_time_varying_p0(t, period=24)
        return alpha - p_0
    else:
        return alpha - p_0 + beta**(l-1) * delta_f(r - (l - 1) * MAX_CHARGE)


def idx_xu_order(state, power,t):
    """基于index_xu的简化策略"""
    N = len(state) // 2
    action = [0] * N
    available_power = power
    chargers = []

    # 收集所有充电桩的index值
    for i in range(N):
        r = state[2 * i]
        l = state[2 * i + 1]
        chargers.append((idx_xu(r, l, t), i))  # (index值, 充电桩索引)

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
        action = idx_xu_order(current_state, power,t)
        actions.append(action)

        reward = reward_function(current_state, action,t)
        total_reward += beta**(t) * reward

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


# Index policy

def index(r: int, l: int, t: int, i: int) -> float:
    # R < (L-1)MAX_CHARGE
    if r <= (l - 1) * MAX_CHARGE:
        return 0

    # R > LW
    if r > l * MAX_CHARGE:
        if 0 <= i <= MAX_CHARGE - 1:
           return  alpha-p_0+ beta**(l-1)*delta_f(r - (l - 1) * MAX_CHARGE - i)
           # return alpha - get_time_varying_p0(t, period=24) +  delta_f(r - (l - 1) * MAX_CHARGE - i)
        else:
            return 0

    # (L-1)W < R ≤ LW
    max_i = r - (l - 1) * MAX_CHARGE - 1
    if 0 <= i <= max_i:
        return alpha - p_0 +  beta**(l-1)*delta_f(r - (l - 1) * MAX_CHARGE - i)
    else:
        return 0


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

        reward = reward_function(current_state, action,t)
        total_reward +=  beta**(t) * reward

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    return total_reward, states, actions


if __name__ == "__main__":
    results = []
    charger_states = []
    alpha = 3  # 单位充电收益
    p_0=1
    T = 500
    beta=0.95
    max_r = max(r_dist)
    max_l = max(l_dist)
    MAX_CHARGE = max_r / 2  # 充电桩最大容量 W
    N = 10
    for penalty_weight in [0.2,0.4,0.6,0.8,1]:
        for power_ratio in [0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8, 0.9, 1.1, 1.2]:
            avg_l = sum(l_dist) / len(l_dist)
            avg_r = sum(r_dist) / len(r_dist)
            total_power = round(N * max_r / max_l * power_ratio)  # 总可用充电功率

            f = lambda x: (x ** 2) * penalty_weight
            delta_f = lambda x: f(x) - f(x - 1)
            print(
                f"power_ratio={power_ratio}, penalty_weight = {penalty_weight}, total_power = {total_power}\n")
            flat_state = tuple([0, 0] * N)
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
                current_arrival_seq = generate_arrival_sequence_poi()
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
                "blank ":'' ,
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
    df.to_excel("poisson_constant_price_(12,3).xlsx", index=False)
