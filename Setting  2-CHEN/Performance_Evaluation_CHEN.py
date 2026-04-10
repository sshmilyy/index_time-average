import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pandas import DataFrame
from scipy.stats import t
import matplotlib.pyplot as plt
from parameter_setting_CHEN106 import MAX_CHARGE, r_dist, l_dist, r_p, l_p, max_r, max_l, S_TO_IDX, N, T, alpha,get_time_varying_prob,get_time_varying_p0,period

# ==========================================
# 1. Chen et al. (2024) Simulation Settings
# Based on "My Electric Avenue" (Residential/Public)
# ==========================================
# ==========================================
# 2. Functions adapted to Chen et al. (2024)
# ==========================================

"""
def get_time_varying_p0(t):
    t_in_period = t % period
    if t_in_period <= 6:
        return 0.67
    elif t_in_period <= 22:
        return 1.33
    elif t_in_period <= 23:
        return 0.67
"""

"""

def get_time_varying_prob(t):

    模拟住宅区/公共充电到达率 (Residential Profile)。
    特征：
    1. 深夜 (0-6点) 几乎为 0。
    2. 上午 (8-11点) 有一个小高峰 (外出办事)。
    3. 傍晚 (16-19点) 是主高峰 (下班回家)。

    hour_of_day = t % 24

    # 使用两个高斯分布叠加来模拟到达率
    # 下班高峰：均值 17:30 (17.5), 标准差 2.0
    evening_peak = 0.8 * np.exp(-((hour_of_day - 17.5) ** 2) / (2 * 2.0 ** 2))

    # 上午小高峰：均值 09:00 (9.0), 标准差 1.5
    morning_peak = 0.3 * np.exp(-((hour_of_day - 9.0) ** 2) / (2 * 1.5 ** 2))

    # 基础噪声
    base_rate = 0.05

    prob = base_rate + evening_peak + morning_peak
    return max(0.0, min(1.0, prob))

"""
def f(x):
    return (x ** 2) * penalty_weight

def delta_f(x):
    return f(x) - f(x - 1)


def generate_arrival_sequence_poi():
    """
    生成泊松到达序列
    """
    arrivals = []
    for t in range(T):
        # 计算 Lambda
        prob = get_time_varying_prob(t)
        # 调整 N 的系数以匹配论文的繁忙程度 (Load Factor)
        # 论文中使用了 20-60 个充电桩，我们可以适当提高到达密度
        lam = N * prob * 0.5

        num_arrivals = np.random.poisson(lam)

        arrivals_at_t = []
        for _ in range(num_arrivals):
            r = np.random.choice(r_dist, p=r_p)
            l = np.random.choice(l_dist, p=l_p)

            # 逻辑修正：如果需求 > 时间*功率，截断需求 (或者假设用户充不满也走)
            # Chen et al. 的论文是做 Robust Optimization，允许需求未满足(有惩罚)
            # 这里我们保持 r 不变，让你的算法去处理 penalty
            arrivals_at_t.append((r, l))

        arrivals.append(arrivals_at_t)
    return arrivals

print(generate_arrival_sequence_poi())
# ==========================================
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
    current_p=get_time_varying_p0(t)
    rewards = [
        (min(r, a) * (alpha - current_p) - (f(max(0, r - a)) if l == 1 else 0))
        for r, l, a in zip(current_state[::2],
                           current_state[1::2],
                           action)
    ]
    return sum(rewards)


def run_experiments(algorithm, arrival_seq, initial, power, table=None, eval_window=100):
    """
    统一运行各种算法的包裹函数，现已支持返回 step_rewards 以供分析和画图。
    """
    if algorithm == "index":
        # 解包 4 个返回值
        avg_reward, states, actions, step_rewards = simulate_idx(
            T, initial, power, arrival_seq, table, eval_window
        )
    else:
        simulate_func = {
            'new': simulate_new,
            'gdy': simulate_gdy,
            'llf': simulate_llf,
            'lrf': simulate_lrf
        }[algorithm]

        # 解包 4 个返回值
        avg_reward, states, actions, step_rewards = simulate_func(
            T, initial, power, arrival_seq, eval_window
        )

    return avg_reward, states, actions, step_rewards


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


def simulate_new(T, initial_state, power, arrival_seq, eval_window=100):
    step_rewards = []
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = new_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action,t)
        step_rewards.append(reward)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state
    eval_window = min(eval_window, T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards

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


def cvt_cts_policy(arrival_seq,eval_window=100):
    dict1, dict2 = process_sequence(arrival_seq)
    r = {(n + 1, t + 1): v for (t, n), v in dict1.items()}
    l = {(n + 1, t + 1): v for (t, n), v in dict2.items()}
    gamma = {(n, t): penalty_weight for n in range(1, N + 1) for t in range(1, T + 1)}
    model = gp.Model()
    w = model.addVars(range(1, N + 1), range(1, T + 1), vtype=GRB.CONTINUOUS, lb=0, ub=MAX_CHARGE, name="w")
    obj = gp.LinExpr()

    for t in range(1, T + 1):
        current_p = get_time_varying_p0(t - 1)
        for n in range(1, N + 1):
            obj += (alpha - current_p) * w[n, t]
            if r.get((n, t), 0) > 0:
                penalty = gamma[n, t] * (r[n, t] - gp.quicksum(w[n, s] for s in range(t, min(t + l[n, t], T + 1)))) ** 2
                obj -=penalty
    model.setObjective(obj, GRB.MAXIMIZE)

    # 约束 (4): ∑_n w[n,t] ≤ W
    for t in range(1, T + 1):
        model.addConstr(gp.quicksum(w[n, t] for n in range(1, N + 1)) <= total_power, name=f"c4_{t}")

    # 约束 (5): r[n,t] - ∑_{s=t}^{t+l[n,t]-1} w[n,s] ≥ 0
    for n in range(1, N + 1):
        for t in range(1, T + 1):
            if r.get((n, t), 0) > 0:
                arrival_time = t
                duration = l[(n, t)]
                end_time = min(arrival_time + duration, T + 1)
                model.addConstr(
                    r[n, t] - gp.quicksum(w[n, s] for s in range(arrival_time, end_time)) >= 0,
                    name=f"c5_{n}_{t}"
                )
    # 求解
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 0.001)  # 设置 0.5% gap
    model.optimize()
    sol = model.getAttr('X', w)
    eval_window = min(eval_window, T)
    start_t = T - eval_window + 1  # 对应 Gurobi 索引的起始时间 (1-based index)
    steady_reward = 0

    for t in range(start_t, T + 1):
        # 【修正】: 这里原先误用了 get_time_varying_prob，应该和 reward_function 保持一致使用 p0
        current_p = get_time_varying_p0(t - 1)
        for n in range(1, N + 1):
            steady_reward += (alpha - current_p) * sol[n, t]

            # 计算惩罚项
            if r.get((n, t), 0) > 0:
                duration = l[(n, t)]
                penalty = gamma[n, t] * (r[n, t] - sum(sol[n, s] for s in range(t, min(t + duration, T + 1))
                )) ** 2
                steady_reward -= penalty

    avg_steady_reward = steady_reward / eval_window
    return sol, avg_steady_reward

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


def simulate_llf(T, initial_state, power, arrival_seq, eval_window=100):
    step_rewards = []
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = llf_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action,t)
        step_rewards.append(reward)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state
    eval_window = min(eval_window, T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards

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


def simulate_gdy(T, initial_state, power, arrival_seq, eval_window=100):
    step_rewards = []
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = gdy_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action,t)
        step_rewards.append(reward)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    eval_window = min(eval_window, T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards


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


def simulate_lrf(T, initial_state, power, arrival_seq, eval_window=100):
    step_rewards = []
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = lrf_policy(current_state, power)
        actions.append(action)

        reward = reward_function(current_state, action,t)
        step_rewards.append(reward)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    eval_window = min(eval_window, T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards



# Index policy

##index by algorithm







def index_algorithm(state, action, t, table):
    periodic = t % 24
    s_idx = S_TO_IDX[state]
    return table[s_idx, int(periodic),int(action)]



def greedy_charging_strategy(current_state, t, power,table):
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
                current_index = index_algorithm((r,l),current_a,t,table)
                candidates.append((current_index, i))

        if not candidates:
            break

        # 选择指数最大的充电桩
        selected = max(candidates, key=lambda x: x[0])
        final_action[selected[1]] += 1
        remaining_power -= 1

    return tuple(final_action)


def simulate_idx(T, initial_state, power, arrival_seq,table, eval_window=100):
    step_rewards = []
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        action = greedy_charging_strategy(current_state, t, power,table)
        actions.append(action)

        reward = reward_function(current_state, action,t)
        step_rewards.append(reward)

        next_state = transition_probability_simu(current_state, action, t, arrival_seq)
        states.append(next_state)
        current_state = next_state

    eval_window = min(eval_window, T)
    avg_reward = sum(step_rewards[-eval_window:]) / eval_window
    return avg_reward, states, actions, step_rewards


def plot_reward_dynamics(T, flat_state, total_power, INDEX_TABLE, eval_window=100):
    """
    运行单次仿真并绘制各算法单步收益（Reward）随时间的变化图。
    """
    print("正在运行单局仿真以绘制稳态验证图...")

    # 1. 生成单次到达序列
    arrival_seq = generate_arrival_sequence_poi()

    # 2. 运行各算法并获取 step_rewards 列表 (假设 run_experiments 已经更新为返回四个值)
    _, _, _, idx_steps = run_experiments('index', arrival_seq, flat_state, total_power, table=INDEX_TABLE,
                                         eval_window=eval_window)
    _, _, _, gdy_steps = run_experiments('gdy', arrival_seq, flat_state, total_power, eval_window=eval_window)
    _, _, _, llf_steps = run_experiments('llf', arrival_seq, flat_state, total_power, eval_window=eval_window)
    _, _, _, lrf_steps = run_experiments('lrf', arrival_seq, flat_state, total_power, eval_window=eval_window)
    _, _, _, new_steps = run_experiments('new', arrival_seq, flat_state, total_power, eval_window=eval_window)

    # 3. 准备绘图数据
    algorithms = {
        'Index': idx_steps,
        'Greedy': gdy_steps,
        'LLF': llf_steps,
        'LRF': lrf_steps,
        'New': new_steps
    }

    colors = {'Index': 'red', 'Greedy': 'blue', 'LLF': 'green', 'LRF': 'orange', 'New': 'purple'}

    # 设置滑动窗口大小（比如 24 期即一天，用来平滑泊松到达带来的噪音）
    ma_window = 24

    plt.figure(figsize=(14, 7))

    for name, rewards in algorithms.items():
        # 绘制原始数据的散点/细线（设置透明度 alpha=0.15 弱化背景噪音）
        plt.plot(range(T), rewards, color=colors[name], alpha=0.15, label=f'{name} (Raw)')

        # 计算并绘制移动平均线 (Moving Average)
        if len(rewards) >= ma_window:
            ma = [sum(rewards[i - ma_window:i]) / ma_window for i in range(ma_window, len(rewards) + 1)]
            # 补齐前端的长度以便对齐 x 轴
            ma_aligned = [None] * (ma_window - 1) + ma
            plt.plot(range(T), ma_aligned, color=colors[name], linewidth=2, label=f'{name} (MA-{ma_window})')

    # 添加一条垂直的虚线，标示你提取稳态平均值的起点
    start_eval_t = T - eval_window
    plt.axvline(x=start_eval_t, color='black', linestyle='--', linewidth=2,
                label=f'Evaluation Start (T={start_eval_t})')
    plt.axvspan(start_eval_t, T, color='gray', alpha=0.1)  # 把评估区域标灰

    plt.title(f'Step Reward Dynamics over Time (T={T}, Power={total_power})')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Reward')
    # 因为图例可能很多，只显示平滑线和参考线的图例
    handles, labels = plt.gca().get_legend_handles_labels()
    filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'Raw' not in l]
    plt.legend([h for h, l in filtered_handles_labels], [l for h, l in filtered_handles_labels], loc='lower right')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    results = []
    charger_states = []

    for penalty_weight in [0.2,0.4,0.6,0.8]:

        file_name =f"index_cache_T24_penal{penalty_weight}.npy"
        print(f"Loading index file: {file_name}")
        INDEX_TABLE = np.load(file_name)
        for power_ratio in [0.2,0.4,0.6,0.8,1.0,1.2,1.5,1.7,2.0]:
            avg_l = sum(l_dist) / len(l_dist)
            avg_r = sum(r_dist) / len(r_dist)
            total_power = round(N * max_r / max_l * power_ratio)  # 总可用充电功率


            print(
                f"power_ratio={power_ratio}, penalty_weight = {penalty_weight}, total_power = {total_power}\n")
            flat_state = tuple([0, 0] * N)
            NUM = 50
            EVAL_WINDOW = 100
            plot_reward_dynamics(T, flat_state, total_power, INDEX_TABLE, eval_window=EVAL_WINDOW)
            index_results = []
            gdy_results = []
            llf_results = []
            lrf_results = []
            new_results = []
            cvt_results = []

            index_gap_results = []
            gdy_gap_results = []
            llf_gap_results = []
            lrf_gap_results = []
            new_gap_results = []
            cvt_gap_results = []
            for exp_id in range(NUM):
                current_arrival_seq = generate_arrival_sequence_poi()
                _, cvt_reward = cvt_cts_policy(current_arrival_seq)
                index_reward = run_experiments('index', current_arrival_seq, flat_state, total_power, table=INDEX_TABLE,eval_window=EVAL_WINDOW)[0]
                gdy_reward = run_experiments('gdy', current_arrival_seq, flat_state, total_power,eval_window=EVAL_WINDOW)[0]
                llf_reward = run_experiments('llf', current_arrival_seq, flat_state, total_power,eval_window=EVAL_WINDOW)[0]
                lrf_reward = run_experiments('lrf', current_arrival_seq, flat_state, total_power,eval_window=EVAL_WINDOW)[0]
                new_reward = run_experiments('new', current_arrival_seq, flat_state, total_power,eval_window=EVAL_WINDOW)[0]
                index_gap = abs(cvt_reward - index_reward) / abs(cvt_reward) if cvt_reward != 0 else 0
                new_gap = abs(cvt_reward - new_reward) / abs(cvt_reward) if cvt_reward != 0 else 0
                lrf_gap = abs(cvt_reward - lrf_reward) / abs(cvt_reward) if cvt_reward != 0 else 0
                gdy_gap = abs(cvt_reward - gdy_reward) / abs(cvt_reward) if cvt_reward != 0 else 0
                llf_gap = abs(cvt_reward - llf_reward) / abs(cvt_reward) if cvt_reward != 0 else 0
                # 将结果添加到列表
                cvt_results.append(cvt_reward)  # 添加这一行
                index_results.append(index_reward)
                gdy_results.append(gdy_reward)
                llf_results.append(llf_reward)
                lrf_results.append(lrf_reward)
                new_results.append(new_reward)
                # 将gap结果添加到列表
                index_gap_results.append(index_gap)
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
                "gdy_mean": gdy_mean,
                "llf_mean": llf_mean,
                "lrf_mean": lrf_mean,
                "new_mean": new_mean,
                "blank": '',
                "index_gap": diff_IDX_mean,
                "gdy_gap": diff_GDY_mean,
                "llf_gap": diff_LLF_mean,
                "lrf_gap": diff_LRF_mean,
                "new_gap": diff_NEW_mean,
                "blank  ": '',
                "index_gap_p": CI_IDX,
                "gdy_gap_p": CI_GDY,
                "llf_gap_p": CI_LLF,
                "lrf_gap_p": CI_LRF,
                "new_gap_p": CI_NEW,
            }
            results.append(row)
            df = DataFrame(results)
    df.to_excel(f"Performance_Gap_CHEN(10,6)_penal={penalty_weight}.xlsx", index=False)
