import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import time

N = 10  # 充电桩数量
power_ratio = 0.10
VEHICLE_GEN_PROB = 0.7  # 车辆生成概率
penalty_weight = 0.8  # 惩罚函数权重
MAX_CHARGE = 5  # 充电桩最大容量 W
alpha = 3  # 单位充电收益
p_0 = 1  # 固定电价
T = 500

r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
r_p = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09]
l_dist = [1, 2, 3, 4, 5]
l_p = [0.2, 0.2, 0.2, 0.2, 0.2]

avg_l = sum(l_dist) / len(l_dist)
avg_r = sum(r_dist) / len(r_dist)
total_power = round(N * max(r_dist) * power_ratio)  # 总可用充电功率
max_r = max(r_dist)
max_l = max(l_dist)
f = lambda x: (x ** 2) * penalty_weight
delta_f = lambda x: f(x) - f(x - 1)


def generate_arrival_sequence():
    return [
        [  # 每个时间步的到达情况
            (np.random.choice(r_dist, p=r_p), np.random.choice(l_dist, p=l_p))
            if random.random() < VEHICLE_GEN_PROB else None
            for _ in range(N)
        ]
        for _ in range(T)
    ]


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






def cvt_policy(arrival_seq):

    dict1, dict2 = process_sequence(arrival_seq)
    r = {(n + 1, t + 1): v for (t, n), v in dict1.items()}
    l = {(n + 1, t + 1): v for (t, n), v in dict2.items()}

    # 辅助变量 β[n,t]
    beta = {(n, t): penalty_weight if t == 1 else (penalty_weight if l[n, t] > l[n, t-1] else 0) for n in range(1,N+1) for t in range(1,T+1)}

    # 初始化模型
    model = gp.Model("QIP")

    # 决策变量：w[n,t] ∈ [0, w_0] ∩ ℕ
    w = model.addVars(range(1,N+1), range(1,T+1), vtype=GRB.INTEGER, lb=0, ub=MAX_CHARGE, name="w")


    # 目标函数
    model.setObjective(
        gp.quicksum(
            (alpha - p_0) * gp.quicksum(
                w[n, t] - beta[n, t] * (r[n, t] - gp.quicksum(w[n, s] for s in range(t, min(t + l[n, t], T + 1)))) ** 2
                for n in range(1,N+1)
            )
            for t in range(1,T+1)
        ),
        GRB.MAXIMIZE
    )


    # 约束 (4): ∑_n w[n,t] ≤ W
    for t in range(1,T+1):
        model.addConstr(gp.quicksum(w[n, t] for n in range(1,N+1)) <= total_power, name=f"c4_{t}")

    # 约束 (5): r[n,t] - ∑_{s=t}^{t+l[n,t]-1} w[n,s] ≥ 0
    for n in range(1,N+1):
        for t in range(1,T+1):
            model.addConstr(
                r[n, t] - gp.quicksum(w[n, s] for s in range(t, min(t + l[n, t], T + 1))) >= 0,
                name=f"c5_{n}_{t}"
            )

    # 求解
    model.setParam("OutputFlag", 1)
    model.setParam("MIPGap", 0.01)  # 设置 0.5% gap
    model.setParam("TimeLimit", 30)
    model.optimize()
    model.optimize()
    sol = model.getAttr('X', w)
    return sol ,model.objVal

def cvt_cts_policy(arrival_seq):

    dict1, dict2 = process_sequence(arrival_seq)
    r = {(n + 1, t + 1): v for (t, n), v in dict1.items()}
    l = {(n + 1, t + 1): v for (t, n), v in dict2.items()}

    # 辅助变量 β[n,t]
    beta = {(n, t): penalty_weight if t == 1 else (penalty_weight if l[n, t] > l[n, t-1] else 0) for n in range(1,N+1) for t in range(1,T+1)}

    # 初始化模型
    model = gp.Model("QIP")

    # 决策变量：w[n,t] ∈ [0, w_0] ∩ ℕ
    w = model.addVars(range(1,N+1), range(1,T+1), vtype=GRB.CONTINUOUS, lb=0, ub=MAX_CHARGE, name="w")


    # 目标函数
    model.setObjective(
        gp.quicksum(
            (alpha - p_0) * gp.quicksum(
                w[n, t] - beta[n, t] * (r[n, t] - gp.quicksum(w[n, s] for s in range(t, min(t + l[n, t], T + 1)))) ** 2
                for n in range(1,N+1)
            )
            for t in range(1,T+1)
        ),
        GRB.MAXIMIZE
    )


    # 约束 (4): ∑_n w[n,t] ≤ W
    for t in range(1,T+1):
        model.addConstr(gp.quicksum(w[n, t] for n in range(1,N+1)) <= total_power, name=f"c4_{t}")

    # 约束 (5): r[n,t] - ∑_{s=t}^{t+l[n,t]-1} w[n,s] ≥ 0
    for n in range(1,N+1):
        for t in range(1,T+1):
            model.addConstr(
                r[n, t] - gp.quicksum(w[n, s] for s in range(t, min(t + l[n, t], T + 1))) >= 0,
                name=f"c5_{n}_{t}"
            )

    # 求解
    model.setParam("OutputFlag", 1)
    model.setParam("MIPGap", 0)  # 设置 0.5% gap
    model.optimize()
    sol = model.getAttr('X', w)
    return sol ,model.objVal


NUM = 1000
index_results = []
cvt_results = []
for exp_id in range(NUM):
    start_total = time.perf_counter()
    current_arrival_seq = generate_arrival_sequence()
    print(f"第{exp_id}次实验\n")
    solution, cvt_reward = cvt_policy(current_arrival_seq)
    print(cvt_reward)
    print(f"{solution[1,2]}")
    total_time = time.perf_counter() - start_total
    print(f"总运行时间: {total_time:.2f}s\n")