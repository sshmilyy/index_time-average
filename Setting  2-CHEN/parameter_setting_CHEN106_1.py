import numpy as np
import itertools

# --- A. 基础物理常量 ---
MAX_CHARGE = 3
alpha = 3  # 单位充电收益
period = 24

# --- B. 需求电量 R (Energy Demand) ---
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
r_p = [0.084046, 0.119917, 0.151303, 0.131780, 0.108572, 0.108605, 0.104358, 0.086877, 0.062384, 0.042158]
max_r = max(r_dist)

# --- C. 停留时长 L (Parking Duration) ---
l_dist = [1, 2, 3, 4, 5, 6]
l_p = [0.223000, 0.401011, 0.211504, 0.129857, 0.021825, 0.012803]
max_l = max(l_dist)

# --- D. 状态与动作空间 ---
R_ACTIVE_SPACE = list(range(max_r + 1))
L_ACTIVE_SPACE = list(range(1, max_l + 1))
S_ACTIVE = list(itertools.product(R_ACTIVE_SPACE, L_ACTIVE_SPACE))
S_SPACE = [(0, 0)] + S_ACTIVE
S_TO_IDX = {s: i for i, s in enumerate(S_SPACE)}
NUM_STATES = len(S_SPACE)
A_SPACE = list(range(int(MAX_CHARGE) + 1))


# --- E. 时间变动函数 ---
def get_time_varying_p0(t):
    hour = t % 24
    # 设定价格相对值
    p_peak = 1.6  # 高峰期成本权重
    p_off = 0.5  # 低谷期成本权重 (约为高峰的30%)

    # 下午 12:00 到 18:00 是高峰 (Chen et al. Setting)
    if 12 <= hour < 18:
        return p_peak
    else:
        return p_off


"""


def get_time_varying_p0(t):
    # 处理初始特殊值
    if t == 0:
        return 3.0
    if t == 1:
        return 0.5

    # 判断奇偶性
    if t % 2 == 0:
        # 偶数情况：从 t=0 (值为3) 开始，每2小时减 0.1
        # 计算距离 t=0 过了多少个“2小时”
        steps = t / 2
        return 3.0 - (steps * 0.1)
    else:
        # 奇数情况：从 t=1 (值为0.5) 开始，每2小时加 0.1
        # 计算距离 t=1 过了多少个“2小时”
        steps = (t - 1) / 2
        return 0.5 + (steps * 0.1)

def get_time_varying_p0(t):
    hour = t % 24
    # 设定价格相对值
    p_peak = 1.6  # 高峰期成本权重
    p_off = 0.5  # 低谷期成本权重 (约为高峰的30%)

    # 下午 12:00 到 18:00 是高峰 (Chen et al. Setting)
    if 12 <= hour < 18:
        return p_peak
    else:
        return p_off

def get_time_varying_p0(t):
    return 1

def get_time_varying_p0(t):
    hour = t % 24

    return hour*0.1+0.5
"""

"""
def get_time_varying_prob( t):
    return 0.5
"""


def get_time_varying_prob(t):
    hour_of_day = t % 24
    prob_values = [0.08736682, 0.08148148, 0.04840183, 0.03845764, 0.03926941, 0.06889904,
                   0.14733638, 0.28391679, 0.5398275, 0.71283612, 0.75596144, 0.80913242,
                   0.91872146, 0.91598174, 0.86037544, 0.80375444, 0.80892948, 0.77148656,
                   0.72683917, 0.57138508, 0.42049721, 0.28533739, 0.20943683, 0.12897007]
    return prob_values[hour_of_day]

