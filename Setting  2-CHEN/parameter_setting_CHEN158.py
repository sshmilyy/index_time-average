import numpy as np
import itertools
from scipy.stats import norm  # 【新增】需要引入正态分布生成
# --- 共享物理参数 ---
MAX_CHARGE = 3
T = 500
N = 10
alpha = 3  # 单位充电收益 (保持原样)

# --- A. 需求电量 R (Energy Demand) ---
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
r_p = [0.07595281, 0.10961129, 0.14108162, 0.12111605, 0.09865113, 0.09883591,
 0.09493615, 0.07896759, 0.0572904, 0.03906562, 0.03097435, 0.02247464,
 0.01454871, 0.00955975, 0.00693398]

# --- B. 停留时长 L (Parking Duration) ---
l_dist = [1, 2, 3, 4, 5, 6, 7, 8]
l_p=[0.20072549, 0.36556546, 0.2357552, 0.14354207, 0.02441966, 0.01296352,
 0.00939442, 0.00763418]

# --- C. 全局参数设置 ---
period = 24
max_r = max(r_dist)
max_l = max(l_dist)


# --- 2. 有效状态空间 ---
# (保持不变)
R_ACTIVE_SPACE = list(range(max_r + 1))
L_ACTIVE_SPACE = list(range(1, max_l + 1))
S_ACTIVE = list(itertools.product(R_ACTIVE_SPACE, L_ACTIVE_SPACE))
S_SPACE = [(0, 0)] + S_ACTIVE
S_TO_IDX = {s: i for i, s in enumerate(S_SPACE)}
NUM_STATES = len(S_SPACE)
A_SPACE = list(range(int(MAX_CHARGE) + 1))




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
    t_in_period = t % period
    if t_in_period <= 6:
        return 0.67
    elif t_in_period <= 22:
        return 1.33
    elif t_in_period <= 23:
        return 0.67
"""

"""
def get_time_varying_prob(t, period=24, min_prob=0.4, max_prob=0.6):
    t_in_period = t % period
    half_period = period / 2
    prob_range = max_prob - min_prob
    if t_in_period <= half_period:
        return min_prob + prob_range * (t_in_period / half_period)
    else:
        return min_prob + prob_range * ((period - t_in_period) / half_period)
"""

def get_time_varying_prob(t):
    hour_of_day = t % 24
    prob_values = [0.08736682, 0.08148148, 0.04840183, 0.03845764, 0.03926941, 0.06889904,
                   0.14733638, 0.28391679, 0.5398275, 0.71283612, 0.75596144, 0.80913242,
                   0.91872146, 0.91598174, 0.86037544, 0.80375444, 0.80892948, 0.77148656,
                   0.72683917, 0.57138508, 0.42049721, 0.28533739, 0.20943683, 0.12897007]
    return prob_values[hour_of_day]

