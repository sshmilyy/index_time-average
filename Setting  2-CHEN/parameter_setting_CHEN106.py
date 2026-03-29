import numpy as np
import itertools
from scipy.stats import norm  # 【新增】需要引入正态分布生成
# --- 共享物理参数 ---
MAX_CHARGE = 3
T = 500
N = 10
alpha = 3  # 单位充电收益 (保持原样)

# --- A. 需求电量 R (Energy Demand) ---
r_dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
r_p = [0.084046,0.119917, 0.151303,0.131780,0.108572,0.108605, 0.104358,   0.086877, 0.062384, 0.042158]

# --- B. 停留时长 L (Parking Duration) ---
l_dist = [1, 2, 3, 4, 5, 6]
l_p=[0.223000, 0.401011, 0.211504,0.129857,0.021825,0.012803]

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

"""
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
    return 1



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
"""
