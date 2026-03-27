import numpy as np
import itertools
from scipy.stats import norm
# --- 共享物理参数 ---
MAX_CHARGE = 3
T = 500
N = 10
alpha = 3  # 单位充电收益 (保持原样)
beta = 0.95  # 折现因子 (Index计算建议接近1，论文通常用0.99)
# 只有这里定义一次，其他文件都 import 这个
# --- 1. 模型参数及分布
# 数据来源: ACN-Data (Caltech) - 样本数: 12043
# 拟合模型: Gaussian Mixture Model (k=2)
UNIT_R =2.0   # 1 r = 2.0 kWh
UNIT_L = 2.0   # 1 l = 2.0 Hour
max_r =14     # 18 * 3 = 54 kWh
max_l = 6     # 24 Hours
# Cluster 1: 短停
w1 = 0.8575
mu_r1 = 6.52 / UNIT_R      # ~2.17
sigma_r1 = 4.91 / UNIT_R   # ~1.64
mu_l1 = 5.31 / UNIT_L      # ~5.31
sigma_l1 = 3.33 / UNIT_L   # ~3.33

# Cluster 2: 长停
w2 = 0.1425
mu_r2 = 23.74 / UNIT_R     # ~7.91
sigma_r2 = 13.67 / UNIT_R  # ~4.56
mu_l2 = 8.92 / UNIT_L      # ~8.92
sigma_l2 = 4.35 / UNIT_L   # ~4.35


# --- C. 全局参数设置 ---
period = 24
# --- 2. 有效状态空间 ---
# (保持不变)
R_VALUES = np.arange(1, max_r + 1, dtype=int)
L_VALUES = np.arange(1, max_l + 1, dtype=int)

R_ACTIVE_SPACE = list(range(max_r + 1))
L_ACTIVE_SPACE = list(range(1, max_l + 1))
S_ACTIVE = list(itertools.product(R_ACTIVE_SPACE, L_ACTIVE_SPACE))
S_SPACE = [(0, 0)] + S_ACTIVE
S_TO_IDX = {s: i for i, s in enumerate(S_SPACE)}
NUM_STATES = len(S_SPACE)
A_SPACE = list(range(int(MAX_CHARGE) + 1))


def get_discrete_gmm_probs(x_values, w1, mu1, sig1, w2, mu2, sig2):
    """
    计算离散点的概率，并强制归一化
    """
    # 1. 计算每个整数点上的概率密度 (PDF)
    p1 = norm.pdf(x_values, loc=mu1, scale=sig1)
    p2 = norm.pdf(x_values, loc=mu2, scale=sig2)

    # 2. 加权混合
    p_total = w1 * p1 + w2 * p2
    #print(p_total)
    #print(sum(p_total))
    # 3. 【关键】归一化 (Normalization)
    # 使得所有整数点的概率之和严格等于 1.0
    prob_sum = np.sum(p_total)
    if prob_sum > 0:
        return p_total / prob_sum
    else:
        # 防止除以0的极端情况，返回均匀分布
        return np.ones_like(x_values) / len(x_values)

r_probs = get_discrete_gmm_probs(R_VALUES, w1, mu_r1, sigma_r1, w2, mu_r2, sigma_r2)
l_probs = get_discrete_gmm_probs(L_VALUES, w1, mu_l1, sigma_l1, w2, mu_l2, sigma_l2)
r_dist = R_VALUES.tolist()
l_dist = L_VALUES.tolist()
r_p = r_probs.tolist()
l_p = l_probs.tolist()
def generate_arrival_sequence_gmm():
    """
    基于 ACN-Data 拟合参数生成合成数据
    """
    arrival_seq = [[] for _ in range(T)]
    for t in range(T):
        lam = get_time_varying_prob(t)*N
        num_arrivals = np.random.poisson(lam)

        for _ in range(num_arrivals):
            if np.random.rand() < w1:
                # --- Cluster 1: 短停 ---
                r_sample = np.random.normal(mu_r1, sigma_r1)
                l_sample = np.random.normal(mu_l1, sigma_l1)
            else:
                # --- Cluster 2: 长停 ---
                r_sample = np.random.normal(mu_r2, sigma_r2)
                l_sample = np.random.normal(mu_l2, sigma_l2)

            # 3. 取整与截断 (Clamping)
            r_val = int(round(r_sample/3))
            l_val = int(round(l_sample))

            # 物理约束检查
            r_val = max(1, min(r_val, max_r))  # 限制在 [1, max_r]

            # 必须保证 l 足够充完 r (最小物理时间)
            min_l_needed = int(np.ceil(r_val / MAX_CHARGE))
            l_val = max(l_val, min_l_needed)
            l_val = min(l_val, max_l)  # 限制在 max_l

            arrival_seq[t].append((r_val, l_val))

    return arrival_seq

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


def get_time_varying_prob(t):
    hour_of_day = t % 24
    prob_values = [0.0808, 0.0354, 0.0177, 0.0126, 0.0076, 0.0227, 0.3636, 0.9192, 3.0732,
 6.7096, 5.1591, 2.1338, 1.8106, 2.2399, 1.5177, 0.9091, 0.8813, 0.9621,
 1.0783, 0.9773, 0.7146, 0.3561, 0.3131, 0.1162]
    return prob_values[hour_of_day]
