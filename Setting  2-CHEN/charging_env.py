# charging_env.py
import numpy as np
import parameter_setting_CHEN106_1 as ps


class ChargingEnv:
    def __init__(self, N, power_ratio, penalty_weight, T=500):
        # 1. 绑定实验参数
        self.N = N
        self.power_ratio = power_ratio
        self.penalty_weight = penalty_weight
        self.T = T

        # 2. 自动计算功率限制
        self.total_power = round(self.N * (ps.max_r / ps.max_l) * self.power_ratio)
        self.avg_power = self.total_power / self.N

    # 3. 统一的惩罚与收益逻辑
    def penalty_func(self, x):
        return self.penalty_weight * (max(0, x) ** 2)

    def reward_func(self, s, a, t):
        R, L = s
        p_t = ps.get_time_varying_p0(t)
        if L > 0:
            charge_benefit = min(R, a) * (ps.alpha - p_t)
            penalty = self.penalty_func(R - a) if L == 1 else 0
            return charge_benefit - penalty
        return 0.0

    # 4. 矩阵预计算 (整合了你原先 r_beta 里的 precompute_matrices)
    def precompute_matrices(self):
        num_states = ps.NUM_STATES
        num_actions = len(ps.A_SPACE)
        P_mat = np.zeros((self.T, num_actions, num_states, num_states))
        R_mat = np.zeros((self.T, num_states, num_actions))

        for t in range(self.T):
            for s_idx, s in enumerate(ps.S_SPACE):
                for a_idx, a in enumerate(ps.A_SPACE):
                    # 算 Reward
                    R_mat[t, s_idx, a_idx] = self.reward_func(s, a, t)
                    R_val, L_val = s
                    if L_val > 1:
                        next_R = max(R_val - a, 0)
                        next_L = L_val - 1
                        next_s = (next_R, next_L)
                        if next_s in ps.S_TO_IDX:
                            next_idx = ps.S_TO_IDX[next_s]
                            P_mat[t, a_idx, s_idx, next_idx] = 1.0
                    else:
                        prob_arrival = ps.get_time_varying_prob(t)
                        next_idx_0 = ps.S_TO_IDX[(0, 0)]
                        P_mat[t, a_idx, s_idx, next_idx_0] += (1.0 - prob_arrival)
                        for r_idx, r_val in enumerate(ps.r_dist):
                            for l_idx, l_val in enumerate(ps.l_dist):
                                prob = prob_arrival * ps.r_p[r_idx] * ps.l_p[l_idx]
                                next_s = (r_val, l_val)
                                if next_s in ps.S_TO_IDX:
                                    next_idx = ps.S_TO_IDX[next_s]
                                    P_mat[t, a_idx, s_idx, next_idx] += prob
        return P_mat, R_mat


# ==========================================
# 独立测试模块
# ==========================================
if __name__ == "__main__":
    print("✅ 测试: ChargingEnv 环境类")
    # 创建一个测试环境
    test_env = ChargingEnv(N=10, power_ratio=0.5, penalty_weight=0.8, T=10)
    print(f"总功率限制: {test_env.total_power}, 单桩平均限制: {test_env.avg_power}")

    s_test = (5, 1)  # 需要5度电，还剩1个时刻
    a_test = 2  # 充2度电
    print(f"状态 {s_test} 采取动作 {a_test} 的收益: {test_env.reward_func(s_test, a_test, 0)}")

    P, R = test_env.precompute_matrices()
    print(f"成功生成矩阵 -> P_mat: {P.shape}, R_mat: {R.shape}")