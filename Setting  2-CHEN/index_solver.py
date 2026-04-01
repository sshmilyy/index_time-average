import os
import numpy as np
import time
import concurrent.futures

# 导入你所有的参数设定
from parameter_setting_CHEN106 import *  # ==========================================================
penalty_weight=0.8

# 第一部分：基础 MDP 逻辑 (全局函数，方便多核调用)
# ==========================================================

def f(x):
    """惩罚函数"""
    return penalty_weight * (max(0, x) ** 2)


def get_reward(s, a, nu, t):
    """计算 t 时刻补贴后的收益"""
    P_COST = get_time_varying_p0(t)
    r, l = s
    if r == 0 or l == 0:
        return 0

    imm_reward = (alpha - P_COST) * min(a, r)
    # 只有在 L=1 (最后一刻) 且没充完电时才给惩罚
    penalty = f(max(0, r - a)) if l == 1 else 0

    # 加上拉格朗日乘子 (即 Whittle Index 补贴/惩罚)
    return imm_reward - penalty - nu * a

def get_arrival_prob(r, l):
    """从 parameter_setting 的分布中获取概率"""
    if r in r_dist and l in l_dist:
        ri = r_dist.index(r)
        li = l_dist.index(l)
        return r_p[ri] * l_p[li]
    return 0


def get_transitions(s, a, t):
    """状态转移概率"""
    r, l = s
    prob_arrival = get_time_varying_prob(t)

    if l > 1:
        next_s = (max(0, r - a), l - 1)
        return [(next_s, 1.0)]
    else:
        trans = []
        trans.append(((0, 0), 1.0 - prob_arrival))
        for rs in r_dist:
            for ls in l_dist:
                prob = prob_arrival * get_arrival_prob(rs, ls)
                if prob > 0:
                    trans.append(((rs, ls), prob))
        return trans


# ==========================================================
# 第二部分：核心求解器 Class
# ==========================================================

class TimeVaryingWhittleSolver:
    def __init__(self, penalty_weight, T_period):
        self.penalty_weight = penalty_weight
        self.T = T_period
        self.filename = f"index_cache_T{self.T}_penal{self.penalty_weight}.npy"
        self.index_table = None

        # 预计算转移概率（方案二加速：大幅减少重复计算）
        self.transition_cache = {}
        self._precompute_transitions()

    def _precompute_transitions(self):
        print(f"初始化 Solver (Penalty={self.penalty_weight}): 正在缓存转移概率...")
        for t in range(self.T):
            for s in S_SPACE:
                for a in range(MAX_CHARGE + 1):
                    trans_list = get_transitions(s, a, t)
                    if trans_list:
                        safe_trans_list = []
                        for ns, p in trans_list:
                            # 如果 ns 是整数 0，自动纠正为 (0,0)
                            if ns == 0:
                                ns = (0, 0)
                            # 强制转换为整数索引，杜绝 Numpy 切片错误
                            safe_trans_list.append((S_TO_IDX[ns], p))
                        self.transition_cache[(s, a, t)] = safe_trans_list
                    else:
                        self.transition_cache[(s, a, t)] = []
        print("转移概率缓存完成！")
    def _solve_rvi(self, nu):
        """相对价值迭代"""
        tol = 1e-4
        h = np.zeros((self.T, NUM_STATES))

        for _ in range(200):
            h_new = np.zeros((self.T, NUM_STATES))
            for t in reversed(range(self.T)):
                t_next = (t + 1) % self.T
                for s_idx, s in enumerate(S_SPACE):
                    q_vals = []
                    for a in range(MAX_CHARGE + 1):
                        r_imm = get_reward(s, a, nu, t)
                        # 极速版未来价值计算
                        future_v = sum(p * h[t_next, ns_idx] for ns_idx, p in self.transition_cache[(s, a, t)])
                        q_vals.append(r_imm + future_v)
                    h_new[t, s_idx] = max(q_vals)

            rho = h_new[0, 0]
            h_new -= rho

            if np.max(np.abs(h_new - h)) < tol:
                h = h_new.copy()
                break
            h = h_new.copy()
        return h

    def _find_index_minimized(self, s, k, t):
        """二分查找 Whittle Index"""
        low, high = -10.0, 20.0
        epsilon = 1e-7

        for _ in range(80):
            mid = (low + high) / 2
            h = self._solve_rvi(mid)

            def q_val(action, t_curr):
                t_next = (t_curr + 1) % self.T
                r_imm = get_reward(s, action, mid, t_curr)
                future_v = sum(p * h[t_next, ns_idx] for ns_idx, p in self.transition_cache[(s, action, t_curr)])
                return r_imm + future_v

            if q_val(k, t) >= q_val(k + 1, t) - epsilon:
                high = mid
            else:
                low = mid
        return high

    def _compute_single_task(self, args):
        """多核并行的单任务包裹函数"""
        s_idx, s, t, k = args
        val = self._find_index_minimized(s, k, t)
        return s_idx, t, k, val

    def _calculate_all(self):
        """执行全量计算并带有进度条"""
        print(f"未找到缓存 {self.filename}，准备执行计算...")
        start_time = time.time()
        table = np.zeros((NUM_STATES, self.T, MAX_CHARGE))

        # 收集所有需要计算的任务
        tasks = []
        for s_idx, s in enumerate(S_SPACE):
            r, l = s
            if r == 0: continue
            for t in range(self.T):
                for k in range(MAX_CHARGE):
                    tasks.append((s_idx, s, t, k))

        total_tasks = len(tasks)
        print(f"总任务数: {total_tasks}。启动多核并行加速...")

        # 启动并行池 (max_workers 默认使用所有可用 CPU 核心)
        completed = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._compute_single_task, tasks)

            for s_idx, t, k, val in results:
                table[s_idx, t, k] = val
                completed += 1
                if completed % 100 == 0:
                    print(f"计算进度: {completed} / {total_tasks} ({(completed / total_tasks) * 100:.1f}%)")

        print(f"全量计算完成！总耗时: {time.time() - start_time:.2f} 秒")
        return table

    def get_index_table(self, force_recompute=False):
        if not force_recompute and os.path.exists(self.filename):
            print(f"✅ 找到缓存文件 {self.filename}，直接闪电加载！")
            self.index_table = np.load(self.filename)
        else:
            self.index_table = self._calculate_all()
            np.save(self.filename, self.index_table)
            print(f"💾 计算结果已永久缓存至 {self.filename}")
        return self.index_table

    def check_index_values(self, target_l, target_t, action=0):
        """控制台打印查错工具"""
        if self.index_table is None:
            self.get_index_table()

        results = []
        for s_idx, s in enumerate(S_SPACE):
            r, l = s
            if l == target_l:
                num_idx = self.index_table[s_idx, target_t, action]
                results.append({'Demand_R': r, 'Numerical_Index': round(num_idx, 4)})

        results = sorted(results, key=lambda x: x['Demand_R'])

        print(f"\n" + "=" * 45)
        print(f"📊 Whittle Index 数值检查")
        print(f"设定条件: 时刻 t = {target_t}, 剩余时间 L = {target_l}, 动作 a = {action}")
        print("=" * 45)
        print(f"{'需求 (R)':<10} | {'数值 Index':<20}")
        print("-" * 45)
        for row in results:
            print(f"{row['Demand_R']:<10} | {row['Numerical_Index']:<20}")
        print("=" * 45 + "\n")


# ==========================================================
# 第三部分：测试与运行入口
# ==========================================================
if __name__ == "__main__":
    # 这里的 T_period 取决于你在 parameter_setting_CHEN106 中的 period
    T_period = 1  # 假设你在 parameter_setting 里面定义了 period (比如 24)

    # 1. 实例化求解器，测试惩罚权重为 0.8 的情况
    solver = TimeVaryingWhittleSolver(penalty_weight=0.8, T_period=T_period)

    # 2. 获取 Index 表（没有缓存就会自动多核计算，有缓存就秒加载）
    table = solver.get_index_table()

    # 3. 打印检查某个特定时间、特定截止时间下的 Index 变化
    # 例如：检查 T=12, 剩余时间 L=3, 动作 a=0 时，Index 随需求 R 的变化
    solver.check_index_values(target_l=2, target_t=0, action=0)