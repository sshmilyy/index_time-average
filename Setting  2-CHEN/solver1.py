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
# 第二部分：核心求解器 Class (终极 Numpy 矩阵加速版)
# ==========================================================

class TimeVaryingWhittleSolver:
    def __init__(self, penalty_weight, T_period):
        self.penalty_weight = penalty_weight
        self.T = T_period
        self.filename = f"index_cache_T{self.T}_penal{self.penalty_weight}.npy"
        self.index_table = None

        # --- 矩阵预分配 ---
        # R_base: 基础收益矩阵 (T, 状态数, 动作数)
        self.R_base = np.zeros((self.T, NUM_STATES, MAX_CHARGE + 1))
        # P_mat: 转移概率巨型矩阵 (T, 当前状态, 动作, 下一状态)
        self.P_mat = np.zeros((self.T, NUM_STATES, MAX_CHARGE + 1, NUM_STATES))
        # A_mat: 动作矩阵 (用于快速减去 nu * a)
        self.A_mat = np.zeros((self.T, NUM_STATES, MAX_CHARGE + 1))

        self._precompute_matrices()

    def _precompute_matrices(self):
        print(f"初始化 Solver (Penalty={self.penalty_weight}): 正在构建底层 Numpy 加速矩阵...")
        for t in range(self.T):
            for s_idx, s in enumerate(S_SPACE):
                for a in range(MAX_CHARGE + 1):
                    # 1. 记录动作值
                    self.A_mat[t, s_idx, a] = a

                    # 2. 缓存基础收益 (令 nu=0)
                    self.R_base[t, s_idx, a] = get_reward(s, a, 0, t)

                    # 3. 填入转移概率矩阵
                    trans_list = get_transitions(s, a, t)
                    if trans_list:
                        for ns, p in trans_list:
                            if ns == 0:
                                ns = (0, 0)
                            ns_idx = S_TO_IDX[ns]
                            self.P_mat[t, s_idx, a, ns_idx] = p

        print("巨型矩阵构建完成！即将开始光速计算...")

    def _solve_rvi(self, nu, h_init=None):
        """相对价值迭代 (完全矩阵化)"""
        tol = 1e-3
        if h_init is None:
            h = np.zeros((self.T, NUM_STATES))
        else:
            h = h_init.copy()
        # 提前算出这一轮的奖励矩阵 (基础奖励 - nu * 动作)
        R_nu = self.R_base - nu * self.A_mat

        for _ in range(50):
            h_new = np.zeros((self.T, NUM_STATES))
            for t in reversed(range(self.T)):
                t_next = (t + 1) % self.T

                # 🚀 核心加速：用 Numpy 矩阵点乘代替所有 for 循环！
                # P_mat[t] 的形状是 (状态, 动作, 下一状态)
                # h[t_next] 的形状是 (下一状态, )
                # np.dot 会瞬间算出所有状态和所有动作下的 future_v，形状为 (状态, 动作)
                future_v = np.dot(self.P_mat[t], h[t_next])

                # 总价值矩阵 = 立即收益 + 未来期望
                q_vals = R_nu[t] + future_v

                # 找出每个状态下最优的动作价值 (按动作维度取最大值)
                h_new[t] = np.max(q_vals, axis=1)

            rho = h_new[0, 0]
            h_new -= rho

            if np.max(np.abs(h_new - h)) < tol:
                h = h_new.copy()
                break
            h = h_new.copy()

        return h

    def _find_index_minimized(self, s, k, t):
        """二分查找 Whittle Index"""
        low, high = 0, 20.0
        epsilon = 1e-2
        s_idx = S_TO_IDX[s]

        # 🚀 优化：二分查找 40 次已达小数点后十二位精度，完全足够
        for _ in range(20):
            mid = (low + high) / 2
            h = self._solve_rvi(mid)

            t_next = (t + 1) % self.T

            # 计算当前动作 k 的 Q 值
            future_v_k = np.dot(self.P_mat[t, s_idx, k], h[t_next])
            q_k = self.R_base[t, s_idx, k] - mid * k + future_v_k

            # 计算下一个动作 k+1 的 Q 值
            future_v_k1 = np.dot(self.P_mat[t, s_idx, k + 1], h[t_next])
            q_k1 = self.R_base[t, s_idx, k + 1] - mid * (k + 1) + future_v_k1

            if q_k >= q_k1 - epsilon:
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

        tasks = []
        for s_idx, s in enumerate(S_SPACE):
            r, l = s
            if r == 0: continue
            for t in range(self.T):
                for k in range(MAX_CHARGE):
                    tasks.append((s_idx, s, t, k))

        total_tasks = len(tasks)
        print(f"总任务数: {total_tasks}。启动多核并行加速...")

        completed = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # chunksize=50 可以让任务分发效率更高
            results = executor.map(self._compute_single_task, tasks, chunksize=50)

            for s_idx, t, k, val in results:
                table[s_idx, t, k] = val
                completed += 1
                # 打印进度更新频率稍微调低，防止霸屏
                if completed % 500 == 0:
                    print(f"计算进度: {completed} / {total_tasks} ({(completed / total_tasks) * 100:.1f}%)")

        print(f"全量计算完成！总耗时: {time.time() - start_time:.2f} 秒")
        return table

    def get_index_table(self, force_recompute=False):
        """对外接口：智能获取 Index 表"""
        if not force_recompute and os.path.exists(self.filename):
            print(f"✅ 找到缓存文件 {self.filename}，直接闪电加载！")
            self.index_table = np.load(self.filename)
        else:
            self.index_table = self._calculate_all()
            np.save(self.filename, self.index_table)
            print(f"💾 计算结果已永久缓存至 {self.filename}")
        return self.index_table

    def display_index(self, target_l, target_t):
        """
        全景视角：查看固定 L 和 t 时，所有 R 和动作 a 对应的 Index
        """
        if self.index_table is None:
            self.get_index_table()

        print(f"\n" + "=" * 65)
        print(f"📊 Whittle Index 全景矩阵 (剩余时间 L={target_l}, 时刻 t={target_t})")
        print("=" * 65)

        # 1. 动态生成表头 (列是不同的动作边界)
        # 这里的动作 k 代表的是在动作 k 和 k+1 之间做选择的 Whittle Index
        header = f"{'需求(R)':<10}"
        for k in range(MAX_CHARGE):
            header += f"| {'动作 ' + str(k) + ' vs ' + str(k + 1):<15}"
        print(header)
        print("-" * 65)

        # 2. 收集并格式化每一行的数据 (行是不同的需求 R)
        results = []
        for s_idx, s in enumerate(S_SPACE):
            r, l = s
            # 只挑选 L 等于我们想要的目标 L 的状态
            if l == target_l:
                row_str = f"{r:<10}"
                for k in range(MAX_CHARGE):
                    val = self.index_table[s_idx, target_t, k]
                    row_str += f"| {val:<15.4f}"
                results.append((r, row_str))

        # 3. 按需求 R 从小到大排序并打印出来
        results = sorted(results, key=lambda x: x[0])
        for r, row_str in results:
            print(row_str)

        print("=" * 65 + "\n")


# ==========================================================
# 第三部分：测试与运行入口
# ==========================================================
if __name__ == "__main__":
    # 这里的 T_period 取决于你在 parameter_setting_CHEN106 中的 period
    T_period = period  # 假设你在 parameter_setting 里面定义了 period (比如 24)

    # 1. 实例化求解器，测试惩罚权重为 0.8 的情况
    solver = TimeVaryingWhittleSolver(penalty_weight=penalty_weight, T_period=T_period)

    # 2. 获取 Index 表（没有缓存就会自动多核计算，有缓存就秒加载）
    table = solver.get_index_table()

    # 3. 打印检查某个特定时间、特定截止时间下的 Index 变化
    # 例如：检查 T=12, 剩余时间 L=3, 动作 a=0 时，Index 随需求 R 的变化
    solver.display_index(target_l=2, target_t=1)