import os
import numpy as np
import time
import concurrent.futures
from pathlib import Path
# 1. 纯净导入基础物理常量
import parameter_setting_CHEN106 as ps


# ==========================================================
# 核心求解器 Class (完美融合 Env 架构 + Numpy 矩阵极速版)
# ==========================================================

class WhittleSolver:
    def __init__(self, env):
        # 1. 接入统一环境
        self.env = env

        self.save_dir = Path("Results_npy")
        self.save_dir.mkdir(parents=True, exist_ok=True)  # 如果不存在则创建文件夹

        # case1:constant, price=1.0,prob=0.5
        self.filename = f"index_cache_period={ps.period}_pw={self.env.penalty_weight}_const1_Bernoulli.npy"
        # case2:time varying p_0=1.6,0.5, and T=24 prob
        #self.filename = f"index_cache_period={ps.period}_pw={self.env.penalty_weight}_1605_Bernoulli.npy"
        # case3:test2
        #self.filename = f"index_cache_period={ps.period}_pw={self.env.penalty_weight}_test2_varying_Bernoulli.npy"

        self.file_path = self.save_dir / self.filename
        self.index_table = None

        # --- 矩阵预分配 ---
        # R_base: 基础收益矩阵 (T, 状态数, 动作数)
        self.R_base = np.zeros((ps.period, ps.NUM_STATES, ps.MAX_CHARGE + 1))
        # P_mat: 转移概率巨型矩阵 (T, 当前状态, 动作, 下一状态)
        self.P_mat = np.zeros((ps.period, ps.NUM_STATES, ps.MAX_CHARGE + 1, ps.NUM_STATES))
        # A_mat: 动作矩阵 (用于快速减去 nu * a)
        self.A_mat = np.zeros((ps.period, ps.NUM_STATES, ps.MAX_CHARGE + 1))

        # 初始化时自动构建底层矩阵
        self._precompute_matrices()

    def _get_arrival_prob(self, r, l):
        """内部方法：计算车辆到达概率"""
        if r in ps.r_dist and l in ps.l_dist:
            ri = ps.r_dist.index(r)
            li = ps.l_dist.index(l)
            return ps.r_p[ri] * ps.l_p[li]
        return 0

    def _get_transitions(self, s, a, t):
        """内部方法：获取状态转移概率分布, Bernoulli distribution"""
        r, l = s
        prob_arrival = ps.get_time_varying_prob(t)

        if l > 1:
            next_s = (max(0, r - a), l - 1)
            return [(next_s, 1.0)]
        else:
            trans = []
            trans.append(((0, 0), 1.0 - prob_arrival))
            for rs in ps.r_dist:
                for ls in ps.l_dist:
                    prob = prob_arrival * self._get_arrival_prob(rs, ls)
                    if prob > 0:
                        trans.append(((rs, ls), prob))
            return trans

    def _precompute_matrices(self):
        """构建加速计算的大矩阵"""
        print(f"Initialize Solver (Penalty={self.env.penalty_weight}):  Numpy Matrix...")
        for t in range(ps.period):
            for s_idx, s in enumerate(ps.S_SPACE):
                for a in range(ps.MAX_CHARGE + 1):
                    # 1. 记录动作值
                    self.A_mat[t, s_idx, a] = a

                    # 2. 缓存基础收益
                    self.R_base[t, s_idx, a] = self.env.reward_func(s, a, t)

                    # 3. 填入转移概率矩阵
                    trans_list = self._get_transitions(s, a, t)
                    if trans_list:
                        for ns, p in trans_list:
                            if ns == 0:
                                ns = (0, 0)
                            ns_idx = ps.S_TO_IDX[ns]
                            self.P_mat[t, s_idx, a, ns_idx] = p

        print("Precompute Matrix Done...")

    def _solve_rvi(self, nu, h_init=None):
        """相对价值迭代 (完全矩阵化)"""
        tol = 1e-3
        if h_init is None:
            h = np.zeros((ps.period, ps.NUM_STATES))
        else:
            h = h_init.copy()

        # 🚀 这就是你原本极其聪明的写法，完美保留！
        R_nu = self.R_base - nu * self.A_mat

        for _ in range(50):
            h_new = np.zeros((ps.period, ps.NUM_STATES))
            for t in reversed(range(ps.period)):
                t_next = (t + 1) % ps.period

                future_v = np.dot(self.P_mat[t], h[t_next])
                q_vals = R_nu[t] + future_v
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
        s_idx = ps.S_TO_IDX[s]

        for _ in range(20):
            mid = (low + high) / 2
            h = self._solve_rvi(mid)

            t_next = (t + 1) % ps.period

            future_v_k = np.dot(self.P_mat[t, s_idx, k], h[t_next])
            q_k = self.R_base[t, s_idx, k] - mid * k + future_v_k

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
        print(f"no {self.filename}，preparing calculation index form...")
        start_time = time.time()
        table = np.zeros((ps.NUM_STATES, ps.period, ps.MAX_CHARGE))

        tasks = []
        for s_idx, s in enumerate(ps.S_SPACE):
            r, l = s
            if r == 0: continue
            for t in range(ps.period):
                for k in range(ps.MAX_CHARGE):
                    tasks.append((s_idx, s, t, k))

        total_tasks = len(tasks)
        num_cores = os.cpu_count()
        print(f"Total tasks: {total_tasks}。 {num_cores} CPU boosting...")

        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # chunksize=50 可以让任务分发效率更高
            results = executor.map(self._compute_single_task, tasks, chunksize=50)

            for s_idx, t, k, val in results:
                table[s_idx, t, k] = val
                completed += 1
                if completed % 500 == 0:
                    print(f"Process: {completed} / {total_tasks} ({(completed / total_tasks) * 100:.1f}%)")

        print(f"Done！Total cost: {time.time() - start_time:.2f} s")
        return table

    def get_index_table(self, force_recompute=False):
        """对外接口：智能获取 Index 表"""
        if not force_recompute and self.file_path.exists():
            print(f"✅  {self.file_path} Loading！")
            self.index_table = np.load(self.file_path)
        else:
            self.index_table = self._calculate_all()
            np.save(self.file_path, self.index_table)
            print(f"💾 Result saved in {self.file_path}")
        return self.index_table

    def display_index(self, target_l, target_t):
        """全景视角查看 Index 矩阵"""
        if self.index_table is None:
            self.get_index_table()

        print(f"\n" + "=" * 65)
        print(f"📊 Whittle Index Matrix (剩余时间 L={target_l}, 时刻 t={target_t})")
        print("=" * 65)

        header = f"{'需求(R)':<10}"
        for k in range(ps.MAX_CHARGE):
            header += f"| {'动作 ' + str(k) + ' vs ' + str(k + 1):<15}"
        print(header)
        print("-" * 65)

        results = []
        for s_idx, s in enumerate(ps.S_SPACE):
            r, l = s
            if l == target_l:
                row_str = f"{r:<10}"
                for k in range(ps.MAX_CHARGE):
                    val = self.index_table[s_idx, target_t, k]
                    row_str += f"| {val:<15.4f}"
                results.append((r, row_str))

        results = sorted(results, key=lambda x: x[0])
        for r, row_str in results:
            print(row_str)

        print("=" * 65 + "\n")


class WhittleSolverXu(WhittleSolver):
    """
    专为 Xu Policy (Binary Action: 0 or MAX_CHARGE) 设计的 Index 求解器。
    它继承自多动作版的 WhittleSolver，但强制约束计算仅在两个极端动作之间进行。
    """

    def __init__(self, env):
        super().__init__(env)


        # 为 Xu index 创建独立的缓存文件
        # case1:time varying
        self.filename_Xu = f"index_Xu_cache_period={ps.period}_pw={self.env.penalty_weight}_1605_Bernoulli.npy"

        # case1:test2
        #self.filename = f"index_Xu_cache_period={ps.period}_pw={self.env.penalty_weight}_test2_varying_Bernoulli.npy"
        # case2:constant, price=1.0,prob=0.5
        #self.filename_Xu = f"index_Xu_cache_period={ps.period}_pw={self.env.penalty_weight}_const1_Bernoulli.npy"
        self.file_path_Xu = self.save_dir / self.filename_Xu

        self.index_table_Xu = None

    def _solve_rvi_Xu(self, nu, h_init=None):
        """限制在 [0, MAX_CHARGE] 两个动作的相对价值迭代"""
        tol = 1e-3
        if h_init is None:
            h = np.zeros((ps.period, ps.NUM_STATES))
        else:
            h = h_init.copy()

        # 仅截取动作 0 和 MAX_CHARGE (ps.MAX_CHARGE 的 index 刚好是 ps.MAX_CHARGE)
        action_indices = [0, ps.MAX_CHARGE]
        R_nu = self.R_base[:, :, action_indices] - nu * self.A_mat[:, :, action_indices]
        P_mat_Xu = self.P_mat[:, :, action_indices, :]

        for _ in range(50):
            h_new = np.zeros((ps.period, ps.NUM_STATES))
            for t in reversed(range(ps.period)):
                t_next = (t + 1) % ps.period
                # 计算期望的未来价值，形状匹配 (NUM_STATES, 2)
                future_v = np.dot(P_mat_Xu[t], h[t_next])
                q_vals = R_nu[t] + future_v
                h_new[t] = np.max(q_vals, axis=1)

            rho = h_new[0, 0]
            h_new -= rho
            if np.max(np.abs(h_new - h)) < tol:
                h = h_new.copy()
                break
            h = h_new.copy()
        return h

    def _find_index_Xu(self, s, t):
        """二分查找 Xu Index: 平衡 a=0 与 a=MAX_CHARGE"""
        low, high = 0, 20.0  # 扩大搜索范围以防出现极端的惩罚
        s_idx = ps.S_TO_IDX[s]

        for _ in range(25):  # 提升二分查找精度
            mid = (low + high) / 2
            h = self._solve_rvi_Xu(mid)
            t_next = (t + 1) % ps.period

            # 动作 a = MAX_CHARGE 的 Q 值
            future_v_max = np.dot(self.P_mat[t, s_idx, ps.MAX_CHARGE], h[t_next])
            q_max = self.R_base[t, s_idx, ps.MAX_CHARGE] - mid * ps.MAX_CHARGE + future_v_max

            # 动作 a = 0 的 Q 值
            future_v_0 = np.dot(self.P_mat[t, s_idx, 0], h[t_next])
            q_0 = self.R_base[t, s_idx, 0] - mid * 0 + future_v_0

            # 寻找临界点
            if q_max > q_0:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def get_index_table_Xu(self):
        """生成并缓存 Xu Index Table。它的形状为 (NUM_STATES, period)，不再有动作维度 k。"""
        if self.file_path_Xu.exists():
            print(f"Loading cached Xu Index table from {self.file_path_Xu}...")
            self.index_table_Xu = np.load(self.file_path_Xu)
            return self.index_table_Xu

        print("Computing Xu Index Matrix (Binary Actions)...")
        # 由于动作仅有0和MAX_CHARGE，所以每个状态只需对应 1 个指数
        self.index_table_Xu = np.zeros((ps.NUM_STATES, ps.period))

        start_time = time.time()
        for t in range(ps.period):
            for s in ps.S_SPACE:
                s_idx = ps.S_TO_IDX[s]
                r, l = s
                # 只有 L>0 且 R>0 时才需要计算有效 Index
                if r == 0 or l == 0:
                    self.index_table_Xu[s_idx, t] = 0.0
                else:
                    self.index_table_Xu[s_idx, t] = self._find_index_Xu(s, t)

        np.save(self.file_path_Xu, self.index_table_Xu)
        print(f"💾 Result saved in {self.file_path_Xu}")
        return self.index_table_Xu

    def display_index_Xu(self, target_l, target_t):
        """全景视角查看 Xu Index 矩阵"""
        if self.index_table_Xu is None:
            self.get_index_table_Xu()

        print(f"\n" + "=" * 65)
        print(f"📊 Xu Index Matrix (剩余时间 L={target_l}, 时刻 t={target_t})")
        print("=" * 65)

        header = f"{'需求(R)':<10}"
        for k in range(ps.MAX_CHARGE):
            header += f"| {'动作 ' + str(k) + ' vs ' + str(k + 1):<15}"
        print(header)
        print("-" * 65)

        results = []
        for s_idx, s in enumerate(ps.S_SPACE):
            r, l = s
            if l == target_l:
                row_str = f"{r:<10}"
                for k in range(ps.MAX_CHARGE):
                    val = self.index_table_Xu[s_idx, target_t]
                    row_str += f"| {val:<15.4f}"
                results.append((r, row_str))

        results = sorted(results, key=lambda x: x[0])
        for r, row_str in results:
            print(row_str)

        print("=" * 65 + "\n")



# ==========================================================
# 测试与运行入口
# ==========================================================
if __name__ == "__main__":
    from charging_env import ChargingEnv

    # 1. 创建你的统一规则对象，传入测试参数
    for penalty_weight in [0.2,0.4,0.6,0.8]:
        test_env = ChargingEnv(N=10, power_ratio=0.6, penalty_weight=penalty_weight)

        # 2. 将环境丢给求解器
        solver = WhittleSolverXu(test_env)
        solver1 = WhittleSolver(test_env)
        # 3. 极速求解或加载
        table_Xu = solver.get_index_table_Xu()
        table = solver.get_index_table()
        # 4. 打印检查 (T=1, 剩余时间 L=2)
        solver.display_index_Xu(target_l=2, target_t=1)
        solver.display_index(target_l=2, target_t=1)

