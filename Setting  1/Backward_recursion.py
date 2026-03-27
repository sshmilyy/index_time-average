import numpy as np
from collections import defaultdict
import pickle
import os
import itertools
from Setting1 import (MAX_CHARGE, T, l_dist, max_l, total_power, N, r_dist)
from Setting1 import (transition_probability_exp, reward_function)


def backward_recursion1(T):
    V = defaultdict(lambda: defaultdict(float))
    policy = defaultdict(dict)
    cache_file= 'backward_recursion_file.pkl'
    if os.path.exists(cache_file):
        print("加载已缓存的Backward Recursion结果...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 生成所有可能的状态组合
    single_charger_states = []
    for _ in range(N):
        charger_states = []
        for r in r_dist:
            for l in l_dist:
                if not (l == 0 and r != 0):  # 有效性检查
                    charger_states.append((r, l))
        single_charger_states.append(charger_states)

    # 初始化所有状态
    all_states = itertools.product(*single_charger_states)
    for t in range(T, -1, -1):
        for state_combo in all_states:
            flat_state = tuple(itertools.chain(*state_combo))
            V[t][flat_state] = 0 if t == T else -float('inf')

    # 反向递归核心逻辑
    for t in range(T - 1, -1, -1):
        all_states = itertools.product(*single_charger_states)
        for state_combo in all_states:
            current_state = tuple(itertools.chain(*state_combo))

            max_value = -float('inf')
            best_action = tuple([0] * N)

            # 生成有效动作空间
            action_space = []
            for i in range(N):
                r, l = state_combo[i]
                if l == 0:
                    action_space.append([0])
                else:
                    action_space.append(list(range(0, min(r, MAX_CHARGE) + 1)))

            # 生成合法动作组合
            for action in itertools.product(*action_space):
                if sum(action) > total_power:
                    continue

                # 计算即时奖励
                reward = reward_function(current_state, action)

                # 计算状态转移
                expected_future = 0
                transitions = transition_probability_exp(current_state, action)
                for next_state, prob in transitions:
                    expected_future += prob * V[t + 1][next_state]

                total_value = reward + expected_future
                if total_value > max_value:
                    max_value = total_value
                    best_action = action

            V[t][current_state] = max_value
            policy[t][current_state] = best_action
    print('完成Backward recursion计算')
    with open(cache_file, 'wb') as f:
        pickle.dump((dict(V), dict(policy)), f)
    print('完成Backward recursion计算并缓存结果')
    return V, policy


def backward_recursion11(T):
    # 预加载缓存（新增内存缓存）
    memo = {}
    cache_file = 'backward_recursion_file.pkl'
    if os.path.exists(cache_file):
        print("加载已缓存的Backward Recursion结果...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 预生成所有有效状态（优化点1）
    single_charger_states = []
    for _ in range(N):
        states = [(r, l) for r in r_dist for l in l_dist if not (l == 0 and r != 0)]
        single_charger_states.append(states)
        print(states)
    # 一次性生成所有状态组合（优化点2）
    all_state_combos = list(itertools.product(*single_charger_states))
    precomputed_states = [tuple(itertools.chain(*sc)) for sc in all_state_combos]

    # 使用普通字典替代defaultdict（优化点3）
    V = {t: {s: 0.0 if t == T else -float('inf') for s in precomputed_states}
         for t in range(T + 1)}
    policy = {t: {} for t in range(T + 1)}

    # 预生成合法动作空间（优化点4）
    action_space_cache = {}
    for sc in all_state_combos:
        state_key = tuple(itertools.chain(*sc))
        action_space = []
        for i in range(N):
            r, l = sc[i]
            if l == 0:
                action_space.append([0])
            else:
                action_space.append(list(range(0, min(r, MAX_CHARGE) + 1)))
        action_space_cache[state_key] = list(itertools.product(*action_space))

    # 反向递归优化（新增进度显示）
    for t in range(T - 1, -1, -1):
        print(f"处理时间步 t={t} (进度:{(T - t) / T * 100:.1f}%)")

        for state in precomputed_states:
            max_value = -float('inf')
            best_action = tuple([0] * N)

            # 过滤合法动作（优化点5）
            valid_actions = [a for a in action_space_cache[state]
                             if sum(a) <= total_power]

            for action in valid_actions:
                # 计算缓存键
                cache_key = (state, action)
                if cache_key not in memo:
                    # 计算奖励和转移概率
                    reward = reward_function(state, action)
                    transitions = transition_probability_exp(state, action)
                    # 缓存计算结果
                    memo[cache_key] = (reward, transitions)
                else:
                    reward, transitions = memo[cache_key]

                # 计算未来期望值
                future = sum(prob * V[t + 1][next_state]
                             for next_state, prob in transitions)

                if (reward + future) > max_value:
                    max_value = reward + future
                    best_action = action

            V[t][state] = max_value
            policy[t][state] = best_action

    # 保存优化结果
    with open(cache_file, 'wb') as f:
        pickle.dump((V, policy), f)
    print('完成优化版Backward recursion计算')
    return V, policy
backward_recursion11()



def simulate_br(T, initial_state, policy):
    total_reward = 0
    current_state = initial_state
    states = [initial_state]
    actions = []

    for t in range(T):
        # 从策略表中获取最优动作
        action = policy[t][current_state]
        actions.append(action)

        reward = reward_function(current_state, action)
        total_reward += reward

        transitions = transition_probability_exp(current_state, action)
        next_states, probs = zip(*transitions)
        selected_idx = np.random.choice(len(next_states), p=probs)#随机选择下一状态
        next_state = next_states[selected_idx]
        states.append(next_state)  # 记录新状态
        current_state = next_state

    return total_reward, states, actions


def test():
    # Test code
    V, policy = backward_recursion1(T)


if __name__ == '__main__':
    test()

