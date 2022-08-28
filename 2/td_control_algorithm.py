import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
import BlackJack as Env
import plotting

matplotlib.style.use('ggplot')


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        p = [epsilon / nA for i in range(nA)]
        greedy_action = np.argmax(Q[observation])
        p[greedy_action] += 1 - epsilon
    return policy_fn

'''初始化:Q为0，policy为要牌与不要牌概率相等，计数器为0'''
def td_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for e in range(num_episodes):
        if e % 1000 == 0:
            print("\rEpisode {}/{}.".format(e, num_episodes), end="")
            sys.stdout.flush()

        '''初始状态通过重置环境得到，并生成第一步动作，增加计数器值'''
        d = False
        s = env._reset()
        a = np.random.choice(range(env.action_space.n), p=policy(s))
        returns_count[(s, a)] += 1

        '''对于每一步：产生下一个状态，使用greedy算法更新策略产生下一个动作，更新Q的值，s、a得到新的值，计数器更新'''
        while not d:
            next_s, r, d, i = env._step(a)
            next_a = np.random.choice(range(env.action_space.n), p=policy(s))
            Q[s][a] += (1/returns_count[(s, a)]) * (r + discount_factor*Q[next_s][next_a] - Q[s][a])

            s = next_s
            a = next_a
            returns_count[(s, a)] += 1

    return Q, policy


env = Env.BlackjackEnv(gym.Env)
Q, policy = td_control_epsilon_greedy(env, num_episodes=50000, epsilon=0.1)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
    print(state, np.argmax(actions))
plotting.plot_value_function(V, title="Optimal Value Function")