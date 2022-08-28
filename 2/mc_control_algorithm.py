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

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for e in range(num_episodes):
        if e % 1000 == 0:
            print("\rEpisode {}/{}.".format(e, num_episodes), end="")
            sys.stdout.flush()
        
        d = False
        s = env._reset()
        s_list = []
        r_list = []
        a_list = []
        
        while not d:
            a = np.random.choice(range(env.action_space.n), p=policy(s))
            next_s, r, d, i = env._step(a)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)

            s = next_s

        total_r = 0

        for s, r, a in zip(s_list[::-1], r_list[::-1], a_list[::-1]):
            total_r = total_r * discount_factor + r
            returns_sum[(s, a)] += total_r
            returns_count[(s, a)] += 1

        for (s, a) in returns_sum:
            Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

    return Q, policy


env = Env.BlackjackEnv(gym.Env)
Q, policy = mc_control_epsilon_greedy(env, num_episodes=50000, epsilon=0.1)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")