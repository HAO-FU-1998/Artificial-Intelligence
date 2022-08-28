import matplotlib
matplotlib.use('TkAgg')  # avoid non-GUI warning for matplotlib
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import gridworld as Env
import random
import math


def mc_control_epsilon_greedy(nve, num_episodes, discount_factor=1.0, epsilon=0.1):
    for i in range(num_episodes):
        state = nve.reset()

        if state == 15:
            print('game is end! please reset')

        obs = nve._obs()
        d = False



        '''第一次的action'''
        action = get_action(nve, state)
        while True:
            next_state, reward, done, _ = nve.step(action)

            save_sample(nve, next_state, reward)

            if done:
                update(nve,discount_factor,epsilon)
                nve.samples.clear()
                print('game is end! please reset,start state'+ str(state))
                break

            action = get_action(nve, next_state)

        for i in range(0,4):
            for j in range(0,4):
                strval = str(i*4+j)
                strval = strval+" "
                print(strval, end="")
                print(nve.value_table.get(str(i*4+j)),end="")
                print(" ", end="")
            print()


def get_action(nve , state):
    next_state = possible_next_state(nve , state)
    action = arg_max(next_state)
    return int(action)


def possible_next_state(nve , state):
    row = state % 4
    col = math.floor(state / 4)
    next_state = [0.0] * 4

    '''
    0上，1右，2下，3左
    '''
    if row != 0:
        next_state[0] = nve.value_table[str(state-1)]
    else:
        next_state[0] = nve.value_table[str(state)]

    if col != 3:
        next_state[1] = nve.value_table[str(state+4)]
    else:
        next_state[1] = nve.value_table[str(state)]

    if row != 3:
        next_state[2] = nve.value_table[str(state+1)]
    else:
        next_state[2] = nve.value_table[str(state)]

    if col != 0:
        next_state[3] = nve.value_table[str(state-4)]
    else:
        next_state[3] = nve.value_table[str(state)]

    return next_state


def arg_max(next_state):
     max_index_list = []
     max_value = next_state[0]

     for index, value in enumerate(next_state):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
     return random.choice(max_index_list)



def save_sample(nve,state, reward):
    nve.samples.append([state, reward])


'''
    更新value表
'''
def update(nve,discount_factor,epsilon):
    G_t = 0
    visit_state = []

    for reward in reversed(nve.samples):
        state = str(reward[0])
        if state not in visit_state:
            visit_state.append(state)
            G_t = discount_factor * (reward[1] + G_t)

            if not abs(nve.value_table[str(state)] - G_t) <= epsilon :
                nve.value_table[str(state)] = G_t





def td_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    '''
    Your code is here.
    '''


def main():
    env = Env.GridworldEnv(live_display=True)
    num_episodes = 20
    mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1)

if __name__ == "__main__":
    main()