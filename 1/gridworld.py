import numpy as np
import sys
import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
from collections import defaultdict

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top right corner.
    You receive a reward of 0 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    '''更改了end_reward'''
    def __init__(self, shape=[4, 4], end_reward=1, live_display=False):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        self.shape = shape
        self.n = np.prod(self.shape)
        self.MAX_X = shape[0]
        self.MAX_Y = shape[1]
        self.nA = 4
        self.start_state = random.randint(0, self.n - 1)
        self.state = self.start_state
        self.reward_in = end_reward
        self.end_reward = end_reward
        self.live_display = live_display
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [0, 1, 2, 3, 4, 5]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        self.ax_imgs = []


        '''定义每一步的sample，定义默认的value table，将每一个点的值设为-100000'''
        self.samples = []
        self.value_table = {"0": -100000, "1": -100000, "2": -100000, "3": -100000, "4": -100000, "5": -100000,
                            "6": -100000, "7": -100000, "8": -100000, "9": -100000, "10": -100000,
                            "11": -100000, "12": -100000, "13": -100000, "14": -100000, "15": -100000}



    def step(self, action):
        '''更改终止状态'''
        if self.state == self.n - 1:
            print('game is end! please reset')
            time.sleep(2)
            return self.n-1, 0, True, {}

        '''更改每一步的reward为-1'''
        reward = -1
        done = False
        '''
        state重新标号
        
        '''

        x = int(self.state / self.shape[0])
        y = int(self.state % self.shape[0])

        if action == 0:
            self.state = self.state if y == 0 else self.state - 1

        elif action == 1:
            self.state = self.state if x == (self.MAX_X - 1) else self.state + self.MAX_X

        elif action == 2:
            self.state = self.state if y == (self.MAX_Y - 1) else self.state + 1

        else:
            self.state = self.state if x == 0 else self.state - self.MAX_X

        '''更改终止状态'''
        if self.state == self.n - 1:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        self.start_state = random.randint(0, self.n - 1)
        self.state = self.start_state
        self.ax_imgs = []
        return self.state

    def _obs(self):
        obs = np.zeros(self.n).reshape(self.shape) + 1
        obs[self.shape[0] - 1][self.shape[0] - 1] = 2
        obs[int(self.state % self.shape[0])][int(self.state / self.shape[0])] = 3
        return obs

    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return
        obs = self._obs()
        if not hasattr(self, 'fig'):
            self.fig, self.ax_full = plt.subplots(nrows=1, ncols=1)
        self.ax_full.axis('off')
        self.fig.show()
        if self.live_display:
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            self.ax_full_img.set_data(obs)
        else:
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
        plt.draw()
        if self.live_display:
            self.fig.canvas.draw()
        else:
            self.ax_imgs.append([self.ax_full_img])  # List of axes to update figure frame
            self.fig.set_dpi(100)
        return self.fig