import matplotlib
matplotlib.use('TkAgg')  # avoid non-GUI warning for matplotlib
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import gridworld as Env

env = Env.GridworldEnv(live_display=True)
episode = 10
per_episode = 10
for i in range(episode):
    env.reset()
    obs = env._obs()
    d = False
    for j in range(per_episode):
        action = np.random.choice([0, 1, 2, 3])
        s, r, d, _ = env.step(action)
        env.render()
        if d:
            break
