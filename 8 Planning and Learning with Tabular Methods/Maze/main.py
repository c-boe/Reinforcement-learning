"""
Solving Blocking Maze (Example 8.2 of Sutton and Barto's "Reinforcement 
Learning) with Dyna-Q
"""

from dyna import Dyna
from maze import BlockingMaze

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

NUM_STEPS = 3000
CHANGE_STEP = 1000

ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1
PLAN_STEPS = 50

NUM_RUNS = 20

#%% Average cumulative reward in maze with Dyna Q

ave_cum_reward = np.zeros(NUM_STEPS + 1)
for run in tqdm(range(NUM_RUNS)):
    bmaze = BlockingMaze()
    dyna = Dyna(bmaze, PLAN_STEPS, CHANGE_STEP, ALPHA, GAMMA,
                EPSILON)
    cum_reward = dyna.DynaQ(NUM_STEPS)
    ave_cum_reward += cum_reward

ave_cum_reward = ave_cum_reward/NUM_RUNS

plt.figure()
plt.plot(ave_cum_reward)
plt.xlabel("time steps")
plt.ylabel("Cumulative reward")
plt.title("Average performance")

