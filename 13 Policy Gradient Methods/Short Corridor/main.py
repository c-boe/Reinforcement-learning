"""
Solve short corridor with switched actions (Example 13.1) of Sutton and Barto's
"Reinforcement learning" with REINFORCE with baseline
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from shortcorridor import ShortCorridor
from shortcorridor import FeatureVectors
from poligradmeth import REINFORCE_with_baseline

#%%
def plot_reward(reward):
    """
    Plot rewards
    """
    plt.figure()
    plt.plot(reward)
    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    

#%% Run REINFORCE with baseline

NUM_RUNS = 100
NUM_EPISODES = 1000
ALPHA_TH = 2**(-9) # stepsize policy gradient
GAMMA = 0.9
ALPHA_W = 2**(-6) # stepsize VF gradient

FV = FeatureVectors() # FV.x_vec : feature vector policy parameterization
SC = ShortCorridor() # environment

sum_rewards = np.zeros(NUM_EPISODES)
for run in tqdm(range(NUM_RUNS)):
    rewards, pi =  REINFORCE_with_baseline(SC, FV.x_vec, ALPHA_TH, 
                                           NUM_EPISODES, GAMMA, ALPHA_W)
    sum_rewards +=  rewards
avg_rewards =  sum_rewards/NUM_RUNS

plot_reward(avg_rewards)    
