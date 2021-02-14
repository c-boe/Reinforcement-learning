"""
Solving multi armed bandit from chapter 2 of Sutton and Barto's 
"Reinforcement Learning" with simple bandit and gradient bandit algorithm
"""
import matplotlib.pyplot as plt
from tqdm import tqdm

from banditalg import  gradient_bandit_alg, simple_bandit_alg
from mabandit import MultiArmedBandit

#%%        
def plot_oa_pct(max_steps, num_runs, bandit_type, epsilon, alpha, 
                     avg_pct_max_q_a):
    '''
    plot percentage of optimal actions as function of number of steps
    '''
    plt.figure()
    plt.plot(range(max_steps), avg_pct_max_q_a/(num_runs),
             label = bandit_type + ", eps=" + str(epsilon) 
             + ", alpha=" + str(alpha))
    plt.legend()
    plt.ylabel("%Optimal action")
    plt.xlabel("Steps")

    return

#%%
"""Set parameters and run algorithms (for detailed explanations see 
chapter 2 of Sutton and Barto's "Reinforcement Learning"  """

max_steps = 1000 # max time steps
k_arms = 10  # number of arms of bandit
epsilon = 0.1 # epsilon value for epsilon greedy policy
alpha = 0.1 # step size parameter; alpha = 0: incremental steps
bandit_type = "stationary" # stationary or nonstationary bandit
num_runs = 100 # nr of runs of bandit algorithms
Q1 = 0 # initial Q values
UCB = False # upper confidence boundary
c = 2 # constant for UCB only

MAB = MultiArmedBandit(k_arms, bandit_type)

#%% simple bandit
avg_pct_max_q_a = 0
for run in tqdm(range(num_runs)):
    Q, q, pct_max_q_a = simple_bandit_alg( 
        MAB, epsilon, max_steps, alpha, Q1, UCB, c, run)
    avg_pct_max_q_a += pct_max_q_a
  
plot_oa_pct(max_steps, num_runs, bandit_type, epsilon, alpha, 
                 avg_pct_max_q_a)

#%% gradient bandit
baseline = True

avg_pct_max_q_a = 0
for run in tqdm(range(num_runs)):
    # remark: the evaluation of the runs can be parallelized for better
    #         performance
    q, pct_max_q_a = gradient_bandit_alg( 
    	MAB, max_steps, alpha, baseline, Q1, run)
    avg_pct_max_q_a += pct_max_q_a


plot_oa_pct(max_steps, num_runs, bandit_type, epsilon, alpha, 
                 avg_pct_max_q_a)