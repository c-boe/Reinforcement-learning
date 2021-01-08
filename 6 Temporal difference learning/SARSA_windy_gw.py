"""
Implementation of Exercises/Examples from Chapter 6 of Sutton and Barto's 
"Reinforcement Learning"
"""

from gridworld import WindyGridworld
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
def gen_zero_q_table(env):
    '''
    Generate q table with zeros as initial values
    Returns
    -------
    q_table : dict
        dict with states as keys and q-values each action as values.

    '''
    q_table = {}
    for state in env.states:
            q_table[state] = np.zeros((len(env.action_space())))
            
    return q_table

#%%
def plot(num_episodes, Returns):
    plt.figure()
    plt.plot(range(num_episodes), Returns)    
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode")


#%% SARSA
def SARSA(env, num_episodes, epsilon, epsilon_decay, alpha, gamma):
    """
    SARSA algorithm for solving windy gridworld example 6.5 from chapter 6 of 
    "Reinforcement learning" by Sutton and Barto
    """
    
    q_table = gen_zero_q_table(env) 
    Returns = []
    for episode in tqdm(range(1, num_episodes+1)):
        
        done = False
        Return = 0
        state = env.reset()
        
        if epsilon < np.random.uniform(0,1,1):
            action = np.argmax(q_table[state])
        else:
            action = np.random.choice(env.action_space())
    
        while not done:
            next_state, reward, done, final_state = env.step(state, action)
            
            if epsilon < np.random.uniform(0,1,1):
                next_action = np.argmax(q_table[state])
            else:
                next_action = np.random.randint(0, len(env.action_space()))
            
            q_current = q_table[state][action]
            q_next = q_table[next_state][next_action]
            
            if final_state:
               q_new = 0 
            else:
               q_new = q_current + alpha*(reward + gamma*q_next -  q_current)
            
            q_table[state][action] = q_new
            
            state = next_state
            action = next_action
            Return += reward
            
            # plot grid_world
            if episode == (num_episodes-1):
                env.render(state)
                
        epsilon *=epsilon_decay    
        Returns.append(Return)    
    
    return Returns

#%%

if __name__ == "__main__":
    
    num_episodes = 5000
    
    epsilon = 0.1
    epsilon_decay = 0.999
    alpha = 0.5
    gamma = 1
    
    kings_moves = True
    x_dim = 10
    y_dim  = 7
    initial_state = (0, 3)
    terminal_state = (6, 3)
    
    stochastic_wind = True
    
    env =  WindyGridworld(x_dim, y_dim, kings_moves, stochastic_wind,
                          initial_state, terminal_state)
    

    Returns = SARSA(env, num_episodes, epsilon, epsilon_decay, alpha, gamma)
    
    plot(num_episodes, Returns)