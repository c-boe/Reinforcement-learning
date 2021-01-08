"""
Implementation of Exercises/Examples from Chapter 6 of Sutton and Barto's 
"Reinforcement Learning"
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gridworld import CliffWalking


#%%

def gen_random_q_table(env):
    '''
    Generate q table with random initial values

    Parameters
    ----------
    states : list
        list of tuples representing the coordinates of the agent.
    nr_actions : int
        number of choices for action.

    Returns
    -------
    q_table : dict
        dict with states as keys and q-values each action as values.

    '''
    q_table = {}
    for state in env.states:
        q_table[state] = np.random.uniform(-1, 0, size = len(env.action_space()))
        
    return q_table

#%%


def plot(num_episodes, Returns):

    plt.figure()
    plt.plot(range(num_episodes),Returns, label = "Q learning")
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    

#%% Q Learning
def Q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay):
    """
    Q- learning algorithm for solving cliffwalking example 6.6 from chapter 6 
    of "Reinforcement learning" by Sutton and Barto
    """
    q_table = gen_random_q_table(env)
    
    Returns = []
    for episode in tqdm(range(num_episodes)):
        done = False 
        Return = 0
        
        state = env.reset()
        
        while not done:
            if  epsilon < np.random.uniform(0,1,1):
                action = np.argmax(q_table[state])
            else:
                action = np.random.choice(env.action_space())
                
            new_state, reward, done, final_state = env.step(state, action)
            
            q_current = np.max(q_table[state])
            q_next = np.max(q_table[new_state])
            
            if final_state:
               q_new = 0 
            else: 
               q_new = q_current + alpha*(reward + gamma*q_next - q_current)
            
            q_table[state][action] = q_new
            
            state = new_state
            Return += reward
    
            if episode == (num_episodes-1):
                env.render(new_state)
    
        epsilon *=epsilon_decay       
        Returns.append(Return)   
        
    return Returns
#%%  


if __name__ == "__main__":
    
    num_episodes = 5000
    alpha = 0.1
    gamma = 1
    
    epsilon =  0.1
    epsilon_decay = 0.95
    
    x_dim = 12
    y_dim = 4
    kings_moves = False
    
    env = CliffWalking(x_dim, y_dim, kings_moves)

    Returns = Q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay)

    plot(num_episodes, Returns)