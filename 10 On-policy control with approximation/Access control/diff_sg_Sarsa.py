
"""
Implementation of example 10.2 (An Access-Control Queuing Task) from Sutton 
and Barto's "Reinforcement learning"
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from accesscontrol import AccessControl

#%%
def init_rew_est():
    """
    Initialize estimated reward

    Returns
    -------
    R_avg : float
        Estimate for average reward.

    """
    R_avg = 0
    return R_avg

def init_weights(dim):
    """
    Initialize weights for linear function approximation

    Parameters
    ----------
    dim : int
        dimension of weight vector

    Returns
    -------
    weights : ndarray
        weights of linear function approximation

    """
    weights = np.zeros(dim)
    return weights

def eps_greedy(q_value, epsilon):
    """
    Epsilon greedy action selection for access-control queuing task

    Parameters
    ----------
    q_value : ndarray, shape (num actions,)
        q values for current state
    epsilon : float
        epsilon greedy action selection.

    Returns
    -------
    action : int
        action of agent
    q_current : float
        q value for current action.

    """
    
    if np.random.random() > (epsilon - epsilon/len(q_value)):
        if q_value[0] == q_value[1]:
            action = np.random.choice([0,1])
        else:   
            action = np.argmax(q_value)
    else: 
            action = np.random.choice([0,1])
            
    q_action = q_value[action]
        
    return action, q_action
        

def q_values(state, actions, weights, xf_vec):
    """
    Calculate q values for current state and actions

    Parameters
    ----------
    state : tuple
        current state-
    actions : ndarray
        possible actions-
    weights : ndarray
        weights of linear function approximation-
    xf_vec : list
        feature vectors for every state and action-

    Returns
    -------
    q : ndarray, shape (nr actions,)
        q values for current state.

    """
    q = np.zeros(len(actions))
    for action in actions:
        q[action] = np.dot(weights, xf_vec[state][action])    
    
    return q

def feat_vec(states, actions):
    """
    Generate feature vector for linear function approximation for discrete
    state space

    Parameters
    ----------
    states : list
        possible states
    actions : ndarray
        possible actions.

    Returns
    -------
    xf_vec : dict
        feature vector for linear function approximation.

    """
    dim = len(states)*len(actions)
    vec = np.eye(dim, dtype=int)
    xf_vec = {state:[np.zeros(dim) for action in range(len(actions))] 
              for state in states}
    
    for count_action, action in enumerate(range(len(actions))):
        for count_state, state in enumerate(states):
            xf_vec[state][action] = vec[len(states)*count_action + count_state]
    
    return xf_vec

def det_policy(states, actions, weights, xf_vec):
    """
    Determine policy from calculated weight vector

    Parameters
    ----------
    states : list
        possible states
    actions : ndarray
        possible actions.
    weights : ndarray
        weights of linear function approximation-
    xf_vec : dict
        feature vector for linear function approximation.

    Returns
    -------
    policy : ndarray
       policy of agent

    """
    policy = {}
    for state in states:
        Q =  q_values(state, actions, weights, xf_vec)
        policy[state] = np.argmax(Q)
     
    return policy
    
def det_diff_vf(states, actions, w, xf_vec):
    """
    Determine differential value of best action from weights

    Parameters
    ----------
    states : list
        possible states
    actions : ndarray
        possible actions.
    weights : ndarray
        weights of linear function approximation-
    xf_vec : dict
        feature vector for linear function approximation.


    Returns
    -------
    diff_vf : dict
        differential value of best action

    """
    
    diff_vf = {}
    for state in states:
        q = q_values(state, env.actions, w, xf_vec)
        action, q_action = eps_greedy(q, 0)

        diff_vf[state] = q_action
        
    return diff_vf


#%% 

def plot_policy(states, policy):
    """plot policy as function of number of free servers and priorities"""
    
    priorities, num_free_servers = zip(*states)
    actions = np.array(list(policy.values()))
    priorities = np.array(priorities)
    num_free_servers = np.array(num_free_servers)
    
    num_free_servers = np.reshape(num_free_servers, [11,4])
    priorities = np.reshape(priorities, [11,4])
    actions = np.reshape(actions, [11,4])
    
    num_free_servers = num_free_servers[1:]
    priorities = priorities[1:]
    actions = actions[1:]
    
    plt.figure()
    plt.imshow(actions)
    
    ax = plt.gca()
    ax.set_yticks(np.arange(0, 10, 1));
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_yticklabels(np.arange(1, 11, 1));
    ax.set_xticklabels(np.array([1,2,4,8]))
    ax.set_ylabel("Number of free servers")
    ax.set_xlabel("Priorities")
    ax.set_title("Policy; yellow: reject, blue: accept")
    
    return

def plot_diff_vf(diff_vf, priorities, num_servers):
    """plot differential valueof best action as a function of number of free 
    servers"""
    
    plt.figure()
    for priority in priorities:
        vf = []
        for servers in range(num_servers + 1):
            vf.append(diff_vf[(priority, servers)])
        plt.plot(vf, label = "priority " + str(priority))
    plt.xlabel("Number of free servers")
    plt.ylabel("Differential value of best action")
    plt.legend(loc="upper right")
    
    return
#%%

def diff_sg_sarsa(env, epsilon, alpha, beta, num_steps):
    """Differential semi-gradient Sarsa for estimating q value function"""

    xf_vec = feat_vec(env.states, env.actions)

    w = init_weights(len(env.states)*len(env.actions))
    R_avg = init_rew_est()
    
    state = env.reset()
    q = q_values(state, env.actions, w, xf_vec)
    action, q_current = eps_greedy(q, epsilon)


    for step in tqdm(range(num_steps)):
        next_state, reward = env.step(state, action)
         
        q_next = q_values(next_state, env.actions, w, xf_vec)
        next_action, q_next = eps_greedy(q_next, epsilon)
      
        delta = reward - R_avg + q_next - q_current
        R_avg = R_avg + beta*delta

        w = w + alpha*delta*xf_vec[state][action]

        state = next_state
        action = next_action
        q_current = q_next    
        
    policy = det_policy(env.states, env.actions, w, xf_vec)
    diff_vf = det_diff_vf(env.states, env.actions, w, xf_vec)
    
    print("R average: " + str(R_avg))
    return policy, diff_vf


#%%
if __name__ == "__main__":
    """Set parameters and run differential semi-gradient Sarsa"""
    
    env = AccessControl()
    
    num_steps = 2000000
    epsilon = 0.1
    alpha = 0.01
    beta = 0.01
    
    policy, diff_vf = diff_sg_sarsa(env, epsilon, alpha, beta, num_steps)
    
    plot_policy(env.states, policy)
    plot_diff_vf(diff_vf, env.priorities, env.num_servers)