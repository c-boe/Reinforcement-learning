"""
Episodic Semi-gradient SARSA for MontainCar enviroment of chapter 10.1 in
Sutton and Barto's "Reinforcement Learning"'
"""
import numpy as np
from tqdm import tqdm

from tilecoding import feature_vecs
from mountaincar import MountainCar

#%%
def init_weights(num_tilings, tiles_per_dim, num_dims, num_actions):
    """
    Initiliaze weights for linear function approximation of value function

    Parameters
    ----------
    num_tilings : int
        number of tilings for tile coding
    tiles_per_dim : int
        numer of tiles in each dimension
    num_dims : int
        number of variables to which tilecoding is applied
    num_actions : int
        number of possible actions

    Returns
    -------
    weights : ndarray
        weights for linear function approximation of value function

    """
    weights =  np.zeros((num_tilings*tiles_per_dim**num_dims*num_actions))
    return weights


def init_Q_values(weights, fvec_idx_per_tiling, idcs_per_action,  nr_actions):
    """
    

    Parameters
    ----------
    weights : ndarray
        weights for linear function approximation of value function
    fvec_idx_per_tiling : ndarray
        non-zero indices of feature vector
    idcs_per_action : int
        total number of feature vector indices per action
    num_actions : int
        number of possible actions

    Returns
    -------
    Q : ndarray
        action value function values for given state.

    """
    Q = np.zeros(3) 
    for action in range(nr_actions):
        idcs = fvec_idx_per_tiling + action*idcs_per_action
        Q[action] = np.sum(weights[idcs])
    return Q

def eps_greedy(Q, epsilon, num_actions):
    """
    Epsilon greey action selection

    Parameters
    ----------
    Q : ndarray
        action value function values for given state.
    epsilon : float
        probability for choosing non-greedy action
    num_actions : int
        number of possible actions

    Returns
    -------
    action : int 
        current action.
    Q_value : float
        q value for current state and action
    """
    if np.random.uniform(0,1,1) > epsilon:
        action = np.argmax(Q)
    else:
        action = np.random.randint(low=0, high=num_actions)
    
    Q_value = Q[action]
    return action,  Q_value


def feature_vec(idcs, idcs_per_action, num_actions):
    """
    feature vector for current action and state

    Parameters
    ----------
    idcs : ndarray
        non-zero indices of feature vector
    idcs_per_action : int
        total number of feature vector indices per action
    num_actions : int
        number of possible actions

    Returns
    -------
    x_fvec : ndarray
        feature vector.

    """
    x_fvec = np.zeros((idcs_per_action*num_actions))
    x_fvec[idcs] = 1
    
    return x_fvec

#%%

def episodic_sg_SARSA(env, fvecs, weights, alpha, epsilon, gamma, num_actions,
                      num_episodes):
    """Episodic semi gradient SARSA (chapter 10.1) for mountain car problem"""

    for episode in tqdm(range(num_episodes)):
        done = False
        
        state = env.reset()
        fvec_idx_per_tiling = fvecs.calc_feature_vec(state)
        Q_vals = init_Q_values(weights, fvec_idx_per_tiling, idcs_per_action,
                               num_actions)
        action, Q_current = eps_greedy(Q_vals, epsilon, num_actions) 
        
        step_count = 0
        while not done:
            step_count += 1
            
            if episode == (num_episodes -1):
                env.render(state[0])
                
            idcs = fvec_idx_per_tiling + action*idcs_per_action
            x_fvec = feature_vec(idcs, idcs_per_action, num_actions)
            
            next_state, reward, done,__ = env.step(state, action)
            if done:
                weights[idcs] += alpha*(reward - Q_current)*x_fvec[idcs]
                break
            
            fvec_idx_per_tiling = fvecs.calc_feature_vec(next_state)
            Q_vals = init_Q_values(weights, fvec_idx_per_tiling, idcs_per_action,
                                   num_actions)
            next_action, Q_next = eps_greedy(Q_vals, epsilon, num_actions)
            weights[idcs] += alpha*(reward + gamma*Q_next - Q_current)*x_fvec[idcs]
            
            state = next_state
            action = next_action
            Q_current = Q_next
            
        env.plot_step_per_ep(episode, step_count)


#%% Run Episodic SARSA
if __name__ == "__main__":

    #
    alpha = 0.1/8
    epsilon = 0.05
    gamma = 1
    num_episodes = 500
    
    dim1 = np.array((-1.2, 0.5))
    dim2 = np.array((-0.07, 0.07))
    
    dims = np.array((dim1, dim2))
    
    num_actions = 3
    num_tilings = 8
    tiles_per_dim = 8
    
    idcs_per_action = num_tilings*tiles_per_dim**len(dims)
    displ_vecs = np.array([(1, 3), (3, 1), (-1,3), (3, -1),
                   (1,-3), (-3,1), (-1,-3), (-3,-1)], dtype="float")
    
    #%%
    fvecs = feature_vecs(dims, num_tilings, tiles_per_dim, displ_vecs)

    weights = init_weights(num_tilings, tiles_per_dim, len(dims), num_actions)
    env = MountainCar()
    
    #%%
    episodic_sg_SARSA(env, fvecs, weights, alpha, epsilon, gamma, num_actions,
                      num_episodes)