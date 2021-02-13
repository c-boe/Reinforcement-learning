"""
Implementation of policy gradient methods for "Short corridor with switched 
actions" (Example 13.1) of Sutton and Barto's "Reinforcement learning"
"""

import numpy as np

#%%

def REINFORCE_with_baseline(env, feat_vec, ALPHA_TH, NUM_EPISODES, GAMMA, ALPHA_W):
    """Implementation of REINFORCE with baseline (p. 330 in Sutton 
    and Barto's "Reinforcement Learning") for short corridor example 13.1
    

    Parameters
    ----------
    env : environment
        short corridor with switched actions
    feat_vec : dict
        feature vectors for state-action pairs.
    ALPHA_TH : float
        step size for policy optimization.
    NUM_EPISODES : float
        number of episodes.
    GAMMA : float
        discount factor.
    ALPHA_W : float
        step size for value function optimization.

    Returns
    -------
    total_rewards: list
        total rewards per episode.
    pi : dict
        probability of action in given state.

    """

    
    theta = init_policy_params()
    w = init_VF_params()
    
    total_rewards = []
    for episode in range(NUM_EPISODES):
        done = False
        
        state = env.reset()
        pi = policy(feat_vec, env.states, env.actions, theta)
        action = choose_action(pi, state, env.actions)

        states = [state]
        actions = [action]
        rewards = []
        
        while not done:
            state, reward, done = env.step(state, action)
            action = choose_action(pi, state, env.actions)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        R = np.array(rewards[:len(states)])      
        for step_count in range(len(states)):
            G = 0
            for k in range(step_count +1, len(states)):
                G += R[k-1]*GAMMA**(k-step_count-1)
            
            state = states[step_count]
            action = actions[step_count]
            
            delta = G - w
            w += ALPHA_W*delta
            theta += ALPHA_TH*(GAMMA**step_count)*delta*feat_vec[(state, action)]/pi[(state, action)]
            pi = policy(feat_vec, env.states, env.actions, theta)

        total_rewards.append(sum(rewards))
    
    return total_rewards, pi

#%%

def policy(feat_vec, states, actions, theta):
    """
    Define policy (parameterization) for short corridor problem

    Parameters
    ----------
    feat_vec : dict
        feature vectors for state-action pairs.
    states : ndarray
        states of short corridor.
    actions : ndarray
        actions of short corridor.
    theta : ndarray
        parameter vector.

    Returns
    -------
    pi : dict
        probability of action in given state.

    """
    pi = {}
    for state in states:
        sum_actions = 0
        for action in actions:
            h = np.dot(theta, feat_vec[(state, action)])
            pi[(state, action)] = np.exp(h)
            
            sum_actions += pi[(state, action)]    
            
        for action in actions: # normalization
            pi[(state, action)] /= sum_actions

    return pi

def init_policy_params(dim = 2):
    """
    Initialize policy parameters for short corridor problem randomly

    Parameters
    ----------
    dim : int, optional
        dimension of parameter vector. The default is 2, which corresponds to
        dimension of feature vector for the short corridor task.

    Returns
    -------
    theta : ndarray
        parameter vector.

    """

    theta = np.array([np.random.randn(),np.random.randn()])
    return theta 

def init_VF_params(dim = 1):
    """
    Initialize value function parameters for short corridor problem to zero

    Parameters
    ----------
    dim : int, optional
        dimension of parameter vector. The default is 1.

    Returns
    -------
    w : ndarray
        parameter vector.

    """

    w = 0
    return w

def choose_action(pi, state, actions):
    """
    Choose action for given state from probability distribution pi

    Parameters
    ----------
    pi : dict
        probability of action in given state.
    state : int
        given state
    actions : ndarray
        actions in short corridor task.

    Returns
    -------
    action : int
        action of agent according to probability pi.

    """
    action =  np.random.choice(actions, p=[pi[(state,-1)], pi[(state,1)]])
    
    return action


#%%
if __name__ == "__main__":
    # Run REINFORCE with baseline
    
    from shortcorridor import ShortCorridor
    from shortcorridor import FeatureVectors
    
    NUM_EPISODES = 1000
    ALPHA_TH = 2**(-9) # stepsize policy gradient
    GAMMA = 0.9
    ALPHA_W = 2**(-6) # stepsize VF gradient
    
    FV = FeatureVectors() # FV.x_vec : feature vector policy parameterization
    SC = ShortCorridor() # environment
    

    rew_per_ep, pi =  REINFORCE_with_baseline(SC, FV.x_vec, ALPHA_TH, 
                                              NUM_EPISODES, GAMMA, ALPHA_W)

    print(pi)