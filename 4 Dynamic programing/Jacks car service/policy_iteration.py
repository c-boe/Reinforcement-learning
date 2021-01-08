"""
Implementation of Exercise 4.7 in Chapter 4 of Sutton and Barto's "Reinforcement 
Learning" 
"""

import numpy as np
from jackscarservice import JacksCarService
import time
import matplotlib.pyplot as plt

#%%

def init_val_fun(max_cars_per_loc):
    '''
    Initialize state value function for iterative policy improvement
    
    Parameters
    ----------
    max_cars_per_loc : int
        maximum number of cars per location.
    Returns
    -------
    V : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        State value function
    '''

    # state value function
    V = np.zeros((max_cars_per_loc + 1, max_cars_per_loc + 1))#
    
    return V

def init_policy(max_cars_per_loc):
    """
    Initialize policy for iterative policy improvement

    Parameters
    ----------
    max_cars_per_loc : int
        maximum number of cars per location.

    Returns
    -------
    pi : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        policy.

    """

    # policy
    pi = np.zeros((max_cars_per_loc + 1, max_cars_per_loc + 1))

    return pi


#%%

def policy_iteration(env, theta, gamma):
    '''
    Policy iteration algorithm

    Parameters
    ----------
    env :
        Jacks car service MDP
    theta : float
       treshold for policy evaluation
    gamma : float
        discount factor of DP

    Returns
    -------
    V : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        state value function
    pi : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        policy

    '''
    actions = env.action_space()
    
    V = init_val_fun(env.max_cars_per_loc)    
    pi = init_policy(env.max_cars_per_loc)

    pi_tmp = np.random.rand(pi.shape[0],pi.shape[1])

    while not np.array_equal(pi, pi_tmp):
        pi_tmp = pi
        # evaluate value function
        V = eval_policy(env, pi_tmp, V, theta, gamma)
        # improve_policy
        pi = improve_policy(env, V, actions)
    
    # plots

    return V, pi


def eval_policy(env, pi, V, theta, gamma):
    '''
    Calculate state value function for given policy

    Parameters
    ----------
    env :
        JCS MDP
    pi : ndarray
        policy
    V : ndarray
        state value function
    theta : float
       treshold for policy evaluation
    gamma : float
        discount factor of DP
        
    Returns
    -------
    V : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        state value function
    '''

    v_tmp = np.zeros(V.shape)
    Delta = theta + 1
    
    while theta < Delta:
        Delta = 0
        for cars_A in range(0,  V.shape[0]):
            for cars_B in range(0, V.shape[1]):
                
                v_tmp[cars_A, cars_B] = V[cars_A, cars_B]
                # Calculate value function for given action
                V[cars_A, cars_B] = value_function(env, (cars_A, cars_B), 
                                                     pi[cars_A, cars_B], V, gamma)
                # compare value function with previous value function
                Delta = np.max([Delta, np.abs(v_tmp[cars_A, cars_B] - 
                                              V[cars_A, cars_B])])
            
    return V

def improve_policy(env, V, actions):
    '''
    Iterate improvement of policy

    Parameters
    ----------
    env :
        JCS MDP
    V : ndarray
        state value function
    actions: int 
        number of cars moved
    Returns
    -------
    V : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        state value function
    '''
    pi = np.zeros(V.shape)

    for cars_A in range(0,  V.shape[0]):
        for cars_B in range(0, V.shape[1]):
            Q_prev = 0
            for action in actions:
                # maximum number of cars which can be shifted
                if cars_A + action >= 0 and cars_B - action >= 0:
                    # Calculate Q value function for given state (cars_A, cars_B) 
                    # and action a
                    Q_next = value_function(env, (cars_A, cars_B), action, V, gamma)
                    
                    if Q_next > Q_prev:
                        Q_prev = Q_next
                        pi[cars_A, cars_B] = action 
                    
    return pi

def value_function(env, current_state, action, Value_function, gamma):
    """
    Calculate value function for given initial state and action

    Parameters
    ----------
    env :
        JCS MDP
    current_state : tuple
        current state consisting of number of cars at A and B
    action :
        number of cars shifted from A to B
    Value_function : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        state value function

    Returns
    -------
    V_value : float
        state value for current state

    """
    
    # probability transition matrix
    p = env.PTM_dict["p"]
    n_B_ret = env.PTM_dict["n_B_ret"] 
    n_A_ret = env.PTM_dict["n_A_ret"]
    n_B_req = env.PTM_dict["n_B_req"]
    n_A_req = env.PTM_dict["n_A_req"]

    # calculate how many cars can be rented currently
    # (assume that requests and returns happen at the same)
    n_A_req_2 = np.array([current_state[0] + n_A_ret + int(action),
                          n_A_req]).min(axis = 0)
    n_B_req_2 = np.array([current_state[1] + n_B_ret - int(action), 
                          n_B_req]).min(axis = 0)
    
    # send all additional cars away
    next_state_A = np.array([current_state[0] + n_A_ret - n_A_req_2  + int(action),
                             env.max_cars_per_loc*np.ones(len(n_A_ret))]).min(axis = 0)
    next_state_B = np.array([current_state[1] + n_B_ret - n_B_req_2  - int(action),
                             env.max_cars_per_loc*np.ones(len(n_A_ret))]).min(axis = 0)
    
    next_state_A = next_state_A.astype(int)
    next_state_B = next_state_B.astype(int)
    
    # calculate rewards and value function
    parking_A = next_state_A > env.nr_free_parking
    parking_B = next_state_B > env.nr_free_parking
    
    reward_parking = env.reward_parking_lot*(next_state_A*parking_A + 
                                             next_state_B*parking_B)
    reward_rent = (env.reward_req*(n_A_req_2 + n_B_req_2))
    reward = reward_rent + reward_parking
    
    reward_shift = env.reward_shift*((action - env.free_shift_AB)*(action>0) - 
                                      (action)*(action<0))    
    
    VF_state =  p*(reward + gamma*Value_function[next_state_A, next_state_B])
    V_value = VF_state.sum() + reward_shift

    return V_value

#%%

def plot_VF(V):
    '''
    Plot value function as a function of number of cars at each location

    Parameters
    ----------
    V : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        state value function

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.imshow(V)
    plt.colorbar()
    plt.title("Value function")
    plt.xlabel("Cars at B")
    plt.ylabel("Cars at A")


def plot_policy(pi):
    '''
    Plot policy as a function of number of cars at each location

    Parameters
    ----------
    pi : ndarray, shape (max_cars_per_loc, max_cars_per_loc)
        policy

    Returns
    -------
    None.

    '''

    plt.figure()
    plt.imshow(pi)
    plt.colorbar()
    plt.title("policy function")
    plt.xlabel("Cars at B")
    plt.ylabel("Cars at A")

#%% 
if __name__ == '__main__':
    """
    Set parameters and run jacks car service (Example 4.2 and Exercise 4.7
    in Chapter 4 of Sutton and Barto's "Reinforcement Learning" )
    """
    max_cars_per_loc = 20 # max number of cars per location
    gamma = 0.9 # discount factor
    theta = 0.01 # treshold for policy evaluation
        
    lbd_req_A = 3 # lambda paramters of poisson distribution for
    lbd_ret_A = 3 # request and return at location A
        
    lbd_req_B = 4 # location B
    lbd_ret_B = 2
            
    max_n = 8 # defines number of considered terms in poisson distr.
        
    min_shift = -5 # max. number of shifted cars (action space)
    max_shift = 5
        
    reward_req = 10 # reward for requested car
    reward_shift = -2 # penalty for car moved over night
    
        
    nr_free_parking = 10 # number of free parking cars
    reward_parking_lot = 0  # penalty for parking more cars over night
    
    free_shift_AB = 1    # shift first car from A to B for free

    start = time.time()
    env_JCS = JacksCarService(
                    max_cars_per_loc, min_shift, max_shift,
                    lbd_req_A, lbd_ret_A, lbd_req_B, lbd_ret_B, max_n,
                    reward_req, reward_shift, reward_parking_lot, 
                    nr_free_parking, free_shift_AB
                    )

    V, pi = policy_iteration(env_JCS, theta, gamma)
    
    end = time.time()
    print(end - start)
    
    plot_policy(pi)
    plot_VF(V)