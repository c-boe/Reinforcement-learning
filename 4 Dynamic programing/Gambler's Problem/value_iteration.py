"""
Implementation of Exercise 4.9 in Chapter 4 of Sutton and Barto's "Reinforcement 
Learning" 
"""
import matplotlib.pyplot as plt
import numpy as np
from gp import GamblersProblem


def value_iteration(env, theta):
    '''
    Value iteration

    Parameters
    ----------
    env :
        Gambler's problem MDP
    theta : float
       treshold for value iteration

    Returns
    -------
    V_optimal : ndarray, shape (number of states,)
        optimal state-value function
    pi : ndarray, shape (number of states,)
        policy

    '''
    
    states = env.state_space()
    terminal_states = env.terminal_states
    p_h = env.p_h
    
    V_init = int_val_fun(terminal_states)    
        
    # Value iteration
    V_optimal = value_loop(V_init, p_h, states)
        
    pi = det_policy(V_optimal, states, terminal_states, p_h)

    return V_optimal, pi

def int_val_fun(terminal_states):
    '''
    Initial value function

    Parameters
    ----------
    terminal_states : tuple
        terminal states of MDP

    Returns
    -------
    V_init : ndarray, shape (number of states,)
        initial state value function
    '''    
    
    V_init = np.zeros(terminal_states[1] + 1)
        
    return V_init
        

def value_loop(V, p_h, states):
    '''

    Parameters
    ----------
    V : ndarray, shape (number of states,)
        state value function
    p_h: float
        probability for head
    states : 
       states of MDP

    Returns
    -------
    V : ndarray, shape (number of states,)
        state value function

    '''
    v_tmp = np.zeros(V.shape)
        
    while(1):
        Delta = 0
        for state in states:
            v_tmp[state] = V[state]
                
            V_prev = 0
            # determine possible actions for given state
            actions = det_actions(state, terminal_states)
            for action in actions:
                # print(s, a)
                # Calculate value function for given action
                V_next = val_fun(state, action, V, p_h)
                if V_next > V_prev:
                    V_prev = V_next
            V[state] = V_prev    
        # compare value function with previous value function
        Delta = np.max([Delta, np.linalg.norm(v_tmp - V)])
        if Delta < theta:
            break

    return V

        
def det_actions(current_state, terminal_states):
    '''
    determine action space for the given current state

    Parameters
    ----------
    current_state : float
        current state of MDP
    terminal_states : tuple , shape (1,1)
        terminal states of MDP

    Returns
    -------
    actions : iterable
        possible action for current state

    '''
    
    actions = range(1, min(current_state,
                               terminal_states[1] - current_state) + 1)
        
    return actions

    
        
def det_policy(V, states, terminal_states, p_h):
    '''
    Determine policy from optimal value function

    Parameters
    ----------
    V : ndarray, shape (number of states,)
        state value function
    states : iterable
        states of MDP
    terminal_states : tuple
        terminal states of MDP
    p_h : float
        probability of head

    Returns
    -------
    pi : ndarray, shape (number of states,)
        policy
    '''

    pi = np.zeros(V.shape)
        
    for state in states:
        Q_prev = 0
        # determine possible actions for given state
        actions = det_actions(state, terminal_states)
        for action in actions:
            # Calculate Q value function for given state (cars_A, cars_B) 
            # and action a
            Q_next = val_fun(state, action, V, p_h)
            if Q_next > Q_prev:
                Q_prev = Q_next
                pi[state] = action
                        
    return pi

def val_fun(current_state, action, V, p_h):
    '''
    Determine value of value function V(s) for given state s

    Parameters
    ----------
    current_state : float
        current state of MDP
    action : action
        current action
    V : ndarray, shape (number of states,)
        state value function
    p_h : float
        probability of head

    Returns
    -------
    V_value : float
        value of current state

    '''

    V_value = 0
        
    # possible next states after taking action a
    next_state_up = current_state + action
    next_state_down = current_state - action
        
    if next_state_up == 100: # terminal state
        reward = 1
        V_value = (p_h*(reward + V[next_state_up]) + 
                            (1 - p_h)*V[next_state_down])
    elif next_state_down == 0:  # terminal state
        V_value = (p_h*(V[next_state_up]) + 
                            (1 - p_h)*V[next_state_down])
    else:
        V_value = (p_h*(V[next_state_up]) + 
                            (1 - p_h)*V[next_state_down])

    return V_value


#%% 

def plot_val_fun(V):
    '''
    plot state value function
    '''
    plt.figure()
    plt.plot(V[1:-1])
    plt.title("Value function")
    plt.xlabel("Capital")
    plt.ylabel("value")

def plot_policy(pi):
    '''
    plot policy
    '''
        
    plt.figure()
    plt.plot(pi[1:-1])
    plt.title("policy function")
    plt.xlabel("Capital")
    plt.ylabel("Stake")
    
#%%
        
if __name__ == "__main__":
    
    theta = 1e-10
    terminal_states = (0, 100)
    num_states = 99
    p_h = 0.25
    
    
    env = GamblersProblem(num_states, terminal_states, p_h)
    V, pi = value_iteration(env, theta)
    
    plot_val_fun(V)
    plot_policy(pi)