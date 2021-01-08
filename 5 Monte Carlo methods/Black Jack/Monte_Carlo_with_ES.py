"""
Implementation of Black jack example 5.1 (Monte Carlo prediction) and example 
5.3 (Monte Carlo control with exploring starts) from Sutton and Barto's 
"Reinforcement Learning"
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time
from tqdm import tqdm

from blackjack import BlackJack

#%%
def init_returns(states):
    """
    Generate dictionrary for saving returns of given states

    Parameters
    ----------
    states : list of tuples
            List with all possible combinations of dealer card, player card, 
            and whether or not the player has a usable ace

    Returns
    -------
    Returns : dict
        Returns of episodes for given states
    """
    
    Returns = {}
    for state in states:
        Returns[state] = []
    
    return Returns

def init_value_function(states):
    """
    Generate initial value function

    Parameters
    ----------
    states : list of tuples
            List with all possible combinations of dealer card, player card, 
            and whether or not the player has a usable ace

    Returns
    -------
    VF : dict
        value of value function for given states

    """
    VF = {}
    for state in states:
        VF[state] = 0
    
    return VF


def policy(states):
    """
    Generate policy

    Parameters
    ----------
    states : list of tuples
            List with all possible combinations of dealer card, player card, 
            and whether or not the player has a usable ace
    Returns
    -------
    pi : dict
        policy which maps states (keys) onto actions (values)

    """
    pi = {}
    for state in states:
        if state[1] >= 20:
            pi[state] = 1 # stick
        else:
            pi[state] = 0 # hit
        
    return pi

#%%

def init_returns_MC_ES(states, nr_actions):
    """
    Generate dictionrary for saving returns of state-action pair

    Parameters
    ----------
    states : list of tuples
            List with all possible combinations of dealer card, player card, 
            and whether or not the player has a usable ace

    Returns
    -------
    Returns : dict
        Returns of episodes for given states and actions
    """
    
    Returns = {}
    for state in states:
        for action in range(nr_actions):
            Returns[state, action] = []
    
    return Returns

def init_q_values(states, nr_actions):
    """
    Initialize Q values

    Parameters
    ----------
    states : list of tuples
            List with all possible combinations of dealer card, player card, 
            and whether or not the player has a usable ace
    nr_actions : int
            nr of of possible actions

    Returns
    -------
    Q : dict
        Q values for given state and actions

    """
    Q = {}
    for state in states:
        Q[state] = np.zeros(nr_actions)

    return Q


#%%

def first_visit_MC_prediction(env, states, policy_player, num_episodes, gamma):
    """
    First visit Monte-Carlo prediction for Black Jack environment

    Parameters
    ----------
    env : object
        Reinforcement learning environment for Blackjack
    num_episodes : int
        Number of episodes
    states: list of tuples
        List with all possible combinations of dealer card, player card,
        and whether or not the player has a usable ace
    policy_player: dict
        policy which maps states (keys) onto actions (values)

    Returns
    -------
    None.

    """

    Returns = init_returns(states)

    VF = init_value_function(states)

    for epsiode in tqdm(range(num_episodes)):
        current_state = env.reset()
        done = False

        # generate epsiode
        episode_states = []
        episode_actions = []
        episode_rewards = []
     
        while not done:
            action = policy_player[tuple(current_state)]
            next_state, reward, done, __ = env.step(current_state, action)
            
            episode_states.append(current_state) 
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            current_state = next_state

        G = 0
        for count_state in reversed(range(len(episode_states))):
            G = gamma*G + episode_rewards[count_state]
            state = episode_states[count_state]

            Returns[tuple(state)].append(G)
            VF[tuple(state)] = sum(Returns[tuple(state)])/len(Returns[tuple(state)])
            # remark: replace by incremental update for better performance
            
    return VF

    
#%%
def Monte_carlo_exploring_starts(env, states, init_policy_player, num_episodes,
                                 gamma):
    """
    Monte-Carlo control with exploring states for Black Jack environment

    Parameters
    ----------
    env : 
        Reinforcement learning environment for Blackjack
    num_episodes : int
        Number of episodes
    states: list of tuples
        List with all possible combinations of dealer card, player card, 
        and whether or not the player has a usable ace
    init_policy_player: dict
        Initial policy which maps states (keys) onto actions (values)
    Returns
    -------
    policy_player: dict
        policy which maps states (keys) onto actions (values)

    """

    policy_player = init_policy_player
    Q = init_q_values(states, env.nr_actions)
    Returns = init_returns_MC_ES(states, env.nr_actions)
    
    #start_time = time.time()
    for epsiode in tqdm(range(num_episodes)):    
        current_state = env.reset()
        action = np.random.choice(env.action_space())
        
        done = False
        
        # generate epsiode
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        while not done:
            next_state, reward, done, __ = env.step(current_state, action)
            
            episode_states.append(current_state) 
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            current_state = next_state
            if not done:
                action = policy_player[tuple(current_state)]
                
        G = 0
        for count_state in reversed(range(len(episode_states))):
            G = gamma*G + episode_rewards[count_state]
            state = episode_states[count_state]
            action  = episode_actions[count_state]
            
            Returns[tuple(state),action].append(G)
            Q[tuple(state)][action] = sum(Returns[tuple(state),action])/len(Returns[tuple(state),action])
            # remark: replace by incremental update for better performance

            policy_player[tuple(state)] = np.argmax(Q[tuple(state)])
    #elapsed_time = time.time() - start_time     
    #print(str(elapsed_time))

    return policy_player

#%%

def plot_policy(policy):
    """
    Plot policy as function of dealer showing card and player sum
    """
    action = np.array(list(policy.values()))
    
    dealer_card , player_card, usable_ace = zip(*policy.keys())
    
    action_no_ua = np.reshape(action[0:100], [10,10])
    action_ua = np.reshape(action[100:], [10,10])
    
    plt.figure() 
    plt.imshow(action_no_ua)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 10, 1));
    ax.set_yticks(np.arange(0, 10, 1))
    ax.set_xticklabels(np.arange(1, 11, 1));
    ax.set_yticklabels(np.arange(12, 22, 1))
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")
    plt.title("Policy (no usable ace); blue: hit, yellow: stick")
    plt.gca().invert_yaxis()

    plt.figure() 
    plt.imshow(action_ua)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 10, 1));
    ax.set_yticks(np.arange(0, 10, 1))
    ax.set_xticklabels(np.arange(1, 11, 1));
    ax.set_yticklabels(np.arange(12, 22, 1))
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")
    plt.title("Policy (usable ace); blue: hit, yellow: stick")
    plt.gca().invert_yaxis()

def plot_value_function(Returns):
    """
    Plot value function as function of dealer showing card and player sum
    """
    values= np.array(list(Returns.values()))
    dealer_card , player_card, usable_ace = zip(*Returns.keys())

    dealer = np.reshape(dealer_card[0:100], [10,10])
    player = np.reshape(player_card[0:100], [10,10])
    values_no_ua = np.reshape(values[0:100], [10,10])
    values_ua = np.reshape(values[100:], [10,10])

    fig = plt.figure() 
    ax = fig.gca(projection='3d') 
    ax.plot_surface(dealer, player, values_no_ua,  cmap=cm.coolwarm,
                       linewidth=0)
    plt.title("State value function (no usable ace)")
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")

    fig = plt.figure() 
    ax = fig.gca(projection='3d') 
    ax.plot_surface(dealer, player, values_ua,  cmap=cm.coolwarm,
                       linewidth=0)
    plt.title("State value function (usable ace)")
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")


#%%
if __name__ ==  "__main__":   

    num_episodes = 500000
    gamma = 1
    env = BlackJack()
    states = env.state_space()
    init_policy_player = policy(states)
    
    policy_player = Monte_carlo_exploring_starts(env, states, 
                                                 init_policy_player,
                                                 num_episodes, gamma) 
    #%%
    VF = first_visit_MC_prediction(env,  states, policy_player, 
                                   num_episodes, gamma) 
    
    #%%
    plot_policy(policy_player)  
    plot_value_function(VF)
    
