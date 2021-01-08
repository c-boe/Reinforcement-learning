"""
Implementation of off-policy Monte-Carlo control for racetrack environment 
from Exercise 5.12 in Sutton and Barto's "Reinforcement learning"
"""
import numpy as np
from tqdm import tqdm

from racetrack import Racetrack


#%%
def init_q_values(states, nr_actions):
    """
    Initilize q values

    Parameters
    ----------
    states : list of tuples
        States consist of positions and velocity of agent
    nr_actions : int
        number of policy actions of agent in environment

    Returns
    -------
    Q : dict
        Dictionary with (initial) q values of each action in each state

    """
    Q = {}
    for pos in states[0]:
        for vel in states[1]:
         #   Q[pos, vel] = -np.random.uniform(list(range(0,env.nr_actions)))*5
         Q[pos, vel] = np.random.uniform(-100, -99, nr_actions)
    return Q
    
def soft_policy(states, nr_actions):
    """
    Define behaviour policy of off-policy MC method as soft policy with 
    uniform probability distribution

    Parameters
    ----------
    states : list of tuples
        States consist of positions and velocity of agent
    nr_actions : int
        number of policy actions of agent in environment

    Returns
    -------
    b : dict
        Dictionary representing the policy  which maps states on actions. Each 
        action has the same probability of 1/nr_actions. Therefore 
        b(action|state) is defined as b[state] =  and utilized correspondingly
        in the off-policy MC algorithm

    """
    b = {}
    for pos in states[0]:
        for vel in states[1]:
            b[pos, vel] = list(range(0,nr_actions))
    
    return b

def init_policy(states, Q):
    """
    Define initial target policy of off-policy MC method

    Parameters
    ----------
    states : list of tuples
        States consist of positions and velocity of agent
    Q : dict
        Dictionary with q values of each action in each state

    Returns
    -------
    pi : dict
        Dictionary representing the policy which maps states on actions.

    """
    pi = {}
    for pos in states[0]:
        for vel in states[1]:
            pi[pos, vel] = np.argmax(Q[pos, vel])
    
    return pi


def init_C(states, nr_actions):
    """
    Initialize sum of importance sampling rations

    Parameters
    ----------
    states : list of tuples
        States consist of positions and velocity of agent
    nr_actions : int
        number of policy actions of agent in environment
        
    Returns
    -------
    C : dict
        sum of importance sampling rations

    """
    C = {}
    for pos in states[0]:
        for vel in states[1]:
            C[pos, vel] = np.zeros(nr_actions)
    
    return C
#%%
def Off_policy_MC(num_episodes, gamma, Track_img):
    """
    Off-policy Monte-Carlo control for Racetrack environment 
    from Exercise 5.12 in Sutton and Barto's "Reinforcement learning"

    Parameters
    ----------
    num_episodes : int
        number of episodes
    gamma : float
        discount factor
    Track : image
        The track is defined by the gray scale values of the image.
        starting line: 127
        finish line: 195
        racetrack: 0
        
        (127 and 195 are the standard gray values in Paint ;-) )
        
    Returns
    -------
    pi : dict
        Dictionary representing the policy which maps states on actions.
    env : obj
        Racetrack environment

    """

    env = Racetrack(Track_img)

    states = env.state_space()
    env.plot_racetrack(env.reset())
    Q = init_q_values(states, env.nr_actions)
    C = init_C(states, env.nr_actions)
    
    pi = init_policy(states, Q)
    
    for episode in tqdm(range(num_episodes)):
        b = soft_policy(states, env.nr_actions)
        state = env.reset()
        action = np.random.choice(b[state[0],state[1]])
        done = False
        
        state_seq = []
        action_seq = []
        reward_seq = []
        
        while not done:
            state_seq.append(state)
            action_seq.append(action)
            
            new_state, reward, done, final_state = env.step(state, action)
            
            if not done:
                state = new_state
                action = np.random.choice(b[state[0],state[1]])
            reward_seq.append(reward)

        G = 0
        W = 1
        
        for step in range(len(state_seq)-1, 0, -1):
            state = state_seq[step]
            action = action_seq[step]
            reward = reward_seq[step]

            G = gamma*G + reward

            C[state[0],state[1]][action] += W
            Q[state[0],state[1]][action] += W/C[state[0],state[1]][action]*(G - Q[state[0],state[1]][action])

            pi[state[0],state[1]] = np.argmax(Q[state[0],state[1]])
            if action != pi[state[0],state[1]]:
                break
            W = W/(1/len(b[state[0],state[1]])) # soft policy with uniform probability distr
            
            # Learn only from paths crossing the finish line
            if state[0][0] == 0:
                break

    return pi, env


#%%
def evaluate_policy(env, pi, state):
    """
    Run episode with given policy and initial state

    Parameters
    ----------
    pi : ndarray
        policy for given state
    state : list of tuples
         state = [position, velocity]

    Returns
    -------
    G: int
        Return after terminated episode

    """
    
    action = pi[state[0],state[1]]
    
    done = False
    G = 0
    env.plot_racetrack(state)
    
    while not done:
        new_state, reward, done, __ = env.step(state, action)
        G = gamma*G + reward
        
        if state == new_state or new_state[0][0] == 0:
            print("Policy failed for initial position. Agent has not found a " +
                  "direct route from initial state to finish line within episodes")
            break
        
        state = new_state
        if done:
            state = [(state[0][0], env.finish_line_x[0]),state[1]]
            
        action = pi[state[0],state[1]]   
        env.plot_racetrack(state)  

    return G

#%%
if __name__ == "__main__":

    num_episodes = 1000
    gamma = 1
    Track_img = "Track1.png"
    
    pi, env = Off_policy_MC(num_episodes, gamma, Track_img)
    
    #%% run episode with learned policy
    
    state = env.reset()
    evaluate_policy(env, pi, state) 

