"""
Implementation of simple bandit and gradient bandit algorithm from chapter 2
of Sutton andBarto's "Reinforcement Learning" 
"""

import numpy as np
from scipy.special import softmax
        
def simple_bandit_alg(MAB, epsilon, max_steps, alpha = 0, Q1 = 0,
                      UCB = False, c = 2, run = 1):
    '''
    Calculate Q values of k-armed bandit by epsilon-greedy policy and sample
    average of observed rewards
    
    Paramaters
    ---------
    MAB: 
        Multi-armed bandit
    epsilon: float
        epsilon value of epsilon greedy policy
    max_steps: int
        maximum number of step for calculation of q values
    alpha: float
        step parameter \in [0,1]
        for alpha = 0 incremental steps are used
    Q1: 
        initial Q values
    UCB bool
        use upper confidence boundary or not
    c: 
        constant for UCB (only if UCB is true)
    run: int
        set seed of random number generator for every run
    Output
    ---------
    q: ndarray, shape (k,)
        q values which define the bandit
    Q: ndarray, shape (k,)
        estimated Q values for every one of the k possible actions
    pct_max_q_a: ndarray, shape (max_steps,)
        percentage of correctly estimated maximum q values during timesteps
    
    '''
    # initialize action values and counter for every possible of k actions
    Q = Q1*np.ones(MAB.k)
    N = np.zeros(MAB.k)
    pct_max_q_a = np.zeros(max_steps)
    
    ct_ind_q_max=0
    for step in range(0, max_steps):

        if step == 0:
            action = np.random.randint(0, MAB.k)
        else:
            
            CI=0
            if UCB:
                if N[action]==0:
                    CI = 0
                else: 
                    CI = c*np.sqrt(np.log(step)/N[action])

            population = [np.argmax(Q + CI), np.random.randint(0, MAB.k)]
            weights = [1 - epsilon, epsilon]
            action = np.random.choice(a=population, size=1, p=weights)      
       
        if MAB.bandit_type == "stationary":
            MAB.seed_q = run
            q, R =  MAB.bandit(action)
        elif MAB.bandit_type == "nonstationary":
            if step == 0:
                q = np.zeros(MAB.k) 
            q, R =  MAB.bandit(action, q)
        
        if alpha == 0:
            N[action] = N[action] + 1
            Q[action] = Q[action] + 1/N[action]*(R - Q[action])
        else:
            Q[action] = Q[action] + alpha*(R - Q[action])

        # percentage of correctly chosen max q value
        if action == np.argmax(q):
            ct_ind_q_max += 1
        pct_max_q_a[step] = ct_ind_q_max/(step + 1)

    return(Q, q, pct_max_q_a)


def gradient_bandit_alg(MAB, max_steps, alpha = 0.01, baseline = True, 
                        Q1 = 0, run = 1):
    '''
    Solve k-arm bandit problem by gradient ascent
    
    Paramaters
    ---------
    MAB: 
        Multi-armed bandit
    max_steps: int
        maximum number of step for calculation of q values
    alpha: float
        step parameter \in (0,1]
    baseline: boolean
        use baseline or not
    Q1: float
        initial Q values
    run: int
        set seed of random number generator for every run
    Output
    ---------
    q: ndarray, shape (k,)
        q values which define the bandit
    pct_max_q_a: ndarray, shape (max_steps,)
        percentage of correctly estimated maximum q values during timesteps
    '''
    
    if alpha <= 0:
        raise ValueError("alpha needs to be larger than 0")
 
    H = Q1*np.ones(MAB.k)
    pi = np.zeros(MAB.k)
    population = [i for i in range(0,MAB.k)]
    
    ct_ind_q_max=0
    R_bl =  np.zeros(max_steps)
    pct_max_q_a = np.zeros(max_steps)
    
    for step in range(0, max_steps):
        # policy
        pi = softmax(H)
        # choose action randomly according to policy
        weights = pi
        action = np.random.choice(population, size=1, p=weights)      
            
        # calculate reward
        MAB.seeq_q = run
        q, R = MAB.stationary_bandit(action)
        
        if baseline: # calculate average rewards up to t
            R_bl[step] = R 
            R_avg = np.mean(R_bl[0:step+1])
        else:
            R_avg = 0
            
        # update preference
        act = int(action)
        H[:act] = H[:act] - alpha*(R - R_avg)*pi[:act]  
        H[act] = H[act] + alpha*(R - R_avg)*(1 - pi[act])
        H[act+1:] = H[act+1:] - alpha*(R - R_avg)*pi[act+1:]    
 
        # percentage of correctly chosen max q value
        if action == np.argmax(q):
            ct_ind_q_max += 1
        pct_max_q_a[step] = ct_ind_q_max/(step + 1)

    return(q, pct_max_q_a)
        

#%% Test
if __name__ == "__main__":
    
    max_steps = 1000 # max time steps
    k_arms = 10  # number of arms of bandit
    epsilon = 0.1 # epsilon value for epsilon greedy policy
    alpha = 0.1 # step size parameter; alpha = 0: incremental steps
    Q1 = 0 # initial Q values
    UCB = False # upper confidence boundary
    c = 2 # constant for UCB only
    baseline = True #  gradient bandit algorithm
    bandit_algorithm = "simple" # simple or gradient
    bandit_type = "stationary" # stationary or nonstationary bandit
    
    from mabandit import MultiArmedBandit
    
    MAB = MultiArmedBandit(k_arms, bandit_type)
    
    print("run simple bandit algorithm")
    Q, q, pct_max_q_a =  simple_bandit_alg(MAB, epsilon, max_steps, alpha, Q1, 
                                           UCB, c, run = 1)
    print(Q)
    
    print("run gradient bandit algorithm")
    q, pct_max_q_a = gradient_bandit_alg(MAB, max_steps, alpha, baseline, Q1, 
                                         run = 1)
    print(q)