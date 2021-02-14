"""
Implementation of multi armed bandit from chapter 2 Sutton and Barto's 
"Reinforcement Learning" 
"""

import numpy as np

class MultiArmedBandit:
     
    def __init__(self, k, bandit_type, var_R = 1, seed_q = 1):
    
        self.k = k  # number of arms of bandit
        self.seed_q = seed_q # seed of random number generator
        self.bandit_type = bandit_type # stationary or nonstationary
        self.var_R = var_R # variance

        if bandit_type not in ["stationary", "nonstationary"]:
            raise ValueError("Bandit type unknown. Choose 'stationary'" +
                             "or 'nonstationary'")

    def bandit(self, action, q = 0):
        """Choose bandit algorithm"""
        if self.bandit_type == "stationary":
            q, R = self.stationary_bandit(action)
        elif self.bandit_type == "nonstationary":
            q, R = self.nonstationary_bandit(action, q)
        
        return q, R
    
    def stationary_bandit(self, action, seed_q = 1):
        ''' 
        Define k armed bandit through normal distributed q values and rewards
    
        Paramaters
        ---------
        action: int
            action, that defines chosen arm
        Output
        ---------
        R: float
            Reward for action a of k-armed bandit
        '''
        # q values as normal distribution with mean 0 and variance one
        np.random.seed(seed=self.seed_q)
        q = np.random.normal(0, 1, size=self.k)
    
        # Choose rewards from normal distribution with mean q(a) and variance one
        np.random.seed(seed=None)
        R = np.random.normal(q[action], self.var_R, size=1)

        return q, R
    
    def nonstationary_bandit(self, action, q):
        ''' 
        Define k armed bandit in which all q values start equal and than take 
        independent random walks through normal distributed q values and rewards
    
        Paramaters
        ---------
        action: 
            action, that defines chosen arm
        q: 
            q values from previous time step

        Output
        ---------
        R:
            Reward for action a of k-armed bandit
        '''
        # q values as normal distribution with mean 0 and variance one
        q += np.random.normal(0, 0.01, size=self.k)
    
        # Choose rewards from normal distribution with mean q(a) and variance one
        R = np.random.normal(q[action], self.var_R, size=1)
    
        return q, R
      
#%% Test
if __name__ == "__main__":
    

    k_arms = 10  # number of arms of bandit
    bandit_type = "stationary" # stationary or nonstationary bandit

    MAB = MultiArmedBandit(k_arms, bandit_type)
    
    action = np.random.randint(0, MAB.k)
    q, R =  MAB.bandit(action)
    
    print(q)