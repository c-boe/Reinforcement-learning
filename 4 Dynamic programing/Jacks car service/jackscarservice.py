"""
Implementation of Exercise 4.7 in Chapter 4 of Sutton and Barto's "Reinforcement 
Learning" 
"""
import numpy as np


class JacksCarService():
    '''
    Define Jacks's car service MDP
    '''
    
    def __init__(self, max_cars_per_loc, min_shift, max_shift,
                 lbd_req_A, lbd_ret_A, lbd_req_B, lbd_ret_B, max_n,
                 reward_req, reward_shift, reward_parking_lot, 
                 nr_free_parking, free_shift_AB):
        
        self.max_cars_per_loc = max_cars_per_loc # max number of cars per location
        self.min_shift = min_shift # max. number of shifted cars (action space)
        self.max_shift = max_shift
        
        
        self.lbd_req_A = lbd_req_A # lambda paramters of poisson distribution for
        self.lbd_ret_A = lbd_ret_A # request and return at location A
        
        self.lbd_req_B = lbd_req_B # location B
        self.lbd_ret_B = lbd_ret_B
        
        self.max_n = max_n # defines number of considered shifted cars in poisson distr.
                
        self.reward_req = reward_req # reward for requested car
        self.reward_shift = reward_shift  # penalty for car moved over night
        
        self.nr_free_parking = nr_free_parking  # number of free parking cars over night
        self.reward_parking_lot = reward_parking_lot # penalty for more  parked cars over night
    
        self.free_shift_AB = free_shift_AB    #first car from A to be for free

        self.PTM_dict = self.PTM()

    def state_space(self):
        '''
        State space

        Returns
        -------
        states : list of tuples
            possible states of JCS (nr_cars_A, n_cars_B)

        '''
        a = range(self.max_cars_per_loc)
        b = range(self.max_cars_per_loc)
        
        
        A, B = np.meshgrid(a,b)
        
        A_vec = A.flatten()
        B_vec = B.flatten()
        
        states = list(zip(A_vec, B_vec))
        
        return states
        
    def action_space(self):
        '''
        Generate action space of Jacks Car service

        Returns
        -------
        actions: 
            possible actions of JCS MDP 

        '''
        actions = range(self.min_shift, self.max_shift + 1)
        
        return actions
    
    def PTM(self):
        '''
        Generate probability transition matrix for JCS

        Returns
        -------
        PTM: dict
            Dictionary with probability transition matrix p and number of 
            requested/ returned n_ret_A, n_ret_B, n_req_A, n_req_B

        '''
        n_A_ret, n_B_ret, n_A_req, n_B_req = self._cart_prod()

        # probability distributions
        pd_req_A = self._prob_distr(self.max_n, self.lbd_req_A)
        pd_req_B = self._prob_distr(self.max_n, self.lbd_req_B)
        pd_ret_A = self._prob_distr(self.max_n, self.lbd_ret_A)
        pd_ret_B = self._prob_distr(self.max_n, self.lbd_ret_B)
        
        # probability of transition to next_state
        p = (pd_req_A[n_A_req]*pd_ret_A[n_A_ret]*
             pd_req_B[n_B_req]*pd_ret_B[n_B_ret])
        
        PTM_dict = {}
        
        PTM_dict["p"] = p
        PTM_dict["n_B_ret"] = n_B_ret
        PTM_dict["n_A_ret"] = n_A_ret
        PTM_dict["n_B_req"] = n_B_req
        PTM_dict["n_A_req"] = n_B_req
    
        return PTM_dict
    
    def _cart_prod(self):
        """
        Generate all possible combinations of returned and requested cars
        at A and B (min. probability defined by max_n)
        
        Returns
        -------
        n_A_ret, n_B_ret, n_A_req, n_B_req : ndarray, shape ((self.max_n +1)**4,)
            Arrays representing all possible combinations of requested and
            returned cars at location A and B
        """
        n = np.array(range(0, self.max_n + 1))

        n_A_ret = np.tile(n, n.shape[0]**3)
        n_B_ret = np.tile(
            np.tile(n, (n.shape[0],1)).flatten(order="F"),n.shape[0]**2)
        n_A_req = np.tile(
            np.tile(n, (n.shape[0]**2,1)).flatten(order="F"),n.shape[0])
        n_B_req = np.tile(n, (n.shape[0]**3,1)).flatten(order="F")
       
        return(n_A_ret, n_B_ret, n_A_req, n_B_req)
        

    def _prob_distr(self, max_n, lbd):
        '''
        Poisson distribution
    
        Parameters
        ----------
        max_n : int
            Maximum number of considered terms in the poisson distribution
        lda : float
            parameter lambda of poisson distribution
    
        Returns
        -------
        pd : int
            poisson distribution with.
    
        '''
    
        # probabilities for requests at A and B
        pd = self._poisson_distribution(max_n, lbd)
        pd = pd/np.sum(pd)
        
        return pd
    
    def _poisson_distribution(self, max_n, lda):
        '''
        Poisson distribution
    
        Parameters
        ----------
        max_n : int
            Maximum number of considered terms in the poisson distribution
        lda : float
            parameter lambda of poisson distribution
    
        Returns
        -------
        pd : nd array
            poisson distribution with.
    
        '''
    
        pd = np.zeros(max_n + 1)
        for n in range(max_n + 1):
            pd[n] = lda**(n)/np.math.factorial(n)*np.exp(-lda)
        
        return pd