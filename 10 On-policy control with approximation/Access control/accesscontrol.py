"""
Environment for Example 10.2 of "Reinforcement learning" by Sutton and Barto
"""

import numpy as np

class AccessControl():
    """
    Access-Control Queuing Task environment
    """
    def __init__(self):
        self.priorities = [1,2,4,8] # rewards
        
        self.num_servers = 10 # total number of servers
        self.free_prob = 0.06 # probability of server becoming free
        
        self.states = self.state_space()
        self.actions = self.action_space()

    def state_space(self):
        """
        State space 

        Returns
        -------
        states : list of tuples (priority, num_free_servers)
            states consisting of priority and number of free servers

        """
        priority = np.array(self.priorities)
        num_free_servers = np.array(range(self.num_servers + 1))
        
        Priority, Num_free_servers = np.meshgrid(priority, num_free_servers)
        
        states = list(zip(Priority.flatten(), Num_free_servers.flatten()))

        return states
    
    def reset(self):
        """
        Choose state with initial priority and number of free servers

        Returns
        -------
        init_state : tuple (priority, num_free_servers)
            random initial priority and number of free servers

        """

        init_state = (np.random.choice(self.priorities), self.num_servers)
        return init_state
    
        
    def action_space(self):
        """
        Specify action space for queuing task
        
        Returns
        -------
        actions : ndarray
            possible actions of queuing task (accept: 1, reject: 0)

        """
        
        actions = np.array([0,1])
        return actions

    def step(self, state, action):
        """
        Receive reward from priority for given state and action

        Parameters
        ----------
        state : tuple
            priority and number of free servers.
        action : int
            accept or reject

        Returns
        -------
        next_state : tuple
            new priority and number of free servers.
        reward : int
            priority
        """
        
        priority = state[0]
        num_free_servers = state[1]
        
        if  num_free_servers == 0:
            reward = 0
        else:
            if action == 1: #reject
               reward = 0 
            elif action == 0: # accept
                reward = priority
                num_free_servers = num_free_servers - 1
        
        priority = np.random.choice(self.priorities)
        num_free_servers = self._free_servers(num_free_servers)
        next_state = (priority, num_free_servers)
        
        return next_state, reward
  
    def _free_servers(self, num_free_servers):
        """
        Servers randomly become free with probability self.free_prob

        Parameters
        ----------
        num_free_servers : int
            current number of free servers

        Returns
        -------
        free_servers : int
            next number of free servers

        """
        free_servers = num_free_servers
        # remark: this for loop can be parallelized for better performance
        for count in range(0, self.num_servers - num_free_servers):
            add_server = np.random.choice([0,1],p=[1-self.free_prob,
                                                          self.free_prob])
            free_servers += add_server
        
        return free_servers
  
