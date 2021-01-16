"""
Implementation of Gamblers problem from Exercise 4.9 in Chapter 4 of
 Sutton and Barto's "Reinforcement Learning" 
"""

class GamblersProblem():
    '''
    Gamblers Problem MDP
    '''
    def __init__(self, num_states, terminal_states, p_h):
        
        self.num_states = num_states # int number of states
        self.terminal_states = terminal_states # tuple, shape (1,1)
        self.p_h = p_h # probability for head 
        
        
    def state_space(self):
        """ Generate states of MDP """
        states = range(1, self.num_states + 1)
        return states