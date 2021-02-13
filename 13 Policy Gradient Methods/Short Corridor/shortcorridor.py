"""
Short corridor with switched actions (Example 13.1) of Sutton and Barto's
"Reinforcement learning"
"""

import numpy as np


class ShortCorridor():
    """Short corridor with switched actions"""
    
    def __init__(self):
        self.num_states = 4
        
        self.states = self.state_space()
        self.actions = self.action_space()
        
        self.initial_state = self.states[0]
        self.terminal_state = self.states[-1]
        
        self.switch_state = self.states[1]
        
        
    def reset(self):
        """Initialize state"""
        state = self.initial_state
        return state
    
    def step(self, state, action):
        """Agent takes action in given state"""
        done = False
        
        if state == self.switch_state:# invert moving direction
            action = -action
        
        next_state = state + action
        
        if next_state < self.initial_state:
            next_state = self.initial_state
            
        if next_state == self.terminal_state:
            done = True
            
        reward = -1
            
        return next_state, reward, done
    
    def state_space(self):
        """Define state space"""
        states = np.array([state for state in range(self.num_states)])
        return states
        
    def action_space(self):
        """
        Define action space

        Returns
        -------
        actions : ndarray
            actions in short corridor: -1: go left; 1: go right.

        """
        actions = np.array([-1, 1])
        return actions
        
#%%      
class FeatureVectors(ShortCorridor):
    """Feature vector for short corridor with switched actions."""
    def __init__(self):
        super().__init__()
        self.states = self.state_space()
        self.actions = self.action_space()
        
        self.x_vec = self.feat_vec()
        
    def feat_vec(self):
        """Define feature vectors for state action pairs (Important: The agent
        cannot distinguish between the states from the given observations)"""
        x_vec = {}
        for state in self.states:
            x_vec[(state, self.actions[0])] = np.array([0,1])
            x_vec[(state, self.actions[1])] = np.array([1,0])
            
        return x_vec
    
#%%
if __name__ == "__main__":
    
    SC = ShortCorridor()
    FV = FeatureVectors()
    
    print(FV.x_vec[(0,1)])
    
    
    #%% Test
    eps= 0.82 # highest prob for eps = 0.82
    NUM_EPISODES = 1000
     
    avg_steps = 0
    for episode in range(NUM_EPISODES):
        steps = 0
        state = SC.reset()
        done = False
        while not done:
            steps += 1
            action = np.random.choice(SC.action_space(), p=[eps/2, 1-eps/2])
            state, reward, done = SC.step(state, action)

        #print(steps)
        avg_steps = avg_steps + 1/(episode+1)*(steps - avg_steps)
    print(avg_steps)