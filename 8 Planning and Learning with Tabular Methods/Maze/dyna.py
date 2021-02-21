"""
Implementation of the  Dyna algorithm for maze environment from section 8.3 of 
Sutton and Barto's "Reinforcement Learning"
"""

import numpy as np

class Dyna():
    """
    Dyna algorithm class for solving Maze environment
    """
    def __init__(self, maze, PLAN_STEPS, CHANGE_STEP, ALPHA = 0.1, GAMMA = 0.95, 
                 EPSILON = 0.1):
        """
        Parameters
        ----------
        maze : 
            Maze environment.
        NUM_STEPS : int
            Total number of steps taken in environment.
        PLAN_STEPS : int
            Max. number of steps for Q-planning
        CHANGE_STEP : int
            Step at which Maze environment is changed
        ALPHA : Q-learning stepsize
            DESCRIPTION.
        GAMMA : float
            discount factor.
        EPSILON : float
            eps-greedy action selection parameter.
        """
        self.maze = maze
        
        self.PLAN_STEPS = PLAN_STEPS
        self.CHANGE_STEP = CHANGE_STEP
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA 
        self.EPSILON = EPSILON 
        
        self.states = maze.state_space()
        self.actions = maze.action_space()

    def DynaQ(self, NUM_STEPS = 1000):
        """
        Dyna Q algorithm (p.164 of Sutton and Barto's "Reinforcement Learning")
        for maze environment
                          
        Parameters
        ----------
        NUM_STEPS : int
            Total number of steps taken in environment.
        Returns
        -------
        cum_reward : ndarray 
            cumulative reward at each step.
    
        """
        Q = self.init_Q_values()
        model = self.init_model()
        states_visited, action_idx_visited = self.init_state_action_track()
        
        cum_reward = np.zeros(NUM_STEPS+1)
        state = self.maze.reset()
        action, action_idx = self.choose_action(Q, state)
        
        done = False 
        for step in range(NUM_STEPS):
    
            if step == self.CHANGE_STEP:
                self.maze.wall = self.maze.gen_wall(position = 2)
            
            next_state, reward, done = self.maze.step(state, action)
    
            Q = self.update_Q_values(Q, state, action_idx, next_state, reward)
            model = self.update_model(model, state, action_idx, reward, next_state)
            states_visited, action_idx_visited = self.update_state_action_track(
                state, action_idx, states_visited, action_idx_visited)
            
            Q = self.Q_planning(model, states_visited, action_idx_visited, Q)
    
            state = next_state
            action, action_idx = self.choose_action(Q, state)
            cum_reward[step + 1] = cum_reward[step] + reward
    
            #maze.render(state)
            if done:
                state = self.maze.reset()
                action, action_idx = self.choose_action(Q, state)
                done = False 
                
        return cum_reward

    def Q_planning(self, model, states_visited, action_idx_visited, Q):
        """
        Q-planning algorithm
    
        Parameters
        ----------
        model : dict
            model of environment 
            model[(state, action_idx)] = [reward, next_state, already visited?]
        states_visited : list
            list with states visited so far
        action_idx_visited : dict
            actions taken in visited states
        Q : dict
            Q values Q[states][actions].
        Returns
        -------
        Q : dict
            updated Q values Q[states][actions].
    
        """
        for pl_step in range(self.PLAN_STEPS):
            # choose previously visited state
            rand_state_idx = np.random.choice(len(states_visited))
            state = states_visited[rand_state_idx]
            
            # choose previously taken action in state
            rand_action_idx = np.random.choice(len(action_idx_visited[state]))
            action_idx = list(action_idx_visited[state])[rand_action_idx]
            
            reward, next_state,_ = model[(state, action_idx)]
            Q = self.update_Q_values(Q, state, action_idx, next_state, reward)
            
        return Q

    def init_Q_values(self):
        """
        Initialize Q values 
    
        Returns
        -------
        Q : dict
            Q values
            Q[states][actions].
    
        """
        Q = {}
        for state in self.states:
            Q[state] = np.zeros(len(self.actions))
        
        return Q
    
    def update_Q_values(self, Q, state, action_idx, next_state, reward):
        """
        Q learning update
        """
        Q_max = np.max(Q[next_state])
        Q[state][action_idx] += self.ALPHA*(reward + self.GAMMA*Q_max 
                                            - Q[state][action_idx])
        
        return Q
        
    def init_model(self):
        """
        Initialize model of environment
    
        Returns
        -------
        model : dict
            model of environment 
            model[(state, action_idx)] = [reward, next_state, already visited?]
    
        """
        model = {}
        for state in self.states:
            for action_idx in range(len(self.actions)):
                model[(state, action_idx)] = [0, (-1,-1), False]
        
        return model
        
    def update_model(self, model, state, action_idx, reward, next_state):
        """
        update model from observed reward, state and action
        """
        model[(state, action_idx)] = [reward, next_state, True]
        
        return model
    
    def choose_action(self, Q, state):
        """
        epsilon-greedy action selection
        """
        if ((Q[state] == np.zeros(len(self.actions))).all() 
            or np.random.rand() < self.EPSILON):
            action_idx = np.random.choice(len(self.actions))  
        else :
            action_idx = np.argmax(Q[state])
    
        action = self.actions[action_idx]
        return action, action_idx
    
    def init_state_action_track(self):
        """
        Initialize state and action tracker
    
        Returns
        -------
        states_visited : list
            list with states visited so far
        action_idx_visited : dict
            actions taken in visited states
            
        """
        states_visited = [] # write states which were visited to list    
        action_idx_visited = {}
        for state in self.states:
            action_idx_visited[state] = set()
            
        return states_visited, action_idx_visited
    
    def update_state_action_track(self, state, action_idx, states_visited, 
                                  action_idx_visited):
        """
        Update state and action tracker with new states and actions
        """
        if state not in states_visited:
            states_visited.append(state)
        if action_idx not in action_idx_visited[state]:
            action_idx_visited[state].add(action_idx)
            
        return states_visited, action_idx_visited
    
#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from maze import ShortcutMaze
    
    NUM_STEPS = 6000
    CHANGE_STEP = 3000
    
    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 0.1
    
    PLAN_STEPS = 100
    
    smaze = ShortcutMaze()
    dyna = Dyna(smaze, PLAN_STEPS, CHANGE_STEP, ALPHA, GAMMA, EPSILON)
    cum_reward = dyna.DynaQ(NUM_STEPS)
    
    plt.figure()
    plt.plot(cum_reward)
