"""
Implementation of gridworld environments (6.5 and 6.6) from Chapter 6 of 
Sutton and Barto's "Reinforcement Learning"
"""
import cv2
import numpy as np
from PIL import Image


#%%
class gridworld():
    """ Gridworld environment"""
    def __init__(self, x_dim, y_dim, kings_moves):
        
        # grid dimensions
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        self.kings_moves = kings_moves # bool; use kings moves or not
        self.states = self.state_space()
        
    def reset(self):
        '''
        Set agent to initial state

        Returns
        -------
        state : tuple
            initial coordinates.

        '''
        state = self.initial_state
        return state
    
    
    def state_space(self):
        '''
        Generate grid and write coordinates into list of tuples

        Returns
        -------
        states : list
            list of tuples representing the coordinates of the agent.

        '''
        
        nr_y = self.y_dim
        nr_x = self.x_dim
        
        x = np.linspace(0, nr_x - 1, nr_x)
        y = np.linspace(0, nr_y - 1, nr_y)
        X, Y = np.meshgrid(x, y)
        x_vec = X.flatten()
        y_vec = Y.flatten()
        
        states = list(zip(x_vec,y_vec))

        return states
    
    def action_space(self):
        """
        Generate action space
        """
        if self.kings_moves:
            actions = range(0,9)
        else:
            actions = range(0,4)
        return actions
    
    def move(self, action):
        '''
        Move 

        Parameters
        ----------
        action : int
            Choose action.

        Returns
        -------
        step : tuple
            Coordinate step.

        '''
        if action == 0:
            step = np.array((0, 1)) 
        elif action == 1:
            step = np.array((1, 0)) 
        elif action == 2:
            step = np.array((0, -1))
        elif action == 3:
            step = np.array((-1, 0))               
        
        elif action == 4: # kings moves
            step = np.array((1, 1)) 
        elif action == 5:
            step = np.array((-1, -1)) 
        elif action == 6:
            step = np.array((1, -1)) 
        elif action == 7:
            step = np.array((-1, 1))             
        elif action == 8:
            step = np.array((0, 0)) 
    
        return step


    def grid_boundary(self, state):
        '''
        Boundaries of grid

        Parameters
        ----------
        state : tuple()
            position of agent on grid.

        Returns
        -------
        state : tuple()
            position of agent on grid.

        '''
        if state[0] < 0:
            state[0] = 0 
        if state[0] > self.x_dim - 1:
            state[0] = self.x_dim - 1
        if state[1] < 0:
            state[1] = 0 
        if state[1] > self.y_dim - 1:
            state[1] = self.y_dim - 1    
            
        return state
                    

class CliffWalking(gridworld):
    '''
    Defines environment according to the Example 6.6: Cliff Walking of 
    "Reinforcement learning" by Sutton
    '''
    
    def __init__(self, x_dim, y_dim, nr_actions):
        
        super().__init__(x_dim, y_dim, nr_actions)
        
        self.initial_state = (0,0)
        self.terminal_state = (x_dim - 1, 0)
        
    
    def step(self, current_state, action):
        '''
        Take action in current state and receive reward

        Parameters
        ----------
        current_state : tuple
            x-y coordinates in gridworld before taking action.
        action : int
            move up, right, down or left.

        Returns
        -------
        next_state: tuple
            x-y coordinates in gridworl after taking action.
        reward : int
            reward after taking action in current state
        done : bool
            Episode done.
        final : bool
            Terminal state reached.
        '''
        next_state = current_state + self.move(action)

        next_state = self.grid_boundary(next_state)

        # cliff
        cliff = False
        if next_state[1] == 0 and not next_state[0] == 0:
            cliff = True
        if next_state[1] == 0 and not next_state[0] == self.x_dim - 1:
            cliff = True
            
        # reward
        done = False
        final_state = False
        if (next_state[0] == self.terminal_state[0] and 
            next_state[1] == self.terminal_state[1]):
            reward = -1
            done = True
            final_state = True
        elif cliff:
            reward = -100
            done = True
        else:
            reward = -1
        
        next_state = tuple(next_state)
        
        return next_state, reward, done, final_state
    
    def render(self, state):
        
        plot = np.zeros((self.y_dim, self.x_dim))
        plot[state[1]][state[0]] = 1

        img = Image.fromarray(plot)
        img = img.resize((300,150),Image.NEAREST)
            
        cv2.imshow("",np.array(img))
        cv2.waitKey(500)


class WindyGridworld(gridworld):
    '''
    Defines environment according to the Exercise 6.9 and 6.10: of 
    "Reinforcement learning" by Sutton and Barto
    '''
    def __init__(self, x_dim, y_dim, kings_moves, stochastic_wind,
                 initial_state, terminal_state):
        
        super().__init__(x_dim, y_dim, kings_moves)

        self.initial_state = initial_state
        self.terminal_state = terminal_state
    
        self.stochastic_wind = stochastic_wind # wind changes randomly
        
    def step(self, state, action):
        '''
        Take action in current state and receive reward

        Parameters
        ----------
        current_state : tuple (x,y)
            x-y coordinates in gridworld before taking action.
        action : int
            move up, right, down or left.

        Returns
        -------
        next_state: tuple (x,y)
            x-y coordinates in gridworl after taking action.
        reward : int
            reward after taking action in current state
        done : bool
            Episode done.
        final : bool
            Terminal state reached.
        '''
        next_state = state + self.move(action)
        next_state = self.grid_boundary(next_state)   

        next_state = next_state + self.wind(next_state)
        next_state = self.grid_boundary(next_state)
        
        done = False
        final_state = False
        if (next_state[0] == self.terminal_state[0] and
            next_state[1] == self.terminal_state[1]):
            reward = 0
            done = True
            final_state = True
        else:
            reward = -1

        return tuple(next_state), reward, done, final_state
        
    def wind(self, state):
        '''
        Defines step by wind 

        Parameters
        ----------
        state : tuple
            position on grid.

        Returns
        -------
        step : tuple
            Coordinate step.

        '''
        if self.stochastic_wind is False:
            add_step = np.array((0,0))
        elif self.stochastic_wind is True:
            add_step = np.random.randint(-1,1)
        
        
        if state[0] in [0,1,2]:
            step = np.array((0, 0)) + add_step
        elif state[0] in [3,4,5]:
            step = np.array((0, 1)) + add_step
        elif state[0] in [6,7]:  
            step = np.array((0, 2)) + add_step
        elif state[0] == 8: 
            step = np.array((0, 1)) + add_step
        elif state[0] >= 9: 
            step = np.array((0, 0)) + add_step
            
        return step
    
    def render(self, state):

        array = np.zeros((self.x_dim, self.y_dim))
        array[self.terminal_state] = 1
        array[(state)] = 1
    
        img = Image.fromarray(array)
        img = img.resize((50*self.x_dim,50*self.y_dim), Image.NEAREST)
    
        img = np.array(img)
        cv2.imshow("",img)
        cv2.waitKey(500)
        