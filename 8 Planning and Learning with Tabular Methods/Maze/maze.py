"""
Implementation of the Maze environment(s) of section 8.3 of Sutton and Barto's 
"Reinforcement Learning"
"""

import cv2
from PIL import Image
import numpy as np
import itertools

class Maze():
    """Maze without wall"""
    def __init__(self, wall_position= 1):
        
        self.x_dim = 9
        self.y_dim = 6

        self.terminal_state = (self.x_dim-1, self.y_dim-1)
        self.initial_state = (3,0)

    def reset(self):
        return self.initial_state
        
    def step(self, state, action):
        """
        Take action in state by going up, down, left or right
        """
        done = False
        reward = 0
        
        next_state = np.array(state) + action
        
        if (next_state == np.array(self.terminal_state)).all():
            reward = 1
            done = True
        else:
            next_state = self._check_border(next_state)
            next_state = self._check_wall(next_state, action)
            
        return tuple(next_state), reward, done
    
    def state_space(self):
        """
        Generate state space

        Returns
        -------
        states : 
            
        """
        x = range(self.x_dim)
        y = range(self.y_dim)
        states = list(itertools.product(x,y))
        return states
    
    def action_space(self):
        """
        Generate action space

        Returns
        -------
        actions : ndarray
            actions up, down, left, right.

        """
        actions = np.array([(-1,0),(1,0),(0,1),(0,-1)])
        return actions
    
    def render(self, state):
        """
        Render maze

        """
        x = range(self.x_dim)
        y = range(self.y_dim)
        
        Y, X = np.meshgrid(np.array(x), np.array(y))
        
        Z = np.zeros(X.shape)#.astype(np.uint8)
        Z[state[1],state[0]] = 1

        for indices in self.wall:
            Z[indices[1],indices[0]] =0.5
            
        Z[self.terminal_state[1], self.terminal_state[0]] = 0.75
            
        img = Image.fromarray(Z)
        img = img.resize((Z.shape[1]*100, Z.shape[0]*100),Image.NEAREST)
        
        img = np.array(img)
        
        cv2.imshow("Maze", np.flipud(img ))
        cv2.waitKey(1)
        
        return

    def _check_border(self,state):
        """
        Check if agent hits border
        """
        if state[0] > self.x_dim-1:
            state[0] = self.x_dim-1
        if state[0] < 0:
            state[0] = 0
        if state[1] > self.y_dim-1:
            state[1] = self.y_dim-1
        if state[1] < 0:
            state[1] = 0   

        return state

    def _check_wall(self, state, action):
        """
        Reverse action in case the agent hits the wall
        """
        if state.tolist() in self.wall.tolist():
            state = state - action
        
        return state
    
class BlockingMaze(Maze):
    """Blocking Maze from example 8.2"""
    def __init__(self):
        super().__init__()
        self.wall = self.gen_wall() 
        
    def gen_wall(self, position = 1):
        """
        Generate a wall in the maze

        """
        if position == 1:
            x = range(0, self.x_dim-1)
            y = range(2,3)
        elif position !=1:
            x = range(1, self.x_dim)
            y = range(2,3)
            
        wall = np.array(list(itertools.product(x,y)))
        
        return wall
    
class ShortcutMaze(Maze):
    """Shortcut Maze from example 8.3"""
    def __init__(self):
        super().__init__()
        self.wall = self.gen_wall() 
        
    def gen_wall(self, position = 1):
        """
        Generate a wall in the maze

        """
        if position == 1:
            x = range(1, self.x_dim)
            y = range(2,3)
        elif position !=1:
            x = range(1, self.x_dim-1)
            y = range(2,3)
            
        wall = np.array(list(itertools.product(x,y)))
        
        return wall
        
#%% TTest
if __name__ == "__main__":

    env = ShortcutMaze()

    state = env.reset()
    env.render(state)
    done = False
    while not done:
        
        action_ind = np.random.choice(len(env.action_space()))
        action = env.action_space()[action_ind]
        next_state, _, done = env.step(state, action)
        env.render(next_state)
        state = next_state
