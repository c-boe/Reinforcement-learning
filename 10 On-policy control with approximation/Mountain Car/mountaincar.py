"""
Implementation of mountaincar environment for example 10.1 of "Reinforcement
learning" by Sutton and Barto
"""

import numpy as np
import matplotlib.pyplot as plt


class MountainCar():
    """Mountain car environment from example 10.1 of Sutton and Barto's
    "Reinforcement Learning" """
    def __init__(self):
        
        self.x_bound_l = -1.2
        self.x_bound_u = 0.5
       
        self.v_bound_l = -0.07
        self.v_bound_u = 0.07
        
        self.x_init_l = -0.6
        self.x_init_u = -0.4
       
    def reset(self):
        """
        Initialize position and velocity of mountain car

        Returns
        -------
        state : list [position, velocity]
            state at beginning of every episode   

        """
        x = np.random.uniform(low=self.x_init_l, high=self.x_init_u, size=1)
        v = 0
        state = [x, v]
        return state
    
    def step(self, state, action):
        """
        Take action in current state

        Parameters
        ----------
        state : list [current position, current velocity]
            state before taking action
        action : int
            action taken by agent

        Returns
        -------
        next_state : list [next position, next velocity]
            state after taking action
        reward : int
            negative reward for every step
        done : bool
            episode finished or not.
        final_state : list [final position, final  velocity]
            state at the end of the episode
        """
        done = False
        final_state = []
        
        x, v = state
    
        throttle = self.action(action)
    
        next_v = self.__bound_v(v + 0.001*throttle - 0.0025*np.cos(3*x))
        next_x = self.__bound_x(x + next_v)
        
        if next_x == self.x_bound_l:
            next_v = 0 
            
        next_state = np.array([next_x, next_v])  
            
        if next_x == self.x_bound_u: 
            done = True
            final_state = [next_x, next_v]
      
        reward = -1
        
        return next_state, reward, done, final_state
    
    
    def __bound_v(self, v):
        """
        Apply velocity boundaries
        """
        if v < self.v_bound_l:
            v = self.v_bound_l
        elif v > self.v_bound_u:
            v = self.v_bound_u
            
        return v
            
    def __bound_x(self, x):
        """
        Apply positional boundaries
        """
        if x < self.x_bound_l:
            x = self.x_bound_l
        elif x > self.x_bound_u:
            x = self.x_bound_u
            
        return x
    
    def action(self, action):
        """
        

        Parameters
        ----------
        action : int
            action taken by agent

        Returns
        -------
        throttle : int
            reverse, forward or zero throttle 

        """
        if action == 0:
            throttle = -1 #reverse
        elif action == 1:
            throttle = 0
        elif action == 2:
            throttle = 1 # foreward
            
        return throttle 
    
    def render(self, x):
        """
        Plot mountain car position x at every step of episode 

        Parameters
        ----------
        x : int
            x coordinate of mountain car

        Returns
        -------
        None.

        """
        x = np.array([x])
        y = np.cos(2*np.pi*(x/2 + 0.75))
        
        x_mountain = np.linspace(self.x_bound_l, self.x_bound_u,100)
        y_mountain = np.cos(2*np.pi*(x_mountain/2 + 0.75))
        
        plt.figure("Mountain car")
        plt.clf()
        plt.plot(x_mountain, y_mountain)
        plt.title("Mountain car")
        if len(x) == 1:
            plt.plot(x, y,'o')
            plt.xlim((self.x_bound_l, self.x_bound_u))
            plt.ylim((-1, 1))
        
            plt.pause(0.01)    
        else:
            
            nr_steps = len(x)
    
            for step in range(nr_steps):
    
                plt.clf()
                plt.plot(x[step], y[step],'o')
                plt.xlim((self.x_bound_l, self.x_bound_u))
                plt.ylim((-1, 1))
            
                plt.pause(0.1)

    def plot_step_per_ep(self, episode, steps):
        """
        Plot number of steps per episode of mountain car as function of episode

        Parameters
        ----------
        episode : int
            current episode
        steps : int
            number of steps until episode is done

        Returns
        -------
        None.

        """
        plt.figure("Steps per episode")
        plt.plot(episode, steps,'o')
        plt.yscale("log")
        plt.pause(0.001)   
        plt.xlabel("Episode")
        plt.ylabel("Steps per episode (log scale)")        

    
