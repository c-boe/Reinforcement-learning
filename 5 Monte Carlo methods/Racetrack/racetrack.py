"""Implementation ofRacetrack environment from Exercise 5.12 in Sutton and 
Barto's "Reinforcement learning"
"""
import numpy as np
from PIL import Image
import cv2
import itertools
import os

class Racetrack():
    """Racetrack environment from Exercise 5.12 in Sutton and Barto's "Re-
    inforcement learning"
    """
    def __init__(self, Track_img):
        
        self.Track_img= Track_img #Ractrack
        self.max_vel = 5 # max velocity
        
        self.acc_x = [-1, 0, 1] # accelerations
        self.acc_y = [-1, 0, 1]
        self.cart_prod = list(itertools.product(self.acc_y, self.acc_x))
        self.nr_actions = len(self.acc_x)*len(self.acc_y)

    def reset(self):
        """
        Set to state to initial state of agent on starting line with zero 
        velocity

        Returns
        -------
        init_state : list of tuples
             Initial state consists of initial position at starting line and 
             zero initial velocity of agent
        """

        init_position = (0, np.random.choice(self.start_line_x))
        init_velocity = (0, 0)
        
        init_state = [init_position, init_velocity]
        
        return init_state

    def step(self, state, action):
        """
        Take step in racetrack environment by accelerating to subsequent 
        velocity and

        Parameters
        ----------
        state : list of tuples
             State consists of position and velocity of agent
        action : int
            action which agent takes

        Returns
        -------
        new_state: list of tuples
             New state consists of updated position and velocity of agent after
             taking action
        reward: int
            reward for action taken by agent
        done: bool
            episode terminated or not
        final_state: list of tuples
            state of agent after termination

        """
        done = False
        final_state = []
        
        # update velocity for given action
        acc = self.accelerate(action)
        
        velocity = np.array(state[1]) + acc
        
        velocity = (velocity > 0)*velocity
        velocity = (velocity <= self.max_vel)*velocity
        
        if (tuple(velocity) == (0,0)):
            velocity = np.array(state[1])
        
        velocity = tuple(velocity)
        
        # update position from new velocity
        position = tuple(np.array(state[0]) + velocity)
        
        # update state
        new_state = [position, velocity]

        # Determine reward
        if (position[0] in self.finish_line_y and
            position[1] >= self.finish_line_x[0]):
            reward = 0
            final_state = new_state
            done = True
        elif (position not in self.init_positions and 
              position not in self.racetrack_positions and
              position not in self.terminal_positions):
            reward = -1 
            new_state = self.reset()
        else:
            reward = -1

        return new_state, reward, done, final_state        

    def _import_track(self):
        """
        Import racetrack from image
        """
        path = os.getcwd() + "\\Tracks\\" + self.Track_img
        Track = cv2.imread(path,0)
        
        return Track
    
    def state_space(self):
        """
        Define state space by given racetrack

        Returns
        -------
        states : list of tuples
            States consist of positions and velocity of agent

        """
        
        Track = self._import_track()
        
        self.start_line_y = np.where(Track == 127)[0]
        self.start_line_x = np.where(Track == 127)[1]
        self.finish_line_y = np.where(Track == 195)[0]  
        self.finish_line_x = np.where(Track == 195)[1]      

        # positions
        nr_y, nr_x  = Track.shape
        self.x_dim = nr_x
        self.y_dim = nr_y
        x = np.linspace(0, nr_x - 1, nr_x)
        y = np.linspace(0, nr_y - 1, nr_y)
        Y, X = np.meshgrid(y, x)
        x_vec = X.flatten().astype(int)
        y_vec = Y.flatten().astype(int)
        
        positions = list(zip(y_vec, x_vec))

        # velocities
        vel_x = np.linspace(0, self.max_vel, self.max_vel + 1)
        vel_y = np.linspace(0, self.max_vel, self.max_vel + 1)
        
        Vel_y, Vel_x = np.meshgrid(vel_y, vel_x)
        vel_x_vec = Vel_x.flatten().astype(int)
        vel_y_vec = Vel_y.flatten().astype(int)

        velocities = list(zip(vel_y_vec, vel_x_vec))

        states = [positions, velocities]
        
        # racetrack
        self.gen_racetrack_grid(Track)

        return states

    def gen_racetrack_grid(self, Track):
        '''
        Generate lists with tuples of initial, final and racetrack psoitions

        Parameters
        ----------
        Track : ndarray
           array representing the race track

        Returns
        -------
        None.

        '''
        self.init_positions = list(zip(self.start_line_y, self.start_line_x))
        self.terminal_positions = list(zip(self.finish_line_y, 
                                           self.finish_line_x))
        
        x = np.where(Track == 0)[1]
        y = np.where(Track == 0)[0]
        self.racetrack_positions = list(zip(y, x))

        return

    def accelerate(self, action):
        """
        Calculate acceleration according to given action
        
        action : int
            action of agent 

        Returns
        -------
        acc : ndarray
            acceleration of agent consisting of x and y component

        """
        
        
        
        acc = self.cart_prod[action]
        acc = np.array(acc)
        
        return acc

    def plot_racetrack(self, current_state):
        """
        Plot racetrack with current position of agent

        Parameters
        ----------
        current_state : tuple
            current position and velocity of agent

        Returns
        -------
        None.

        """
        Track = self._import_track()
        
        Track[current_state[0]] = 255

        img = Image.fromarray(Track)
        img_res = img.resize((self.x_dim*30, self.y_dim*30), Image.NEAREST)

        Track_res = np.array(img_res)

        cv2.imshow("", Track_res)
        cv2.waitKey(200)
        return



