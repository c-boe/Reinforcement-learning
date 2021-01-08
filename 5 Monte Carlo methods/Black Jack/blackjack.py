"""
Implementation of Black jack environment from example 5.1 (Monte Carlo 
prediction) and example 5.3 (Monte Carlo control with exploring starts) from 
Sutton and Barto's "Reinforcement Learning"
"""

import numpy as np

class BlackJack():
    
    def __init__(self):      
        self.nr_actions = 2 # stick or hit
        self.policy_dealer = 17 # dealer sticks at
        
    def reset(self):
        """
        Generate initial state

        Returns
        -------
        initial_state : tuple (int, int, bool)
            random initial state consisting of dealer card, player card, and
            whether or not the player has a usable ace

        """
        dealer_card = np.random.randint(1,11)
        player_card = np.random.randint(12, 22)
        usable_ace = np.random.randint(1,14)<3 # two chances for picking ace 
        
        initial_state = [dealer_card, player_card, usable_ace]
        
        return initial_state
        
    def state_space(self):
        """
        Generate state space

        Returns
        -------
        states : list of tuples
            List with all possible combinations of dealer card, player card, 
            and whether or not the player has a usable ace

        """
        
        state_dealer = range(1,11)
        state_player = range(12, 22)

        State_dealer, State_player = np.meshgrid(state_dealer, state_player)

        State_dealer_vec = np.concatenate([State_dealer.flatten(),
                                           State_dealer.flatten()])
        State_player_vec = np.concatenate([State_player.flatten(),
                                           State_player.flatten()])
        State_usable_ace_vec = 100*[False] + 100*[True]
        
        states = list(zip(State_dealer_vec, State_player_vec 
                          ,State_usable_ace_vec))
        
        return states

    def action_space(self):
        """ hit (0) or stick (1) """
        return range(self.nr_actions)

    def step(self, current_state, action):
        """
        Given the current state (dealer cards, player cards, usable or no
        usable ace) the player takes an action (hit or stick) and receives 
        reward

        Parameters
        ----------
        current_state : tuple (int, int, bool)
            state consisting of dealer card, player card, and whether or not 
            the player has a usable ace
        action : int
            possible action: 0 (hit) or 1 (stick)

        Returns
        -------
        next_state :  tuple (int, int, bool)
            state after action consisting of dealer card, player card, and 
            whether or not the player has a usable ace
        reward : int
            reward for win (1), draw (0) or loss (-1)
        done : bool
            Episode finished (True) or not (False)
        final_state : tuple (int, int, bool)
            state after finished episode consisting of dealer card, 
            player card, and whether or not the player has a usable ace

        """
        reward = 0
        done = False
        final_state = []
        state = np.array([current_state[0], current_state[1]])
        usable_ace = current_state[2]
        

        if action == 0: # hit
            # player
            pick_card = min(np.random.randint(1,14), 10)
            state[1] += pick_card
        
            if state[1] > 21:
                if usable_ace:
                    state[1] -= 10
                    usable_ace = False
                else:    
                    reward = -1
                    done = True

        elif action == 1: # stick
            done = True
            #dealer
            while state[0] < self.policy_dealer:
                pick_card = min(np.random.randint(1,14), 10)
                if pick_card == 1 and state[0] <= 10:
                    pick_card = 11
                    
                state[0] += pick_card

            if state[0] > 21:
                reward = 1
            else:
                if state[0] == state[1]:
                    reward = 0
                elif state[0] < state[1]:
                    reward = 1
                elif state[0] > state[1]:
                    reward = -1
                
        next_state = [state[0],state[1], usable_ace]
        
        if done:
            final_state = next_state

        return next_state, reward, done, final_state

