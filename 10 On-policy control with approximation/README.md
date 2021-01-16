# Chapter 10: On-policy Control with Approximation

Implementation of Example 10.1 (Mountain Car Task) and 10.2 (An Access-Control Queuing Task) from Sutton and Barto's [Reinforcement learning](http://incompleteideas.net/book/the-book.html).


## Mountain Car Task (Section 10.1)

The mountain car environment (mountaincar.py) is solved with Episodic Semi-gradient Sarsa (episodic_Sarsa_w_FA.py). 
The feature construction (tilecoding.py) for the linear function approximation of the action value function is done using tile coding (section 9.5.4).

## Access-Control Queuing Task (Section 10.3)

The Access-Control environment (accesscontrol.py) is solved by utilizing Differential semi-gradient Sarsa (diff_sg_Sarsa.py) 