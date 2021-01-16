# Chapter 10: On-policy Control with Approximation

Implementation of example 10.1 (Mountain Car Task) in section 10.1 and example 10.2 (An Access-Control Queuing Task) in section 10.3 from Sutton and Barto's [Reinforcement learning](http://incompleteideas.net/book/the-book.html).


## Mountain Car Task

The [mountain car environment](https://github.com/c-boe/Reinforcement-learning/blob/main/10%20On-policy%20control%20with%20approximation/Mountain%20Car/mountaincar.py) is solved with [episodic Semi-gradient Sarsa](https://github.com/c-boe/Reinforcement-learning/blob/main/10%20On-policy%20control%20with%20approximation/Mountain%20Car/episodic_SARSA_w_FA.py). 
The feature construction for the linear function approximation of the action value function is done using [tile coding](https://github.com/c-boe/Reinforcement-learning/blob/main/10%20On-policy%20control%20with%20approximation/Mountain%20Car/tilecoding.py) (section 9.5.4).

## Access-Control Queuing Task

The [Access-Control environment](https://github.com/c-boe/Reinforcement-learning/blob/main/10%20On-policy%20control%20with%20approximation/Access%20control/accesscontrol.py) is solved by utilizing [differential semi-gradient Sarsa](https://github.com/c-boe/Reinforcement-learning/blob/main/10%20On-policy%20control%20with%20approximation/Access%20control/diff_sg_Sarsa.py) 