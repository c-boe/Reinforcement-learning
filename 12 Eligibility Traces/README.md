# Chapter 12: Eligibility traces

Implementation of example 12.2 (Mountain Car Task) in section 12.7 from Sutton and Barto's [Reinforcement learning](http://incompleteideas.net/book/the-book.html).


## Mountain Car Task

The environment is solved by with [Sarsa(lambda) with binary features and linear function approximation](https://github.com/c-boe/Reinforcement-learning/blob/main/12%20Eligibility%20Traces/Mountain%20Car/SARSA_w_FA.py). The mountain car environment and the feature construction by tile coding are identical to the implementation of [Mountain car task with Episodic Sarsa with function approximation](https://github.com/c-boe/Reinforcement-learning/tree/main/10%20On-policy%20control%20with%20approximation/Mountain%20Car). The Sarsa(lambda) function was added to [Sarsa_w_FA.py](https://github.com/c-boe/Reinforcement-learning/blob/main/12%20Eligibility%20Traces/Mountain%20Car/SARSA_w_FA.py). The major difference to episodic Semi-gradient Sarsa is the utilization of eligibility traces.
