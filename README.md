# Reinforcement learning

This repository contains implementations of some exercises/examples of Sutton and Barto's [Reinforcement learning](http://incompleteideas.net/book/the-book.html).

## General information

The codes are written in Python 3.8 and only utilize standard python packages like e.g. numpy or matplotlib. 

Each folder in this repository corresponds to a chapter in the book [Reinforcement learning](http://incompleteideas.net/book/the-book.html) by Sutton and Barto. Each exercise/example consists of a .py file containing the environment of the specific problem and a second .py file in which the algorithm is implemented. 

The algorithm-.py files can be run and tested directly without modifying the code. To do some experiments the parameteres at the end of the algorithm-.py files can be changed.

Environments from OpenAI gym were not utilized but it was tried to keep the structure (env.step(), env.render(), etc.) of the environments close to gym.

## Currently implemented

* [Stationary and nonstationary multi-armed bandit with a simple bandit algorithm and a gradient bandit algorithm](https://github.com/c-boe/Reinforcement-learning/tree/main/2%20Multi-armed%20bandit)
* [Gambler's problem with value iteration](https://github.com/c-boe/Reinforcement-learning/tree/main/4%20Dynamic%20programming/Jacks%20car%20rental)
* [Jack's car rental with policy iteration](https://github.com/c-boe/Reinforcement-learning/tree/main/4%20Dynamic%20programming/Jacks%20car%20rental)
* [Black Jack with Monte-Carlo control with exploring starts](https://github.com/c-boe/Reinforcement-learning/tree/main/5%20Monte%20Carlo%20methods/Black%20Jack)
* [Racetrack with off-policy Monte-Carlo control](https://github.com/c-boe/Reinforcement-learning/tree/main/5%20Monte%20Carlo%20methods/Racetrack)
* [Windy gridworld with Sarsa and cliffwalking with Q-learning](https://github.com/c-boe/Reinforcement-learning/tree/main/6%20Temporal%20difference%20learning)
* [Mountain car task with Episodic Sarsa with function approximation](https://github.com/c-boe/Reinforcement-learning/tree/main/10%20On-policy%20control%20with%20approximation/Mountain%20Car)
* [Access-control queuing task with differential semi-gradient Sarsa](https://github.com/c-boe/Reinforcement-learning/tree/main/10%20On-policy%20control%20with%20approximation/Access%20control)

## Acknowledgement

Thanks to Richard S. Sutton and Andrew G. Barto for making their book freely available.