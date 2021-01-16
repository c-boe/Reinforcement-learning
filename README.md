# Reinforcement learning

This repository contains implementations of some exercises/examples of Sutton and Barto's [Reinforcement learning](http://incompleteideas.net/book/the-book.html).

## General information

The codes are written in Python 3.8 and only utilize standard python packages like e.g. numpy or matplotlib. 

Each folder in this repository corresponds to a chapter in the book [Reinforcement learning](http://incompleteideas.net/book/the-book.html) by Sutton and Barto. Each exercise/example consists of .py file containing the environment of the specific problem and a second .py file in which the algorithm is implemented. 

The algorithm-.py files can be run and tested directly without modifying the code. To do some experiments the parameteres at the end of the algorithm-.py files can be changed.

Environments from OpenAI gym were not utilized but it was tried to keep the structure (env.step(), env.render(), etc.) of the environments close to gym.

## Acknowledgement

Thanks to Richard S. Sutton and Andrew G. Barto for making their book freely available.