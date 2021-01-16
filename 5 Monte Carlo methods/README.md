# Chapter 5: Monte-Carlo Methods

Implementation of Example 5.3 (Blackjack) in section 5.3 and Exercise 5.12 (Racetrack) in section 5.7 of Sutton and Barto's [Reinforcement learning](http://incompleteideas.net/book/the-book.html).


## Blackjack

The [Blackjack environment](https://github.com/c-boe/Reinforcement-learning/blob/main/5%20Monte%20Carlo%20methods/Black%20Jack/blackjack.py) is solved using [Monte-Carlo Control with exploring states](https://github.com/c-boe/Reinforcement-learning/blob/main/5%20Monte%20Carlo%20methods/Black%20Jack/Monte_Carlo_with_ES.py).

## Racetrack

The [racetrack environment](https://github.com/c-boe/Reinforcement-learning/blob/main/5%20Monte%20Carlo%20methods/Racetrack/racetrack.py) is solved using [off-policy Monte-Carlo control](https://github.com/c-boe/Reinforcement-learning/blob/main/5%20Monte%20Carlo%20methods/Racetrack/Off_policy_MC.py). 
Users can design their own track in Paint, add a .png to the [Tracks folder](https://github.com/c-boe/Reinforcement-learning/tree/main/5%20Monte%20Carlo%20methods/Racetrack/Tracks) and set the "Track_img"-parameter in [Off_policy_MC.py](https://github.com/c-boe/Reinforcement-learning/blob/main/5%20Monte%20Carlo%20methods/Racetrack/Off_policy_MC.py). The colors of the track, the starting and the goal line should be the same as for the example tracks ([Track1.png](https://github.com/c-boe/Reinforcement-learning/blob/main/5%20Monte%20Carlo%20methods/Racetrack/Tracks/Track1.png) and [Track2.png](https://github.com/c-boe/Reinforcement-learning/blob/main/5%20Monte%20Carlo%20methods/Racetrack/Tracks/Track2.png)) to be imported correctly.
For the problem to be solvable keep in mind the action space of Racetrack when designing a track.

