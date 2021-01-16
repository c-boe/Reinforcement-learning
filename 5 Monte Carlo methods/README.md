# Chapter 5: Monte-Carlo Methods

Implementation of Example 5.3 (Blackjack) and Exercise 5.12 (Racetrack) of Sutton and Barto's [Reinforcement learning](http://incompleteideas.net/book/the-book.html).


## Blackjack (Section 5.3)

The Blackjack environment (blackjack.py) is solved using Monte-Carlo Control with exploring states (Monte_Carlo_with_ES).

## Racetrack (Section 5.7)

The racetrack environment (racetrack.py) is solved using off-policy Monte-Carlo control (Off_policy_MC.py). 
Users can design their own track in Paint, add a .png to the Tracks folder and set the "Track_img"-parameter in "Off_policy_MC.py". The colors of the track, the starting and the goal line should be the same as for the example tracks (Track1.png and Track2.png) to be imported correctly.
For the problem to be solvable keep in mind the action space of Racetrack when designing a track.

