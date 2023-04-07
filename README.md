# Computer_Vision_Final_Project
The final project involves game development based on Computer Vision tools.
In this project we created a game based on the principles of the game "Dartboard". 
The game combines active participation of a competitor whose goal is to throw a ball towards a screen on which balloons moving upwards are projected,
and hit the balloon. Each hit earns the competitor a point and the rate of the balloons rising increases so that the level of difficulty
is increased. During the game, each competitor has 6 attempts to hit the balloons.

In this project we used image processing tools, which we learned in the course, in order to build an interactive game. As part of the game, we project balloons which gradually rise above the screen when the player's goal is to hit the balloon with a ball and "blow up" it.
The project was developed in python. The main tools and libraries used are openCV, pygame, multitreading, matplotlib.

The game interface was developed in the game.py file.
The ball hit detection using computer vision tools was developed in the detection_game_ver1.py file.
We used multitreading to support real-time, see in the main.py file.
