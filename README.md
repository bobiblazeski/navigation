# Navigation
Navigation project from Udacity Deep Reinforcement Learning Nanodegree.
It demonstrates how to teach an agent to collect yellow bananas while avoiding blue bananas. 

## Installation
### Install deep reinforcement learning repository
1. Clone [deep reinforcement learning repository](https://github.com/udacity/deep-reinforcement-learning)
2. Fallow the instructions to install necessary [dependencies](https://github.com/udacity/deep-reinforcement-learning#dependencies)
### Download the Unity Environment
1. Download environment for your system into this repository root

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Unzip (or decompress) the archive
### Run the project
1. Start the jupyter server
2. Open the Navigation.ipynb notebook
3. Change the kernel to drlnd
4. You should be able to run all the cells

## Environment
This project uses the Unity based environment prepared by the Udacity  team.

There is one agent interacting with the environment.

There are 4 actions available to the agent:
* 0 - walk forward
* 1 - walk backward
* 2 - turn left
* 3 - turn right

The state is represented as a vector of 37 dimensions.

There is a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana.

## Weights
The directory `saves` contains saved weights for 4 different agents:

* `checkpoint_single_16.pth` - DQN 
* `checkpoint_double_16.pth` - Double DQN
* `checkpoint_dueling_16.pth` - Dueling Double DQN
* `checkpoint_priority.pth` - Priority Experience + Dueling Double DQN


## Credits
Most of the code is based on Deep Q-Networks lesson. The Experience Replay Buffer and SumTree are minimally adapted from Yuan Liu's [RainBow](https://github.com/cmusjtuliuyuan/RainBow) implementation.