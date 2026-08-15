## RCDTM-MARL ##

## RUN ## 

-Simply run the main file to see the training process of a 5x5x5 network.

## agent.py ## 

-This file defines the attributes and methods of the agent, including the neural network, action selection function, training function, and so on.

## buoy.py ## 
-This file defines three classes: sensor, sensor network, and data packet.

## environment.py ## 

-This file defines the environment, including functions for obtaining observations, executing actions, generating data packets, refreshing the network, and more.

## replay_buffer.py ## 

-This file defines two experience replay buffers.

## runner.py ## 

-This is the execution file for reinforcement learning, mainly involving the interaction between the agent and the environment. It also includes the tracing mechanism.