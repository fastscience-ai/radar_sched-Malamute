#from env import radar
import gym
#from env import register
import numpy as np
import os
import random
import radar
# Register the environment
gym.envs.registration.register(
    id='Radar-v0',
    entry_point='radar:Radar',
)



#(1) Create gym environment 
env = gym.make("Radar-v0")
state = env.reset()
print(state)
action = random.randint(0,10)
print(action)
state,reward,done,_=env.step(action)
print(reward)


