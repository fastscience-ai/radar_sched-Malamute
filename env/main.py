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
# env.set_seed(72)
state, info = env.reset(seed=72)
print("state",state)
# action = random.randint(0,10)
action = random.randint(0,5)
print("action",action)
state,reward,done,info=env.step(action)
print("reward",reward, "state", state)


observation, info = env.reset()
for _ in range(1000):
   # action = random.randint(0,10)
   action = random.randint(0,5)
   observation, reward, done, info = env.step(action)
   print(observation, reward, done)
   if done:
      observation, info = env.reset()
env.close()

