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
env.seed(72)
state = env.reset()
print("state",state)
action = random.randint(0,10)
print("action",action)
state,reward,done,_=env.step(action)
print("reward",reward, "state", state)


observation, info = env.reset()
for _ in range(1000):
   action = random.randint(0,10)
   observation, reward, done, _ = env.step(action)
   print(observation, reward, done)
   if done:
      observation, info = env.reset()
env.close()

