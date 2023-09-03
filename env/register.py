import gym
from radar import Radar

# Register the environment
gym.register(
    id='Radar-v0',
    entry_point='radar:Radar', 
)
