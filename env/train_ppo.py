import radar
import gym
import os
import numpy as np
import time
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import matplotlib.pyplot as plt
# from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True
# Register the environment
gym.envs.registration.register(
    id='Radar-v0',
    entry_point='radar:Radar',
)



# # Create gym environment
# env = gym.make("Radar-v0")
# #env.seed(42)
# state = env.reset()
# #print(state)
# action = random.randint(0,10)
# #print(action)
# state,reward,done,_=env.step(action)
# #print(reward)

# Train PPO
train_timesteps = '1e6'
# log_dir ="./rew_pd/timesteps_" + train_timesteps
log_dir ="./rew_det_strength/timesteps_" + train_timesteps
# log_dir ="./temp_model/"
os.makedirs(log_dir, exist_ok = True)
env = gym.make("Radar-v0")
state = env.reset()
print(env.observation_space, env.action_space)
env = Monitor(env, log_dir)
gym.logger.MIN_LEVEL = 0 # to prevent error
model = PPO2(MlpPolicy, env, verbose=1, learning_rate=1e-04 )
# Train the agent
time_steps = eval(train_timesteps)
# time_steps = 100
# check_freq = 2
check_freq = 1000
callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
t_start = time.time()
model.learn(total_timesteps=int(time_steps), callback=callback)
t_end = time.time()
print(f'Training took {(t_end - t_start)/60} minutes')
model.save(os.path.join(log_dir, "last_model"))
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO Radar")
plt.savefig(os.path.join(log_dir, 'results.png'))
plt.show()
plt.close()


# test
target_det_strength=0
target_avg_detection=0
target_pd=0
obs = env.reset()
# dones_list = []
# rewards_list = []
num_episodes = 0
num_targets = env.env.env.env.NUM_TARGETS
episode_timesteps = env.env.env.env.SIM_TIME

test_timesteps = 10000
for i in range(test_timesteps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
        num_episodes += 1
    # rewards_list.append(rewards)
    # dones_list.append(dones)
    target_det_strength += sum(obs[:, 2]) # target detection strength
    target_avg_detection += sum(obs[:, 5]) # target detected or not
    target_pd += sum(obs[:, 6]) # target pd

total_timesteps = num_episodes * episode_timesteps

# Random baseline

target_det_strength_random=0
target_avg_detection_random=0
target_pd_random=0
obs = env.reset()
# dones_list = []
# rewards_list = []
num_episodes_random = 0
num_targets = env.env.env.env.NUM_TARGETS
episode_timesteps = env.env.env.env.SIM_TIME
num_actions = env.env.env.env.num_actions

test_timesteps = 10000
for i in range(test_timesteps):
    # action, _states = model.predict(obs)
    action = np.random.randint(num_actions)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
        num_episodes_random += 1
    # rewards_list.append(rewards)
    # dones_list.append(dones)
    target_det_strength_random += sum(obs[:, 2]) # target detection strength
    target_avg_detection_random += sum(obs[:, 5]) # target detected or not
    target_pd_random += sum(obs[:, 6]) # target pd

total_timesteps_random = num_episodes_random * episode_timesteps


print("\n\n------Evaluation results-----\n")

print(f"No. of targets: {num_targets}, Test timesteps: {test_timesteps}, Test episodes: {num_episodes}, Avg. episode length: {test_timesteps/num_episodes}")
print(f"Avg. target detection strength per timestep: {target_det_strength/total_timesteps/num_targets}")
print(f"Avg. target detection per timestep: {target_avg_detection/total_timesteps/num_targets}")
print(f"Avg. target pd per timestep: {target_pd/total_timesteps/num_targets}")
# print(f"Avg. episode rewards: {sum(rewards_list)/len(rewards_list)}")



print("\n\n------Random Baseline Results-------\n")

print(f"No. of targets: {num_targets}, Test timesteps: {test_timesteps}, Test episodes: {num_episodes_random}, Avg. episode length: {test_timesteps/num_episodes_random}")
print(f"Avg. target detection strength per timestep: {target_det_strength_random/total_timesteps_random/num_targets}")
print(f"Avg. target detection per timestep: {target_avg_detection_random/total_timesteps_random/num_targets}")
print(f"Avg. target pd per timestep: {target_pd_random/total_timesteps_random/num_targets}")
# print(f"Avg. episode rewards: {sum(rewards_list)/len(rewards_list)}")


env.close()

