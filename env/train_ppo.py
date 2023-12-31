import radar
import gym, random
import os
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import matplotlib.pyplot as plt
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
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



# Create gym environment 
env = gym.make("Radar-v0")
#env.seed(42)
state = env.reset()
#print(state)
action = random.randint(0,10)
#print(action)
state,reward,done,_=env.step(action)
#print(reward)

# Train PPO
log_dir ="./ppo_temponly/"
os.makedirs(log_dir, exist_ok = True)
env = gym.make("Radar-v0")
state = env.reset()
print(env.observation_space, env.action_space)
env = Monitor(env, log_dir)
gym.logger.MIN_LEVEL = 0 # to prevent error
model = PPO2(MlpPolicy, env, verbose=1, learning_rate=1e-04 )
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Train the agent
time_steps = 1e7
model.learn(total_timesteps=int(time_steps), callback=callback)
model.save("ppo_radar_temponly_1e7")
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO Radar")
plt.savefig("ppo_radar_temponly_1e7.png")
plt.close()


#test
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

env.close()

