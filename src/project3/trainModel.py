import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(__file__))
from shower_env import ShowerEnv

log_path = os.path.join('Training', 'Logs')

env = DummyVecEnv([lambda: ShowerEnv()])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=100000)

ppo_path = os.path.join('Training', 'SavedModels', 'PPO_Shower_100K')
model.save(ppo_path)

del model
model = PPO.load(ppo_path, env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print("Mean reward: %.2f +/- %.2f" % (mean_reward, std_reward))
