import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


environment_name = "CartPole-v0"
log_path = os.path.join('Training', 'Logs')

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000)

PPO_path = os.path.join('Training', 'SavedModels', 'PPO_model')
model.save(PPO_path)

del model
model = PPO.load(PPO_path, env=env)
model.learn(total_timesteps=2000)
model.save(PPO_path)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
