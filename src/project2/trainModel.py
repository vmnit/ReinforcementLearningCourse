import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = "CarRacing-v3"

env = DummyVecEnv([lambda: gym.make(environment_name, continuous=True)])
env = VecTransposeImage(env)

log_path = os.path.join('Training', 'Logs')

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=100000)

ppo_path = os.path.join('Training', 'SavedModels', 'PPO_CarRacing_100K')
model.save(ppo_path)

del model
model = PPO.load(ppo_path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=False)
