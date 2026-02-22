import ale_py
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

gym.register_envs(ale_py)

environment_name = "ALE/Breakout-v5"
#environment_name = "Breakout-v0"
env = make_atari_env(environment_name, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

log_path = os.path.join('Training', 'Logs')

model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=3000000)

a2c_path = os.path.join('Training', 'SavedModels', 'A2C_3M_model')
model.save(a2c_path)

del model
model = A2C.load(a2c_path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)