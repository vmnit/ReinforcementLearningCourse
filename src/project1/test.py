import ale_py
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["SDL_RENDER_DRIVER"] = "software"

gym.register_envs(ale_py)

#environment_name = "Breakout-v0"
environment_name = "ALE/Breakout-v5"
env = gym.make(environment_name, render_mode="human")

episodes = 5
for episode in range(1, episodes+1):
    state, info = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
