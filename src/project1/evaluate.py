import ale_py
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["SDL_RENDER_DRIVER"] = "software"

gym.register_envs(ale_py)

environment_name = "ALE/Breakout-v5"
a2c_path = os.path.join('Training', 'SavedModels', 'A2C_3M_model')

# Quantitative evaluation (no rendering)
eval_env = make_atari_env(environment_name, n_envs=1, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=4)

model = A2C.load(a2c_path, env=eval_env)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
print("Evaluation over 10 episodes:")
print("  Mean reward: %.2f +/- %.2f" % (mean_reward, std_reward))

eval_env.close()

# Visual evaluation with rendering
render_env = make_atari_env(environment_name, n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
render_env = VecFrameStack(render_env, n_stack=4)

episodes = 5
for episode in range(1, episodes + 1):
    obs = render_env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = render_env.step(action)
        score += rewards[0]
        done = dones[0]
    print("Episode %d - Score: %.1f" % (episode, score))

render_env.close()
