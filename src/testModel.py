import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


environment_name = "CartPole-v0"
log_path = os.path.join('Training', 'Logs')

env = gym.make(environment_name, render_mode="human")
env = DummyVecEnv([lambda: env])

PPO_path = os.path.join('Training', 'SavedModels', 'PPO_model')

model = PPO.load(PPO_path, env=env)

episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        #frames.append(env.render())
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
#imageio.mimsave('simulation.gif', frames, fps=10)

