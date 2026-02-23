import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import os

os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["SDL_RENDER_DRIVER"] = "software"

environment_name = "CarRacing-v3"
ppo_path = os.path.join('Training', 'SavedModels', 'PPO_CarRacing_100K')

env = DummyVecEnv([lambda: gym.make(environment_name, continuous=True, render_mode="human")])
env = VecTransposeImage(env)

model = PPO.load(ppo_path, env=env)

episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        env.render()
        score += rewards[0]
        done = dones[0]
    print("Episode %d - Score: %.1f" % (episode, score))

env.close()
