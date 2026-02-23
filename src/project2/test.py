import gymnasium as gym
import os

os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["SDL_RENDER_DRIVER"] = "software"

environment_name = "CarRacing-v3"
env = gym.make(environment_name, render_mode="human")

episodes = 5
for episode in range(1, episodes + 1):
    state, info = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
    print("Episode:{} Score:{:.2f}".format(episode, score))
env.close()
