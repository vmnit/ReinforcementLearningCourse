
#import imageio
import gymnasium as gym
#from gymnasium.wrappers import RecordVideo

#import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


environment_name = "CartPole-v0"
#env = gym.make(environment_name, render_mode="rgb_array")
env = gym.make(environment_name, render_mode="human")
#env = RecordVideo(env, "./output")
#frames=[]
#env = gym.wrappers.Monitor(env, './output', force=True)


episodes = 20
for episode in range(1, episodes + 1):
    state = env.reset()
    print(f"{state=}")
    done = False
    score = 0

    while not done:
        env.render()
        #frames.append(env.render())
        action = env.action_space.sample()
        n_state, reward, finished, truncated, info = env.step(action)
        done = finished or truncated
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
#imageio.mimsave('simulation.gif', frames, fps=10)

