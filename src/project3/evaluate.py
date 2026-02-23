import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(__file__))
from shower_env import ShowerEnv

ppo_path = os.path.join('Training', 'SavedModels', 'PPO_Shower_100K')

eval_env = DummyVecEnv([lambda: ShowerEnv()])
model = PPO.load(ppo_path, env=eval_env)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
print("Evaluation over 10 episodes:")
print("  Mean reward: %.2f +/- %.2f" % (mean_reward, std_reward))
eval_env.close()

render_env = ShowerEnv(render_mode="human")
episodes = 5
for episode in range(1, episodes + 1):
    obs, info = render_env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        done = terminated or truncated
        score += reward
    print("Episode %d - Total Score: %.1f" % (episode, score))

render_env.close()
