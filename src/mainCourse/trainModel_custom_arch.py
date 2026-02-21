import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Custom network: policy (pi) and value (vf) each have 4 layers of 128 units
net_arch = [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]

environment_name = "CartPole-v0"
log_path = os.path.join("Training", "Logs")
save_path = os.path.join("Training", "SavedModels", "PPO_custom_arch")
eval_save_dir = os.path.join("Training", "SavedModels")

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=stop_callback,
    eval_freq=10000,
    best_model_save_path=eval_save_dir,
    verbose=1,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_path,
    policy_kwargs={"net_arch": net_arch},
)

model.learn(total_timesteps=20000, callback=eval_callback)
model.save(save_path)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
