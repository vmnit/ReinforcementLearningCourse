"""
Generic training script: train any Stable-Baselines3 algorithm with callbacks and evaluate.
No custom net_arch; uses default policy architecture per algorithm.
"""
import argparse
import os
import gymnasium as gym
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train a Stable-Baselines3 model with callbacks and evaluate (no custom net_arch).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="DQN",
        choices=list(ALGOS.keys()),
        help="Algorithm to use (default: DQN).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v0",
        help="Gymnasium environment ID (default: CartPole-v0).",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=20000,
        help="Total training timesteps (default: 20000).",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=190.0,
        help="Stop training when eval mean reward reaches this (default: 190).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Saved model name (default: <algo>_callback).",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Evaluate every N steps (default: 10000).",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes for final evaluation (default: 10).",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Run final evaluation without rendering.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: Training/Logs/<algo>).",
    )
    args = parser.parse_args()

    AlgoClass = ALGOS[args.algo]
    model_name = args.model_name if args.model_name is not None else f"{args.algo}_callback"
    log_path = args.log_dir if args.log_dir is not None else os.path.join("Training", "Logs", args.algo)
    save_path = os.path.join("Training", "SavedModels", model_name)
    eval_save_dir = os.path.join("Training", "SavedModels")

    env = gym.make(args.env)
    env = DummyVecEnv([lambda: env])

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=args.reward_threshold,
        verbose=1,
    )
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=stop_callback,
        eval_freq=args.eval_freq,
        best_model_save_path=eval_save_dir,
        verbose=1,
    )

    model = AlgoClass(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(save_path)

    evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_eval_episodes,
        render=not args.no_render,
    )
    env.close()


if __name__ == "__main__":
    main()
