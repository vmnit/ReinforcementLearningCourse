import argparse
import os
import sys

import gymnasium as gym
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

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
        description="Evaluate a saved Stable-Baselines3 model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the saved model (e.g. PPO_model, PPO_2). Path: Training/SavedModels/<model_name>.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        choices=list(ALGOS.keys()),
        help="Algorithm type (default: PPO).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v0",
        help="Gymnasium environment ID (default: CartPole-v0).",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Run evaluation without rendering.",
    )
    args = parser.parse_args()

    model_path = os.path.join("Training", "SavedModels", args.model_name)
    path_zip = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.isfile(model_path) and not os.path.isfile(path_zip):
        print(
            f"Error: Model not found at {model_path} or {path_zip}.",
            file=sys.stderr,
        )
        sys.exit(1)

    AlgoClass = ALGOS[args.algo]
    render_mode = "human" if not args.no_render else None
    env = gym.make(args.env, render_mode=render_mode)
    env = DummyVecEnv([lambda: env])

    model = AlgoClass.load(model_path, env=env)
    evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_episodes,
        render=not args.no_render,
    )
    env.close()


if __name__ == "__main__":
    main()
