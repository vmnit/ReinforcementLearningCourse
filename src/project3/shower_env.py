import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ShowerEnv(gym.Env):
    """Custom environment for controlling shower temperature.

    The agent must learn to adjust the shower temperature to keep it
    within a comfortable range (37-39 degrees Celsius).

    Actions: 0 = decrease temp, 1 = do nothing, 2 = increase temp
    Observation: current temperature (float)
    Reward: +1 if temp is in [37, 39], -1 otherwise
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([100.0]),
            dtype=np.float32,
        )

        self.target_low = 37.0
        self.target_high = 39.0
        self.max_steps = 100

        self.render_mode = render_mode
        self.state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 38.0 + self.np_random.uniform(-10, 10)
        self.current_step = 0
        return np.array([self.state], dtype=np.float32), {}

    def step(self, action):
        noise = self.np_random.uniform(-0.5, 0.5)

        if action == 0:
            self.state -= 1.0
        elif action == 2:
            self.state += 1.0

        self.state += noise
        self.state = np.clip(self.state, 0.0, 100.0)

        self.current_step += 1

        if self.target_low <= self.state <= self.target_high:
            reward = 1.0
        else:
            reward = -1.0

        terminated = False
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return (
            np.array([self.state], dtype=np.float32),
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self):
        if self.render_mode == "human":
            in_range = self.target_low <= self.state <= self.target_high
            marker = "*" if in_range else " "
            print(
                "Step %3d | Temp: %5.1f [%s]"
                % (self.current_step, self.state, marker)
            )
