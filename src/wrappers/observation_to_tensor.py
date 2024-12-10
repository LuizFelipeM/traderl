import gymnasium as gym
import torch
from typing import Any, SupportsFloat


class ObservationToTensor(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return torch.from_numpy(obs).float(), info

    def step(
        self, action: torch.Tensor
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action = action.squeeze().numpy()
        obs, reward, done, truncated, info = super().step(action)

        obs = torch.from_numpy(obs).float()
        reward = torch.from_numpy(reward).unsqueeze(1).float()
        done = torch.from_numpy(done).unsqueeze(1)
        truncated = torch.from_numpy(truncated).unsqueeze(1)

        return obs, reward, done, truncated, info
