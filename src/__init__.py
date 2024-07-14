from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
import os
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torch import nn

from agents import Reinforce, AdvantageActorCritic
from datasets import ParallelLearningDataset
from environments.trading import register_trading_env
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from policies.gradient_policy import GradientPolicy


register_trading_env()


class TorchEnvProcessor(gym.Wrapper):
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


def create_env(env_name: str, num_envs: np.int32, **kwargs) -> gym.vector.VectorEnv:
    env = gym.vector.make(env_name, num_envs=num_envs.item(), **kwargs)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    env = TorchEnvProcessor(env)
    return env


data = pd.read_csv("BTCUSDT-1s-2023-01.csv")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_envs = np.int32(
    torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
)
env = create_env("TradingEnv-v0", num_envs=np.int32(2), data=data)


policy = GradientPolicy(
    in_features=np.int32(env.observation_space.shape[1]),
    n_actions=np.int32(env.single_action_space.n),
    device=device,
)
critic = nn.Sequential(
    nn.Linear(env.observation_space.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
dataset = ParallelLearningDataset(env=env, policy=policy, steps_per_epoch=2, gamma=0.8)

if __name__ == "__main__":
    algo = Reinforce(
        policy,
        dataset,
        create_env("TradingEnv-v0", num_envs=num_envs, data=data),
        episodes=np.int32(500),
        device=device,
    )
    # algo = AdvantageActorCritic(
    #     policy,
    #     critic,
    #     dataset,
    #     create_env("TradingEnv-v0", num_envs=num_envs, data=data),
    #     episodes=np.int32(500),
    #     device=device,
    # )

    algo.run()


# print(env.reset())

# action = env.action_space.sample()
# print(action)
# print(env.step(action))

# print(f"----- 0 workers -----")
# print(list(torch.utils.data.DataLoader(dataset, num_workers=0)))

# print(f"----- 2 workers -----")
# print(list(torch.utils.data.DataLoader(dataset, num_workers=2)))

# print(f"----- 12 workers -----")
# print(list(torch.utils.data.DataLoader(dataset, num_workers=12)))
