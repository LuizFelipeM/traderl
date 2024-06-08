from typing import Iterator
import pandas as pd
import gymnasium as gym
import torch.utils
import torch.utils.data
import torch
import numpy as np

import torch.utils.data.dataloader

from environments.trading import register_trading_env
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from policies.gradient_policy import GradientPolicy

from datasets.parallel_learning_dataset import ParallelLearningDataset

register_trading_env()


def create_env(env_name: str, num_envs: np.int32, **kwargs) -> gym.vector.VectorEnv:
    env = gym.vector.make(env_name, num_envs=num_envs, **kwargs)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    return env


data = pd.read_csv("BTCUSDT-1s-2023-01.csv")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env = create_env("TradingEnv-v0", num_envs=2, data=data)


policy = GradientPolicy(
    in_features=np.int32(env.observation_space.shape[1]),
    n_actions=np.int32(env.single_action_space.n),
    device=device,
)
dataset = ParallelLearningDataset(env=env, policy=policy, steps_per_epoch=2, gamma=0.8)

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
