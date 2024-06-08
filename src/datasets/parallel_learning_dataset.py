import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn

from typing import Iterator
from torch.utils.data import IterableDataset


class ParallelLearningDataset(IterableDataset):
    def __init__(
        self,
        *,
        env: gym.vector.VectorEnv,
        policy: nn.Module,
        steps_per_epoch: np.int32,
        gamma: np.float32,
    ) -> None:
        self.env = env
        self.policy = policy
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma

        self.obs, _ = self.env.reset()

    def __iter__(self) -> Iterator:
        # Get worker info to know how many workers are running
        # worker_info = torch.utils.data.get_worker_info()

        transitions = []

        for _ in range(self.steps_per_epoch):
            policy_actions: torch.Tensor = self.policy(self.obs)
            actions: np.ndarray = policy_actions.multinomial(1).cpu().numpy()
            next_obs, reward, done, _, _ = self.env.step(actions.flatten())
            transitions.append((self.obs, actions, reward, done))
            self.obs = next_obs

        obs_batch, action_batch, reward_batch, done_batch = map(
            np.stack,
            zip(*transitions),
            # *transitions -> convert to tuple
            # zip function will take each element of the tuple and pas to the stack function
            # this will convert list((obs, action, reward, done)) and produce (ndarray(obs), ndarray(action), ndarray(reward), ndarray(done))
        )

        running_return: np.ndarray = np.zeros(self.env.num_envs, dtype=np.float32)
        return_batch: np.ndarray = np.zeros_like(reward_batch)

        for row in range(self.steps_per_epoch - 1, -1, -1):
            running_return = (
                reward_batch[row] + (1 - done_batch[row]) * self.gamma * running_return
            )
            return_batch[row] = running_return

        num_samples = self.env.num_envs * self.steps_per_epoch
        obs_batch = obs_batch.reshape(num_samples, -1)
        action_batch = action_batch.reshape(num_samples, -1)
        return_batch = return_batch.reshape(num_samples, -1)

        idx = list(range(num_samples))
        random.shuffle(idx)

        for i in idx:
            yield obs_batch[i], action_batch[i], return_batch[i]
