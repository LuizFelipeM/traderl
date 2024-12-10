from collections import namedtuple
from typing import Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm


class CriticDto:
    def __init__(self, value: torch.Tensor, target: torch.Tensor) -> None:
        self.value = value
        self.target = target


class AdvantageActorCritic:
    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        env: gym.Env | gym.vector.VectorEnv,
        *,
        episodes=np.int32(200),
        batch_size=np.int32(1_024),
        alpha=np.float32(1e-4),
        gamma=np.float32(0.99),
        entropy_coef=np.float32(0.01),
        policy_optim=AdamW,
        critic_optim=AdamW,
        device=torch.device("cpu"),
        logger=SummaryWriter(),
    ) -> None:
        self.env = env
        self.num_envs = 1 if self.env is gym.Env else self.env.num_envs

        state, _ = self.env.reset()
        expected_size = torch.Size([self.num_envs, 1])
        critic_size = critic(state).size()
        assert (
            critic_size == expected_size
        ), f"The critic does not an output of the correct format. Given {critic_size} but expected for {expected_size}"

        # Policy is consider the actor in this scenario
        self.policy = policy
        self.critic = critic
        self.episodes = episodes
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.policy_optim = policy_optim(self.policy.parameters(), lr=self.alpha)
        self.critic_optim = critic_optim(self.critic.parameters(), lr=self.alpha)
        self.device = device
        self.logger = logger

    def run(self) -> None:
        for episode in tqdm(range(1, self.episodes + 1)):
            ep_return = torch.zeros((self.num_envs, 1))

            state, _ = self.env.reset()
            done_b = torch.zeros((self.num_envs, 1), dtype=torch.bool)
            self.I = 1.0

            while not done_b.all():
                policy_action = self.policy(state).multinomial(1).detach()
                next_state, reward, done, _, _ = self.env.step(policy_action)

                critic, critic_loss = self._critic(state, reward, done, next_state)
                actor_loss = self._actor(state, policy_action, critic)

                ep_return += reward
                done_b |= done
                state = next_state
                self.I = self.I * self.gamma

            self.logger.add_scalar(
                "Policy Gradient Loss/Episode", actor_loss.item(), episode
            )
            self.logger.add_scalar(
                "Critic Gradient Loss/Episode", critic_loss.item(), episode
            )
            self.logger.add_scalar(
                "Mean Return/Episode", ep_return.mean().item(), episode
            )

            if hasattr(self.env, "episode_returns") and hasattr(
                self.env, "return_queue"
            ):
                self.logger.add_scalar(
                    "Last Return Queue/Episode", self.env.return_queue[-1], episode
                )

        self.logger.flush()
        self.logger.close()

    def _critic(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_state: torch.Tensor,
    ) -> tuple[CriticDto, torch.Tensor]:
        value: torch.Tensor = self.critic(state)
        target: torch.Tensor = (
            reward + ~done * self.gamma * self.critic(next_state).detach()
        )
        critic_loss = F.mse_loss(value, target)

        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return CriticDto(value, target), critic_loss

    def _actor(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        critic: CriticDto,
    ) -> torch.Tensor:
        advantage = (critic.target - critic.value).detach()
        probabilities = self.policy(state)
        log_probabilities = torch.log(probabilities + 1e-6)
        action_log_probabilities = log_probabilities.gather(1, action)
        entropy = -torch.sum(probabilities * log_probabilities, dim=-1, keepdim=True)
        actor_loss = (
            -self.I * action_log_probabilities * advantage - self.entropy_coef * entropy
        )
        actor_loss = actor_loss.mean()

        self.policy.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        return actor_loss
