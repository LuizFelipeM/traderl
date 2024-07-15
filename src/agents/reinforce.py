import gymnasium as gym
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm


class Reinforce:
    def __init__(
        self,
        policy: nn.Module,
        env: gym.Env | gym.vector.VectorEnv,
        *,
        episodes=np.int32(200),
        batch_size=np.int32(1_024),
        alpha=np.float32(1e-4),
        gamma=np.float32(0.99),
        entropy_coef=np.float32(0.01),
        optim=AdamW,
        device=torch.device("cpu"),
        logger=SummaryWriter(),
    ) -> None:
        self.policy = policy
        self.env = env
        self.num_envs = 1 if self.env is gym.Env else self.env.num_envs
        self.episodes = episodes
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.optim = optim(self.policy.parameters(), lr=self.alpha)
        self.device = device
        self.logger = logger

    def run(self) -> None:
        for episode in tqdm(range(1, self.episodes + 1)):
            transitions, ep_return = self._collect_episode_experience()
            total_loss = self._backpropagate(transitions)

            self.logger.add_scalar(
                "Policy Gradient Loss/Episode", total_loss.item(), episode
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

    def _collect_episode_experience(
        self,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
        transitions = []
        ep_return = torch.zeros((self.num_envs, 1))

        state, _ = self.env.reset()
        done_b = torch.zeros((self.num_envs, 1), dtype=torch.bool)

        while not done_b.all():
            action = self.policy(state).multinomial(1).detach()
            next_state, reward, done, _, _ = self.env.step(action)
            transitions.append((state, action, ~done * reward))
            ep_return += reward
            done_b |= done
            state = next_state

        return transitions, ep_return

    def _backpropagate(
        self, transitions: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        G = torch.zeros((self.num_envs, 1))

        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            G = reward_t + self.gamma * G
            probs_t = self.policy(state_t)
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(1, action_t)

            entropy_t = -torch.sum(probs_t * log_probs_t, dim=-1, keepdim=True)
            gamma_t = self.gamma**t

            pg_loss_t = -gamma_t * action_log_prob_t * G
            total_loss_t = (pg_loss_t - self.entropy_coef * entropy_t).mean()

            self.policy.zero_grad()
            total_loss_t.backward()
            self.optim.step()

        return total_loss_t
