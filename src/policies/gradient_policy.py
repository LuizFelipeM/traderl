from typing import Any
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class GradientPolicy(nn.Module):
    device: torch.device

    def __init__(
        self,
        *,
        in_features: np.int32,
        n_actions: np.int32,
        hidden_size=np.int32(128),
        device=torch.device("cpu")
    ) -> None:
        """
        `in_features` is the number of features in the input layer of the policy

        `n_actions` is the number of actions that the policy can generate in the output layer

        `hidden_size` is the Number of neurons of the policy's hidden layer
        """
        super(GradientPolicy, self).__init__()

        self.input_layer = nn.Linear(in_features.item(), hidden_size.item())
        self.hidden_layer = nn.Linear(hidden_size.item(), hidden_size.item())
        self.output_layer = nn.Linear(hidden_size.item(), n_actions.item())

        self.device = device

    def forward(self, input: Any) -> torch.Tensor:
        x = torch.tensor(input).float().to(self.device)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.softmax(self.output_layer(x), dim=-1)
        return x
