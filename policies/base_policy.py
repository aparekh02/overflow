"""
BasePolicy — abstract interface all policies implement.

All policies expose the same predict() and train_step() API so the
curriculum trainer can swap them out transparently.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BasePolicy(nn.Module, abc.ABC):
    """
    Abstract base for all driving policies.

    Subclasses implement:
        forward(obs_tensor)  → action_tensor, value_tensor
        encode_obs(obs_np)   → torch.Tensor
    """

    def __init__(self, obs_dim: int, action_dim: int = 3):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_mean  — shape (B, action_dim)
            value        — shape (B, 1)
        """
        ...

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Numpy in, numpy out. Used by the env during rollout."""
        self.eval()
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, _ = self.forward(t)
            if deterministic:
                action = mean
            else:
                action = mean + torch.randn_like(mean) * 0.1
        return action.squeeze(0).numpy()

    @staticmethod
    def _mlp(dims: list[int], activation=nn.Tanh) -> nn.Sequential:
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        return nn.Sequential(*layers)
