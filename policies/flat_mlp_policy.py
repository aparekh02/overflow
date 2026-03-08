"""
FlatMLPPolicy — sanity-check baseline.

Concatenates the full observation (ego + all tickets flattened) and passes
it through a standard MLP. No attention, no structure.

Use this to:
  1. Verify the reward signal and environment are working
  2. Establish a performance floor
  3. Confirm that TicketAttentionPolicy actually improves over this

If FlatMLPPolicy can't learn Stage 1 survival, the reward or env is broken.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_policy import BasePolicy


class FlatMLPPolicy(BasePolicy):
    """Standard 3-layer MLP over the full flat observation."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__(obs_dim)

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),                          nn.Tanh(),
            nn.Linear(hidden, hidden // 2),                     nn.Tanh(),
            nn.Linear(hidden // 2, 3),                          nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),    nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        return self.actor(obs), self.critic(obs)
