"""
TicketAttentionPolicy — the main policy (Stage 2+).

Architecture: two-pass "reflective" cross-attention.

    Pass 1: ego queries tickets → raw threat context
    Pass 2: (ego + raw context) queries tickets again → refined context
    This forces the policy to "think twice" — first perceive, then plan.

    [ego | refined_context] → steer head  → steer action
                            → drive head  → throttle, brake
                            → critic head → value

Why two-pass:
  The first pass gathers what threats exist. The second pass re-examines
  tickets knowing what the overall threat picture looks like. This prevents
  the impulsive single-shot responses that cause wild oscillation.

Why separate heads:
  Steering requires smooth, conservative output (off-road = death).
  Throttle/brake can be more aggressive. Separate heads + separate
  noise levels let each dimension learn at its own pace.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_policy import BasePolicy
EGO_STATE_DIM = 11
MAX_TICKETS = 16
TICKET_VECTOR_DIM = 37


class TicketAttentionPolicy(BasePolicy):
    """
    Two-pass reflective attention policy.

    Pass 1: perceive — what threats exist?
    Pass 2: plan    — given what I see, which threats matter most?
    Output: separate steer head (conservative) + drive head (throttle/brake)
    """

    def __init__(
        self,
        obs_dim:      int,
        ego_embed:    int = 64,
        ticket_embed: int = 64,
        n_heads:      int = 4,
        hidden:       int = 256,
    ):
        super().__init__(obs_dim)
        assert ego_embed % n_heads == 0
        assert ticket_embed == ego_embed

        self.ego_embed    = ego_embed
        self.max_tickets  = MAX_TICKETS
        self.ticket_dim   = TICKET_VECTOR_DIM

        # ── Encoders ──────────────────────────────────────────────────────
        self.ego_encoder = nn.Sequential(
            nn.Linear(EGO_STATE_DIM, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, ego_embed),
            nn.LayerNorm(ego_embed),
        )
        self.ticket_encoder = nn.Sequential(
            nn.Linear(TICKET_VECTOR_DIM, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, ticket_embed),
            nn.LayerNorm(ticket_embed),
        )

        # ── Pass 1: perceive (ego queries tickets) ───────────────────────
        self.attn_pass1 = nn.MultiheadAttention(
            embed_dim=ego_embed, num_heads=n_heads,
            dropout=0.0, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(ego_embed)

        # ── Reflection gate: fuse ego + pass1 context for second query ───
        self.reflect_proj = nn.Sequential(
            nn.Linear(ego_embed * 2, ego_embed),
            nn.LayerNorm(ego_embed),
            nn.Tanh(),
        )

        # ── Pass 2: plan (refined query re-attends to tickets) ───────────
        self.attn_pass2 = nn.MultiheadAttention(
            embed_dim=ego_embed, num_heads=n_heads,
            dropout=0.0, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(ego_embed)

        # ── Fused representation ─────────────────────────────────────────
        fused_dim = ego_embed + ego_embed  # ego + refined context

        # ── Steer head (conservative, smooth output) ─────────────────────
        self.steer_head = nn.Sequential(
            nn.Linear(fused_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 1),
            nn.Tanh(),
        )

        # ── Drive head (throttle + brake) ────────────────────────────────
        self.drive_head = nn.Sequential(
            nn.Linear(fused_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 2),
            nn.Tanh(),
        )

        # ── Critic head ──────────────────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Very small initial actions — start by doing almost nothing
        nn.init.orthogonal_(self.steer_head[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.drive_head[-2].weight, gain=0.01)
        # Critic starts near zero
        nn.init.orthogonal_(self.critic[-1].weight, gain=0.1)

    def _attend(self, attn_module, norm_module, query, tk_emb, is_padding, all_empty):
        """Run one attention pass with NaN-safe masking."""
        B = query.shape[0]
        q = query if query.dim() == 3 else query.unsqueeze(1)

        if all_empty.all():
            return torch.zeros(B, self.ego_embed, device=query.device)

        safe_mask = is_padding.clone()
        safe_mask[all_empty, 0] = False
        attn_out, _ = attn_module(
            query=q, key=tk_emb, value=tk_emb,
            key_padding_mask=safe_mask,
        )
        context = attn_out.squeeze(1)
        context[all_empty] = 0.0
        return norm_module(context)

    def forward(self, obs: torch.Tensor):
        B = obs.shape[0]

        # Split observation
        ego_raw = obs[:, :EGO_STATE_DIM]
        tk_raw  = obs[:, EGO_STATE_DIM:].view(B, self.max_tickets, self.ticket_dim)

        # Encode
        ego_emb = self.ego_encoder(ego_raw)
        tk_emb  = self.ticket_encoder(tk_raw)

        # Padding mask
        is_padding = (tk_raw.abs().sum(dim=-1) == 0)
        all_empty  = is_padding.all(dim=-1)

        # ── Pass 1: perceive ─────────────────────────────────────────────
        ctx1 = self._attend(self.attn_pass1, self.norm1,
                            ego_emb, tk_emb, is_padding, all_empty)

        # ── Reflect: combine ego + initial context into refined query ────
        reflected = self.reflect_proj(torch.cat([ego_emb, ctx1], dim=-1))

        # ── Pass 2: plan (re-attend with richer query) ───────────────────
        ctx2 = self._attend(self.attn_pass2, self.norm2,
                            reflected, tk_emb, is_padding, all_empty)

        # ── Fuse and decode ──────────────────────────────────────────────
        fused = torch.cat([ego_emb, ctx2], dim=-1)

        steer  = self.steer_head(fused)          # (B, 1)
        drive  = self.drive_head(fused)           # (B, 2)
        action = torch.cat([steer, drive], dim=-1)  # (B, 3)
        value  = self.critic(fused)               # (B, 1)

        return action, value

    def get_attention_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns pass-2 attention weights for interpretability."""
        B = obs.shape[0]
        ego_raw = obs[:, :EGO_STATE_DIM]
        tk_raw  = obs[:, EGO_STATE_DIM:].view(B, self.max_tickets, self.ticket_dim)
        ego_emb = self.ego_encoder(ego_raw)
        tk_emb  = self.ticket_encoder(tk_raw)
        is_padding = (tk_raw.abs().sum(dim=-1) == 0)
        all_empty = is_padding.all(dim=-1)

        # Pass 1
        ctx1 = self._attend(self.attn_pass1, self.norm1,
                            ego_emb, tk_emb, is_padding, all_empty)
        reflected = self.reflect_proj(torch.cat([ego_emb, ctx1], dim=-1))

        # Pass 2 — get weights
        safe_mask = is_padding.clone()
        safe_mask[all_empty, 0] = False
        query = reflected.unsqueeze(1)
        _, weights = self.attn_pass2(
            query=query, key=tk_emb, value=tk_emb,
            key_padding_mask=safe_mask,
            need_weights=True, average_attn_weights=False,
        )
        weights[all_empty] = 0.0
        return weights
