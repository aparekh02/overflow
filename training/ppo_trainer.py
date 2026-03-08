"""
PPO trainer — ported directly from openenv/training/ppo_trainer.py.

Same algorithm, same hyperparameters, same GAE implementation.
Only change: uses OverflowGymEnv instead of CarEnv3D.

Usage:
    from overflow_env.training.ppo_trainer import run_training
    run_training(policy_type="attention", total_steps=2_000_000)
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .overflow_gym_env import OverflowGymEnv
from .curriculum import CurriculumManager
from .reward import compute_episode_bonus
from ..policies.base_policy import BasePolicy
from ..policies.policy_spec import OBS_DIM


# ── Rollout buffer ─────────────────────────────────────────────────────────────
# Identical to openenv/training/ppo_trainer.py

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, device: torch.device):
        self.n    = n_steps
        self.obs  = torch.zeros(n_steps, obs_dim, device=device)
        self.acts = torch.zeros(n_steps, 3,        device=device)
        self.rew  = torch.zeros(n_steps,            device=device)
        self.val  = torch.zeros(n_steps,            device=device)
        self.logp = torch.zeros(n_steps,            device=device)
        self.done = torch.zeros(n_steps,            device=device)
        self.ptr  = 0

    def add(self, obs, act, rew, val, logp, done):
        i = self.ptr
        self.obs[i]  = torch.as_tensor(obs, dtype=torch.float32)
        self.acts[i] = torch.as_tensor(act, dtype=torch.float32)
        self.rew[i]  = float(rew)
        self.val[i]  = float(val)
        self.logp[i] = float(logp)
        self.done[i] = float(done)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n

    def reset(self):
        self.ptr = 0

    def compute_returns(self, last_val: float, gamma: float, gae_lambda: float):
        """Generalized Advantage Estimation — identical to openenv."""
        adv = torch.zeros_like(self.rew)
        gae = 0.0
        for t in reversed(range(self.n)):
            next_val = last_val if t == self.n - 1 else float(self.val[t + 1])
            delta    = self.rew[t] + gamma * next_val * (1 - self.done[t]) - self.val[t]
            gae      = delta + gamma * gae_lambda * (1 - self.done[t]) * gae
            adv[t]   = gae
        self.ret = adv + self.val


# ── PPO Trainer ────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Identical to openenv PPOTrainer — same hyperparameters, same PPO update.
    Environment is OverflowGymEnv instead of CarEnv3D.
    """

    def __init__(
        self,
        policy:        BasePolicy,
        env:           OverflowGymEnv,
        curriculum:    Optional[CurriculumManager] = None,
        # PPO hyperparameters — same defaults as openenv
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_range:    float = 0.2,
        clip_range_vf: float = 0.2,
        ent_coef:      float = 0.02,
        vf_coef:       float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps:       int   = 2048,
        batch_size:    int   = 256,
        n_epochs:      int   = 10,
        save_dir:      str   = "checkpoints",
        log_interval:  int   = 10,
        device:        str   = "auto",
    ):
        self.policy     = policy
        self.env        = env
        self.curriculum = curriculum or CurriculumManager()
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip       = clip_range
        self.clip_vf    = clip_range_vf
        self.ent_coef   = ent_coef
        self.vf_coef    = vf_coef
        self.max_grad   = max_grad_norm
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.log_every  = log_interval
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else \
                     "mps"  if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.policy.to(self.device)

        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=500,
        )

        self.buffer = RolloutBuffer(n_steps, OBS_DIM, self.device)

        self.ep_rewards  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self.total_steps = 0
        self.n_updates   = 0

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(self, total_steps: int = 2_000_000) -> None:
        print(f"\n{'='*70}", flush=True)
        print(f"  OpenENV PPO Training — policy={self.policy.__class__.__name__}", flush=True)
        print(f"  total_steps={total_steps}  n_steps={self.n_steps}  lr={self.optimizer.param_groups[0]['lr']:.0e}", flush=True)
        print(f"  gamma={self.gamma}  gae_lambda={self.gae_lambda}  clip={self.clip}  ent_coef={self.ent_coef}", flush=True)
        print(f"{'='*70}\n", flush=True)

        obs, _ = self.env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        t0 = time.time()

        while self.total_steps < total_steps:
            self.buffer.reset()
            self.policy.eval()

            # ── Collect rollout ──────────────────────────────────────────────
            for _ in range(self.n_steps):
                # Curriculum step (returns [] for OverflowEnv — kept for API compat)
                self.curriculum.step(self.env._sim_time)

                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    act_mean, val = self.policy(obs_t.unsqueeze(0))
                act_mean = act_mean.squeeze(0)
                val      = val.squeeze(0)

                dist   = torch.distributions.Normal(act_mean, torch.ones_like(act_mean) * 0.3)
                action = dist.sample().clamp(-1, 1)
                logp   = dist.log_prob(action).sum()

                next_obs, reward, term, trunc, info = self.env.step(action.cpu().numpy())

                self.buffer.add(
                    obs, action.cpu().numpy(), reward,
                    float(val), float(logp), float(term or trunc),
                )

                obs        = next_obs
                ep_reward += reward
                ep_steps  += 1
                self.total_steps += 1

                if term or trunc:
                    bonus = compute_episode_bonus(
                        total_steps=ep_steps,
                        survived=not info.get("collision", False),
                    )
                    ep_reward += bonus
                    self.ep_rewards.append(ep_reward)
                    self.ep_lengths.append(ep_steps)
                    advanced = self.curriculum.record_episode_reward(ep_reward)

                    outcome = "CRASH" if info.get("collision") else ("GOAL" if info.get("goal_reached") else "timeout")
                    print(
                        f"  ep#{len(self.ep_rewards):>4d} | "
                        f"steps={ep_steps:>3d} | "
                        f"reward={ep_reward:>8.2f} | "
                        f"outcome={outcome:<8} | "
                        f"stage={self.curriculum.current_stage} | "
                        f"total_steps={self.total_steps}",
                        flush=True,
                    )

                    obs, _ = self.env.reset()
                    ep_reward = 0.0
                    ep_steps  = 0

            # ── PPO update ───────────────────────────────────────────────────
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                _, last_val = self.policy(obs_t.unsqueeze(0))
            self.buffer.compute_returns(float(last_val), self.gamma, self.gae_lambda)

            self.policy.train()
            self._ppo_update()
            self.n_updates += 1
            self.scheduler.step()

            elapsed = time.time() - t0
            sps     = self.total_steps / max(elapsed, 1)
            mean_r  = np.mean(self.ep_rewards) if self.ep_rewards else 0.0
            mean_l  = np.mean(self.ep_lengths) if self.ep_lengths else 0.0
            print(
                f"\n[PPO update #{self.n_updates}] "
                f"step={self.total_steps}  "
                f"mean_reward={mean_r:.2f}  "
                f"mean_ep_len={mean_l:.0f}  "
                f"stage={self.curriculum.current_stage}  "
                f"sps={sps:.0f}\n",
                flush=True,
            )

            # ── Checkpoint ───────────────────────────────────────────────────
            if self.n_updates % 50 == 0:
                ckpt = self.save_dir / f"policy_step{self.total_steps}_stage{self.curriculum.current_stage}.pt"
                torch.save({
                    "step":   self.total_steps,
                    "stage":  self.curriculum.current_stage,
                    "policy": self.policy.state_dict(),
                    "optim":  self.optimizer.state_dict(),
                }, ckpt)
                print(f"[PPO] Saved checkpoint → {ckpt}")

    # ── PPO update pass — identical to openenv ─────────────────────────────────

    def _ppo_update(self):
        obs      = self.buffer.obs
        acts     = self.buffer.acts
        old_logp = self.buffer.logp
        adv      = self.buffer.ret - self.buffer.val
        adv      = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret      = self.buffer.ret
        old_val  = self.buffer.val

        indices = torch.randperm(self.n_steps, device=self.device)

        for _ in range(self.n_epochs):
            for start in range(0, self.n_steps, self.batch_size):
                idx = indices[start: start + self.batch_size]

                act_mean, val = self.policy(obs[idx])
                val = val.squeeze(-1)

                dist    = torch.distributions.Normal(act_mean, torch.ones_like(act_mean) * 0.3)
                logp    = dist.log_prob(acts[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio    = torch.exp(logp - old_logp[idx])
                pg_loss1 = -adv[idx] * ratio
                pg_loss2 = -adv[idx] * ratio.clamp(1 - self.clip, 1 + self.clip)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                val_unclipped = (val - ret[idx]) ** 2
                val_clipped   = (
                    old_val[idx]
                    + (val - old_val[idx]).clamp(-self.clip_vf, self.clip_vf)
                    - ret[idx]
                ) ** 2
                vf_loss = 0.5 * torch.max(val_unclipped, val_clipped).mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.optimizer.step()


# ── Entry point ────────────────────────────────────────────────────────────────

def run_training(
    policy_type: str = "attention",
    total_steps: int = 2_000_000,
    start_stage: int = 1,
    checkpoint:  Optional[str] = None,
    device:      str = "auto",
) -> None:
    from ..policies.ticket_attention_policy import TicketAttentionPolicy
    from ..policies.flat_mlp_policy         import FlatMLPPolicy

    policy_map = {
        "attention": lambda: TicketAttentionPolicy(obs_dim=OBS_DIM),
        "mlp":       lambda: FlatMLPPolicy(obs_dim=OBS_DIM),
    }
    policy = policy_map[policy_type]()

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        policy.load_state_dict(ckpt["policy"])
        print(f"[PPO] Loaded checkpoint from {checkpoint}")

    env = OverflowGymEnv()
    cm  = CurriculumManager()
    if start_stage > 1:
        cm.force_stage(start_stage)

    trainer = PPOTrainer(policy=policy, env=env, curriculum=cm, device=device, n_steps=512)
    trainer.train(total_steps=total_steps)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--policy",     default="attention", choices=["attention", "mlp"])
    p.add_argument("--steps",      default=2_000_000,  type=int)
    p.add_argument("--stage",      default=1,           type=int)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device",     default="auto")
    args = p.parse_args()
    run_training(args.policy, args.steps, args.stage, args.checkpoint, args.device)
