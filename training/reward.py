"""
Reward shaping for OverflowEnvironment — ported from openenv/training/reward.py.

Same core principle: BASE + THREAT_RESPONSE with clear gradient direction.
Adapted to OverflowEnvironment's signals (no EventTicket objects — uses
collision/near-miss flags and raw reward from the environment).

  BASE:           survival + speed + lane  ~+0.4/step
  COLLISION:      -50  (terminal)
  NEAR MISS:      -0.8 per event
  GOAL REACHED:   +5.0 (terminal bonus)
  SMOOTH DRIVING: small bonus when no threats
"""

from __future__ import annotations

import numpy as np

# ── Same weights as openenv/training/reward.py ────────────────────────────────

W_ALIVE         =  0.40
W_SPEED         =  0.10
W_LANE          =  0.15
W_SMOOTH        =  0.03
TARGET_SPEED    = 11.0    # m/s (~40 km/h)
TARGET_SPEED_TOL =  3.0

W_COLLISION     = -50.0
W_NEAR_MISS     =  -0.8
W_GOAL          =   5.0
W_SURVIVE_BONUS =   5.0

ROAD_HALF_WIDTH = 3.7 * 1.5   # ~2.5 lanes worth of tolerance


def compute_reward(
    ego_speed:    float,
    ego_y:        float,
    action:       np.ndarray,
    prev_action:  np.ndarray,
    collision:    bool,
    goal_reached: bool,
    near_miss:    bool,
    raw_reward:   float,       # OverflowEnvironment's built-in reward (used as baseline)
) -> float:
    """
    Shaped reward. Mirrors openenv reward structure:
      - collision → large terminal penalty
      - base survival + speed + lane keeping
      - near-miss penalty
      - goal bonus
      - smooth driving bonus when clear
    """
    if collision:
        return W_COLLISION

    reward = 0.0

    # 1. Survival
    reward += W_ALIVE

    # 2. Speed maintenance (same formula as openenv)
    speed_err = abs(ego_speed - TARGET_SPEED)
    if speed_err < TARGET_SPEED_TOL:
        reward += W_SPEED * (1.0 - speed_err / TARGET_SPEED_TOL)
    else:
        reward -= 0.03 * min(speed_err - TARGET_SPEED_TOL, 5.0)

    # 3. Lane keeping
    norm_y = abs(ego_y) / ROAD_HALF_WIDTH
    reward += W_LANE * max(0.0, 1.0 - norm_y ** 2)

    # 4. Near miss penalty
    if near_miss:
        reward += W_NEAR_MISS

    # 5. Goal bonus
    if goal_reached:
        reward += W_GOAL

    # 6. Smooth driving
    action_delta = np.abs(action - prev_action).sum()
    reward += W_SMOOTH * max(0.0, 1.0 - action_delta * 3.0)

    return float(reward)


def compute_episode_bonus(total_steps: int, survived: bool) -> float:
    """End-of-episode bonus — same as openenv."""
    if not survived:
        return 0.0
    bonus  = W_SURVIVE_BONUS
    bonus += min(total_steps, 500) * 0.02   # longevity reward
    return float(bonus)
