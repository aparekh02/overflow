"""
Gymnasium wrapper around OverflowEnvironment.

Bridges the gap between OverflowEnvironment (text actions, structured obs)
and our PPO trainer (continuous actions, numeric obs vector).

Observation: 603-dim float32 vector (ego state + per-car incident tickets)

Action:      [steer, throttle, brake] all in [-1, 1]
             → mapped to text decision for OverflowEnvironment

Reward:      incident-response grading — every step classifies the scene
             into an incident type and scores the action against it.
             This produces large, directional reward signals rather than
             diffuse per-step shaping.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..server.overflow_environment import OverflowEnvironment
from ..models import OverflowAction
from ..policies.policy_spec import build_obs, build_ticket_vector, OBS_DIM
from .reward import compute_reward, compute_episode_bonus, IncidentType


# ── Action mapping ────────────────────────────────────────────────────────────

def _action_to_decision(action: np.ndarray) -> str:
    steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
    if abs(steer) > 0.35:
        return "lane_change_left" if steer < 0 else "lane_change_right"
    if brake > 0.25:
        return "brake"
    if throttle > 0.20:
        return "accelerate"
    return "maintain"


# ── Observation extraction ────────────────────────────────────────────────────

def _obs_to_vector(overflow_obs) -> np.ndarray:
    """OverflowObservation → 603-dim numpy vector matching policy_spec layout."""
    cars = overflow_obs.cars
    if not cars:
        return np.zeros(OBS_DIM, dtype=np.float32)

    ego = next((c for c in cars if c.carId == 0), cars[0])
    ego_speed_ms = ego.speed / 4.5
    ego_x        = ego.position.x
    ego_y        = (ego.lane - 2) * 3.7

    ticket_vectors = []
    for car in cars:
        if car.carId == 0:
            continue
        rel_x    = car.position.x - ego_x
        rel_y    = (car.lane - ego.lane) * 3.7
        car_spd  = car.speed / 4.5
        distance = math.sqrt(rel_x ** 2 + rel_y ** 2)
        if distance > 80:
            continue
        closing  = max(ego_speed_ms - car_spd * math.copysign(1, max(rel_x, 0.01)), 0.1)
        ttc      = min(distance / closing, 30.0)
        severity = 1.0 if distance < 8 else (0.75 if distance < 15 else 0.5)
        ticket_vectors.append(build_ticket_vector(
            severity_weight=severity, ttl=5.0,
            pos_x=rel_x, pos_y=rel_y, pos_z=0.0,
            vel_x=car_spd, vel_y=0.0, vel_z=0.0,
            heading=0.0,
            size_length=4.0, size_width=2.0, size_height=1.5,
            distance=distance, time_to_collision=ttc,
            bearing=math.atan2(rel_y, max(rel_x, 0.01)),
            ticket_type="collision_risk", entity_type="vehicle", confidence=1.0,
        ))

    tv = np.array(ticket_vectors, dtype=np.float32) if ticket_vectors else None
    return build_obs(
        ego_x=ego_x, ego_y=ego_y, ego_z=0.0,
        ego_vx=ego_speed_ms, ego_vy=0.0,
        heading=0.0, speed=ego_speed_ms,
        steer=0.0, throttle=0.5, brake=0.0,
        ticket_vectors=tv,
    )


# ── Gymnasium wrapper ─────────────────────────────────────────────────────────

class OverflowGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around OverflowEnvironment.
    Uses incident-response grading for rewards.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self._env = OverflowEnvironment()
        self._last_overflow_obs = None
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._sim_time = 0.0
        self._step_dt  = 0.1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._last_overflow_obs = self._env.reset(seed=seed)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._sim_time = 0.0
        return _obs_to_vector(self._last_overflow_obs), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        decision = _action_to_decision(action)
        overflow_action = OverflowAction(decision=decision, reasoning="")
        overflow_obs = self._env.step(overflow_action)
        self._last_overflow_obs = overflow_obs
        self._sim_time += self._step_dt

        obs_vec = _obs_to_vector(overflow_obs)

        # Extract ego state
        ego = next((c for c in overflow_obs.cars if c.carId == 0), None)
        ego_speed_ms = (ego.speed / 4.5) if ego else 0.0
        ego_x        = ego.position.x if ego else 0.0
        ego_lane     = ego.lane if ego else 2
        ego_y        = (ego_lane - 2) * 3.7

        collision    = any(
            "CRASH" in line and "Car 0" in line
            for line in (overflow_obs.incident_report or "").split("\n")
        )
        goal_reached = overflow_obs.done and not collision

        # Goal position — read from env internals (Car 0's goal_position)
        goal_x = 180.0  # default fallback
        if hasattr(self._env, "_cars") and self._env._cars:
            agent_car = next((c for c in self._env._cars if c.car_id == 0), None)
            if agent_car:
                goal_x = agent_car.goal_position

        reward, incident_ctx, grade_desc = compute_reward(
            ego_speed    = ego_speed_ms,
            ego_x        = ego_x,
            ego_lane     = ego_lane,
            ego_y        = ego_y,
            decision     = decision,
            action       = action,
            prev_action  = self._prev_action,
            collision    = collision,
            goal_reached = goal_reached,
            cars         = overflow_obs.cars,
            goal_x       = goal_x,
        )

        self._prev_action = action.copy()

        terminated = overflow_obs.done
        truncated  = False
        info: Dict[str, Any] = {
            "collision":      collision,
            "goal_reached":   goal_reached,
            "incident":       overflow_obs.incident_report,
            "incident_type":  incident_ctx.incident_type.value,
            "decision":       decision,
            "grade_desc":     grade_desc,
            "reward":         reward,
        }

        return obs_vec, reward, terminated, truncated, info
