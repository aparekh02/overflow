"""
RewardEngine — Multi-objective reward shaping for the review agent.

Novel: The reward is constraint-adaptive — identical behaviour earns different
rewards depending on the Waymo-derived weather/road conditions. Smooth braking
in rain is rewarded more than on dry roads. This grounds the policy in real
physical risk rather than abstract game scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .agents import AgentState
from .constraints import EnvironmentConstraints


@dataclass
class RewardComponents:
    collision: float = 0.0
    comfort: float = 0.0
    efficiency: float = 0.0
    safety_margin: float = 0.0
    incident_prevention: float = 0.0
    constraint_compliance: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


class RewardEngine:
    """
    Computes per-step reward for the RL review agent.

    Weights (tunable):
      w_collision          = -100 (terminal)
      w_safety_margin      = -0.5..0  (proportional to inverse TTC)
      w_efficiency         = +0.3  (maintain ~speed limit)
      w_comfort            = -0.2  (penalise jerk / harsh steer)
      w_incident_prevention= +50   (avoided the incident scenario)
      w_constraint_compliance = +0.1 (respect weather-adjusted limits)
    """

    # Collision radii
    COLLISION_DIST_EGO_VEHICLE = 2.5    # m (half-diagonal of car)
    COLLISION_DIST_EGO_PED = 1.5        # m

    # TTC (Time-To-Collision) threshold for safety margin reward
    TTC_CRITICAL = 1.5   # s
    TTC_WARNING  = 4.0   # s

    def __init__(
        self,
        w_collision: float = -100.0,
        w_safety: float = -0.5,
        w_efficiency: float = 0.3,
        w_comfort: float = -0.2,
        w_incident_prevention: float = 50.0,
        w_constraint: float = 0.1,
    ):
        self.w_collision = w_collision
        self.w_safety = w_safety
        self.w_efficiency = w_efficiency
        self.w_comfort = w_comfort
        self.w_incident_prevention = w_incident_prevention
        self.w_constraint = w_constraint

        self._prev_steer: float = 0.0
        self._prev_accel: float = 0.0

    def reset(self) -> None:
        self._prev_steer = 0.0
        self._prev_accel = 0.0

    def compute(
        self,
        ego: AgentState,
        agents: List[AgentState],
        constraints: EnvironmentConstraints,
        episode_step: int,
        max_steps: int,
        incident_avoided: bool = False,
    ) -> RewardComponents:
        rc = RewardComponents()

        # ── Collision ─────────────────────────────────────────────────────
        collision = self._check_collision(ego, agents)
        rc.collision = self.w_collision if collision else 0.0

        # ── Safety margin (TTC-based) ──────────────────────────────────────
        min_ttc = self._min_ttc(ego, agents)
        if min_ttc < self.TTC_CRITICAL:
            # Penalise harder in bad weather (risk is higher)
            urgency = (self.TTC_CRITICAL - min_ttc) / self.TTC_CRITICAL
            rc.safety_margin = self.w_safety * urgency * (1.0 + (1.0 - constraints.friction))
        elif min_ttc < self.TTC_WARNING:
            urgency = (self.TTC_WARNING - min_ttc) / (self.TTC_WARNING - self.TTC_CRITICAL)
            rc.safety_margin = self.w_safety * 0.3 * urgency

        # ── Efficiency (speed relative to constraint-adjusted limit) ───────
        v_target = constraints.max_safe_speed_ms
        v_ratio = min(1.0, ego.speed / max(v_target, 0.1))
        rc.efficiency = self.w_efficiency * v_ratio

        # ── Comfort (steer rate + jerk) ─────────────────────────────────
        steer_rate = abs(ego.steering - self._prev_steer) / 0.1   # rad/s
        jerk = abs(ego.accel - self._prev_accel) / 0.1            # m/s³
        # Scale comfort weight by friction — abrupt inputs matter more on ice
        comfort_scale = 1.0 + (1.0 - constraints.friction) * 2.0
        rc.comfort = self.w_comfort * (steer_rate * 0.3 + jerk * 0.1) * comfort_scale
        self._prev_steer = ego.steering
        self._prev_accel = ego.accel

        # ── Incident prevention (reached end of episode without incident) ──
        if incident_avoided and episode_step >= max_steps - 1:
            rc.incident_prevention = self.w_incident_prevention

        # ── Constraint compliance (speed, hard braking in rain) ───────────
        overspeed = max(0.0, ego.speed - constraints.max_safe_speed_ms)
        rc.constraint_compliance = -self.w_constraint * overspeed if overspeed > 0 else self.w_constraint * 0.5

        # ── Total ──────────────────────────────────────────────────────────
        rc.total = (
            rc.collision + rc.safety_margin + rc.efficiency
            + rc.comfort + rc.incident_prevention + rc.constraint_compliance
        )
        return rc

    # ── Private helpers ──────────────────────────────────────────────────────

    def _dist(self, a: AgentState, b: AgentState) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _check_collision(self, ego: AgentState, agents: List[AgentState]) -> bool:
        for ag in agents:
            d = self._dist(ego, ag)
            threshold = (
                self.COLLISION_DIST_EGO_PED
                if ag.agent_type in ("pedestrian", "cyclist")
                else self.COLLISION_DIST_EGO_VEHICLE
            )
            if d < threshold:
                return True
        return False

    def _min_ttc(self, ego: AgentState, agents: List[AgentState]) -> float:
        min_ttc = float("inf")
        for ag in agents:
            dx = ag.x - ego.x
            dy = ag.y - ego.y
            dvx = ag.vx - ego.vx
            dvy = ag.vy - ego.vy
            dot = dx * dvx + dy * dvy
            if dot >= 0:
                continue  # not approaching
            dist = math.sqrt(dx * dx + dy * dy)
            closing_speed = abs(dot) / max(dist, 0.1)
            ttc = dist / max(closing_speed, 0.01)
            min_ttc = min(min_ttc, ttc)
        return min_ttc
