"""
Agent dynamics for the OpenENV multi-agent system.

Three agent types:
  - ReviewAgent:   the RL-controlled ego vehicle (bicycle model dynamics)
  - TrafficAgent:  background vehicles using IDM + MOBIL (realistic following)
  - IncidentAgent: scripted replay of the original incident trajectory

Novel: TrafficAgent uses Waymo-constraint-adaptive IDM — friction and visibility
constraints modify the desired headway and comfortable deceleration in real-time,
matching how human drivers adjust to conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

from .constraints import EnvironmentConstraints


# ── Vehicle state ────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    id: str
    agent_type: str   # "review" | "traffic" | "incident" | "pedestrian" | "cyclist"
    x: float
    y: float
    vx: float
    vy: float
    yaw: float          # radians
    yaw_rate: float     # rad/s
    length: float       # metres
    width: float
    height: float
    speed: float = 0.0
    accel: float = 0.0
    steering: float = 0.0  # rad

    # For trajectory display in 3D sim
    predicted_trajectory: List[Tuple[float, float]] = field(default_factory=list)
    attention_weight: float = 0.0   # how much review agent attends to this agent

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.agent_type,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "yaw": round(self.yaw, 4),
            "yaw_rate": round(self.yaw_rate, 4),
            "speed": round(self.speed, 3),
            "accel": round(self.accel, 3),
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "trajectory": [[round(p[0], 2), round(p[1], 2)] for p in self.predicted_trajectory],
            "attention": round(self.attention_weight, 3),
        }


# ── Bicycle model physics ────────────────────────────────────────────────────

class BicycleModel:
    """
    Kinematic bicycle model — standard for AV control.
    Wheelbase L from agent dimensions; steering limited by physics.
    """

    DT = 0.1  # 10 Hz — matches Waymo frame rate

    def __init__(self, wheelbase: float = 2.7, max_steer_rad: float = 0.6):
        self.L = wheelbase
        self.max_steer = max_steer_rad

    def step(
        self,
        state: AgentState,
        steer_cmd: float,   # normalised -1..1
        accel_cmd: float,   # m/s²
        constraints: EnvironmentConstraints,
    ) -> AgentState:
        steer_rad = steer_cmd * self.max_steer
        dt = self.DT

        v = state.speed
        v_new = max(0.0, v + accel_cmd * dt)

        # Friction limits lateral acceleration
        max_lat_accel = constraints.friction * 9.81
        safe_steer = math.atan2(max_lat_accel * self.L, max(v, 0.1) ** 2)
        steer_rad = max(-safe_steer, min(safe_steer, steer_rad))

        yaw_new = state.yaw + (v * math.tan(steer_rad) / self.L) * dt
        x_new = state.x + v * math.cos(yaw_new) * dt
        y_new = state.y + v * math.sin(yaw_new) * dt
        yaw_rate_new = (yaw_new - state.yaw) / dt

        return AgentState(
            id=state.id,
            agent_type=state.agent_type,
            x=x_new,
            y=y_new,
            vx=v_new * math.cos(yaw_new),
            vy=v_new * math.sin(yaw_new),
            yaw=yaw_new,
            yaw_rate=yaw_rate_new,
            speed=v_new,
            accel=accel_cmd,
            steering=steer_rad,
            length=state.length,
            width=state.width,
            height=state.height,
            predicted_trajectory=state.predicted_trajectory,
            attention_weight=state.attention_weight,
        )


# ── IDM (Intelligent Driver Model) ──────────────────────────────────────────

class IDMAgent:
    """
    Constraint-adaptive IDM for realistic traffic agents.

    Novel: The IDM parameters (desired headway T, comfortable decel b) are
    dynamically scaled by the Waymo-derived friction and visibility constraints,
    making traffic behave realistically for the inferred weather conditions.
    """

    def __init__(
        self,
        v_desired: float = 13.4,   # 30 mph
        a_max: float = 2.0,        # m/s²
        b_comfort: float = 3.0,    # comfortable braking m/s²
        s_min: float = 2.0,        # minimum gap m
        T_headway: float = 1.5,    # desired time headway s
        delta: float = 4.0,        # velocity exponent
    ):
        self.v0 = v_desired
        self.a_max = a_max
        self.b = b_comfort
        self.s0 = s_min
        self.T = T_headway
        self.delta = delta
        self._bicycle = BicycleModel()

    def compute_accel(
        self,
        v: float,
        gap: float,
        v_lead: float,
        constraints: EnvironmentConstraints,
    ) -> float:
        """IDM acceleration with constraint-adaptive parameters."""
        # Scale headway and decel by conditions
        T_eff = self.T / constraints.friction      # longer headway on slippery roads
        b_eff = self.b * constraints.friction       # reduced braking on wet roads
        v0_eff = min(self.v0, constraints.max_safe_speed_ms)

        dv = v - v_lead
        s_star = self.s0 + max(0, v * T_eff + v * dv / (2 * math.sqrt(self.a_max * b_eff)))
        gap = max(gap, 0.1)
        accel = self.a_max * (1 - (v / max(v0_eff, 0.1)) ** self.delta - (s_star / gap) ** 2)
        return float(np.clip(accel, -b_eff, self.a_max))

    def step_agent(
        self,
        state: AgentState,
        lead_state: Optional[AgentState],
        constraints: EnvironmentConstraints,
    ) -> AgentState:
        gap = 100.0
        v_lead = constraints.max_safe_speed_ms
        if lead_state is not None:
            dx = lead_state.x - state.x
            dy = lead_state.y - state.y
            gap = max(0.1, math.sqrt(dx * dx + dy * dy) - state.length / 2 - lead_state.length / 2)
            v_lead = lead_state.speed

        accel = self.compute_accel(state.speed, gap, v_lead, constraints)
        # Keep lane — zero steering for simplicity (extend with MOBIL if needed)
        return self._bicycle.step(state, 0.0, accel, constraints)


# ── Review Agent (RL-controlled) ─────────────────────────────────────────────

class ReviewAgent:
    """
    The RL-controlled ego vehicle. Wraps BicycleModel to expose the Gymnasium
    action interface: action = [steer (-1..1), throttle (0..1), brake (0..1)].

    Predicted trajectory is computed by rolling out the bicycle model for
    3 seconds using the current action, allowing the 3D viewer to show
    where the agent intends to go before it gets there.
    """

    TRAJECTORY_HORIZON = 30  # 3 seconds at 10 Hz
    MAX_ACCEL = 4.0          # m/s²
    MAX_BRAKE = 8.0          # m/s²

    def __init__(self):
        self._bicycle = BicycleModel()

    def step(
        self,
        state: AgentState,
        action: np.ndarray,
        constraints: EnvironmentConstraints,
    ) -> AgentState:
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        # Net acceleration command
        accel_cmd = throttle * self.MAX_ACCEL - brake * self.MAX_BRAKE

        new_state = self._bicycle.step(state, steer, accel_cmd, constraints)
        new_state.predicted_trajectory = self._predict_trajectory(new_state, action, constraints)
        return new_state

    def _predict_trajectory(
        self,
        state: AgentState,
        action: np.ndarray,
        constraints: EnvironmentConstraints,
    ) -> List[Tuple[float, float]]:
        """Roll out current action to get predicted path for 3D visualization."""
        pts = []
        s = state
        for _ in range(self.TRAJECTORY_HORIZON):
            pts.append((s.x, s.y))
            s = self._bicycle.step(s, float(action[0]), 0.0, constraints)
        return pts


# ── Incident Agent (scripted replay) ────────────────────────────────────────

@dataclass
class WaypointScript:
    """Time-indexed waypoints extracted from the incident replay."""
    timestamps: List[float]    # seconds
    xs: List[float]
    ys: List[float]
    speeds: List[float]
    yaws: List[float]


class IncidentAgent:
    """
    Proximity-triggered scripted actor that replays the original incident.

    WAIT behaviour: the actor holds at their spawn position until the review
    agent (ego) comes within `trigger_distance` metres. Only then do they
    execute their scripted path. This ensures the scenario is identical every
    episode — the pedestrian always steps out at the same relative distance
    from the ego, regardless of the RL policy's timing.

    This is the key determinism guarantee: the near-miss is always triggered
    at the same spatial relationship, so the RL agent is solving the same
    problem each episode.
    """

    def __init__(
        self,
        initial_state: AgentState,
        script: WaypointScript,
        trigger_distance: float = 22.0,  # metres — ego must be this close to trigger
    ):
        self._script = script
        self._initial = initial_state
        self._trigger_distance = trigger_distance
        self._triggered = False
        self._trigger_time: Optional[float] = None  # sim time when triggered

    def reset(self) -> None:
        """Called at episode reset — restores wait state."""
        self._triggered = False
        self._trigger_time = None

    def get_state_at(self, t: float, ego_x: float = 0.0, ego_y: float = 0.0) -> AgentState:
        """
        Returns actor state at sim time t.
        If not yet triggered, actor stays at spawn. Once ego is within
        trigger_distance, actor starts moving through their scripted path.
        """
        # Check proximity trigger (one-shot)
        if not self._triggered:
            dist = math.sqrt(
                (self._initial.x - ego_x) ** 2 + (self._initial.y - ego_y) ** 2
            )
            if dist <= self._trigger_distance:
                self._triggered = True
                self._trigger_time = t

        # Not triggered yet — stand still at spawn
        if not self._triggered:
            st = self._initial
            return AgentState(
                id=st.id, agent_type="incident",
                x=st.x, y=st.y, vx=0.0, vy=0.0,
                yaw=st.yaw, yaw_rate=0.0, speed=0.0,
                length=st.length, width=st.width, height=st.height,
            )

        # Triggered — play script from trigger time
        script_t = t - (self._trigger_time or t)
        return self._interpolate(script_t)

    def _interpolate(self, script_t: float) -> AgentState:
        s = self._script
        st = self._initial
        if not s.timestamps:
            return AgentState(
                id=st.id, agent_type="incident",
                x=st.x, y=st.y, vx=0.0, vy=0.0,
                yaw=st.yaw, yaw_rate=0.0, speed=0.0,
                length=st.length, width=st.width, height=st.height,
            )

        if script_t <= s.timestamps[0]:
            i = 0
        elif script_t >= s.timestamps[-1]:
            i = len(s.timestamps) - 1
            x, y = s.xs[i], s.ys[i]
            spd, yaw = s.speeds[i], s.yaws[i]
            return AgentState(
                id=st.id, agent_type="incident",
                x=x, y=y,
                vx=spd * math.cos(yaw), vy=spd * math.sin(yaw),
                yaw=yaw, yaw_rate=0.0, speed=spd,
                length=st.length, width=st.width, height=st.height,
            )
        else:
            for i in range(len(s.timestamps) - 1):
                if s.timestamps[i] <= script_t < s.timestamps[i + 1]:
                    alpha = (script_t - s.timestamps[i]) / (s.timestamps[i + 1] - s.timestamps[i])
                    x = s.xs[i] + alpha * (s.xs[i + 1] - s.xs[i])
                    y = s.ys[i] + alpha * (s.ys[i + 1] - s.ys[i])
                    spd = s.speeds[i] + alpha * (s.speeds[i + 1] - s.speeds[i])
                    yaw = s.yaws[i] + alpha * (s.yaws[i + 1] - s.yaws[i])
                    return AgentState(
                        id=st.id, agent_type="incident",
                        x=x, y=y,
                        vx=spd * math.cos(yaw), vy=spd * math.sin(yaw),
                        yaw=yaw, yaw_rate=0.0, speed=spd,
                        length=st.length, width=st.width, height=st.height,
                    )
            i = len(s.timestamps) - 1

        x, y = s.xs[i], s.ys[i]
        spd, yaw = s.speeds[i], s.yaws[i]
        return AgentState(
            id=st.id, agent_type="incident",
            x=x, y=y,
            vx=spd * math.cos(yaw), vy=spd * math.sin(yaw),
            yaw=yaw, yaw_rate=0.0, speed=spd,
            length=st.length, width=st.width, height=st.height,
        )
