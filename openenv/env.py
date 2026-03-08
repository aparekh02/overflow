"""
WaymoEnv — Gymnasium environment bridging Waymo 3D simulation with RL.

Novel architecture: Unlike typical RL driving envs (CARLA, MetaDrive) that
simulate physics independently, WaymoEnv maintains a *live bidirectional
connection* to the React/Three.js 3D simulator via the middleware layer.
The 3D sim is not a passive viewer — it's an active part of the training loop:

  ┌─────────────────────────────────────────────────────────────┐
  │  WaymoEnv.step(action)                                      │
  │    ↓ simulate physics                                       │
  │    ↓ update constraints from Waymo data                     │
  │    ↓ push state → Middleware → 3D sim (live visualization)  │
  │    ↓ receive user annotations ← 3D sim (counterfactuals)    │
  │    → return (obs, reward, terminated, truncated, info)      │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .agents import AgentState, ReviewAgent, IDMAgent, IncidentAgent
from .constraints import EnvironmentConstraints, WaymoConstraints
from .incident_parser import IncidentParser, IncidentTicket, InitialState
from .reward import RewardEngine, RewardComponents


# ── Observation space dimensions ─────────────────────────────────────────────

EGO_DIM     = 6    # x, y, vx, vy, yaw, yaw_rate (normalised)
AGENT_DIM   = 7    # rel_x, rel_y, rel_vx, rel_vy, rel_yaw, type_enc, ttc
MAX_AGENTS  = 10   # nearest N agents included in obs
CONSTRAINT_DIM = 5 # friction, visibility, rain, fog, traffic_density
TIME_DIM    = 1    # step progress

OBS_DIM = EGO_DIM + MAX_AGENTS * AGENT_DIM + CONSTRAINT_DIM + TIME_DIM  # = 82


class WaymoEnv(gym.Env):
    """
    Gymnasium-compatible RL environment driven by Waymo scene data.

    Observation space:  Box(82,) — ego state, nearest agents, constraints, time
    Action space:       Box(3,)  — [steer(-1..1), throttle(0..1), brake(0..1)]

    Episode lifecycle:
      reset(options={"ticket": <IncidentTicket>})  → seeds from incident
      step(action)                                  → advances 0.1s (10 Hz)
      close()                                       → shuts down middleware
    """

    metadata = {"render_modes": ["websocket", "none"]}

    def __init__(
        self,
        render_mode: str = "none",
        max_steps: int = 200,
        api_key: Optional[str] = None,
        waymo_frame_data: Optional[dict] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self._waymo_frame_data = waymo_frame_data

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32,
        )

        # Components
        self._parser = IncidentParser(api_key=api_key)
        self._review_agent = ReviewAgent()
        self._reward_engine = RewardEngine()
        self._middleware: Optional[Any] = None  # set by SimMiddleware

        # Episode state
        self._ego: Optional[AgentState] = None
        self._agents: List[AgentState] = []
        self._incident_agents: List[IncidentAgent] = []
        self._idm_controllers: Dict[str, IDMAgent] = {}
        self._constraints: EnvironmentConstraints = EnvironmentConstraints()
        self._initial_state: Optional[InitialState] = None
        self._step: int = 0
        self._episode_reward: float = 0.0
        self._last_reward_components: Optional[RewardComponents] = None

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step = 0
        self._episode_reward = 0.0
        self._reward_engine.reset()

        ticket: Optional[IncidentTicket] = options.get("ticket") if options else None

        if ticket is not None:
            init = self._parser.parse(ticket)
        elif self._initial_state is not None:
            # Replay same scenario (for counterfactual exploration)
            init = self._initial_state
        else:
            # Default: random scenario from templates
            import random
            scenario_types = list(self._parser.SCENARIO_TEMPLATES.keys())
            ticket = IncidentTicket(
                ticket_id="default",
                raw_text=random.choice(scenario_types),
            )
            init = self._parser.parse(ticket)

        self._initial_state = init
        self._constraints = init.constraints

        # Override constraints from real Waymo data if available
        if self._waymo_frame_data:
            self._constraints = self._update_constraints_from_waymo(self._waymo_frame_data)

        # Spawn agents
        self._ego = init.get_ego_state()
        all_actor_states = init.get_actor_states()

        self._agents = [s for s in all_actor_states if s.agent_type == "traffic"]
        self._idm_controllers = {s.id: IDMAgent(v_desired=self._constraints.max_safe_speed_ms)
                                  for s in self._agents}

        # Incident agents use scripted replay
        self._incident_agents = []
        if init.incident_script is not None:
            incident_actor_states = [s for s in all_actor_states if s.agent_type == "incident"]
            for s in incident_actor_states:
                self._incident_agents.append(IncidentAgent(s, init.incident_script, trigger_distance=22.0))
        else:
            # Reset existing incident agents for deterministic replay
            for ia in self._incident_agents:
                ia.reset()

        obs = self._make_obs()
        info = self._make_info()

        if self._middleware is not None:
            asyncio.ensure_future(
                self._middleware.broadcast_state(self._serialize_state(0.0, False, info))
            )

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._ego is not None, "Call reset() before step()"

        self._step += 1
        t = self._step * 0.1  # seconds

        # 1. Update constraints (can change if Waymo data is streaming)
        if self._waymo_frame_data:
            self._constraints = self._update_constraints_from_waymo(self._waymo_frame_data)

        # 2. Step review agent
        self._ego = self._review_agent.step(self._ego, action, self._constraints)

        # 3. Step traffic agents (IDM)
        new_agent_states = []
        for ag in self._agents:
            lead = self._find_lead(ag, self._agents + [self._ego])
            controller = self._idm_controllers.get(ag.id)
            if controller:
                new_agent_states.append(controller.step_agent(ag, lead, self._constraints))
            else:
                new_agent_states.append(ag)
        self._agents = new_agent_states

        # 4. Update incident agents — proximity-triggered by ego position
        incident_states = []
        for ia in self._incident_agents:
            incident_states.append(
                ia.get_state_at(t, ego_x=self._ego.x, ego_y=self._ego.y)
            )

        # All agents for reward/obs
        all_agents = self._agents + incident_states

        # 5. Compute attention weights (how much review agent attends to each agent)
        self._update_attention_weights(all_agents)

        # 6. Reward
        incident_avoided = len(incident_states) > 0 and self._step >= self.max_steps - 1
        rc = self._reward_engine.compute(
            self._ego, all_agents, self._constraints,
            self._step, self.max_steps, incident_avoided,
        )
        self._last_reward_components = rc
        reward = rc.total
        self._episode_reward += reward

        # 7. Termination
        terminated = rc.collision < -50.0 or self._ego_out_of_bounds()
        truncated = self._step >= self.max_steps

        obs = self._make_obs()
        info = self._make_info(rc=rc, incident_states=incident_states)

        # 8. Broadcast to 3D sim
        if self._middleware is not None:
            asyncio.ensure_future(
                self._middleware.broadcast_state(
                    self._serialize_state(reward, terminated or truncated, info, incident_states)
                )
            )

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._middleware:
            asyncio.ensure_future(self._middleware.shutdown())

    def attach_middleware(self, middleware: Any) -> None:
        """Called by SimMiddleware to register itself."""
        self._middleware = middleware

    def update_waymo_frame(self, frame_data: dict) -> None:
        """Called when a new Waymo parquet frame is available (streaming mode)."""
        self._waymo_frame_data = frame_data
        self._constraints = self._update_constraints_from_waymo(frame_data)

    # ── Observation builder ───────────────────────────────────────────────────

    def _make_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        i = 0

        # Ego state (normalised)
        ego = self._ego
        obs[i:i+6] = [
            ego.x / 100.0,
            ego.y / 50.0,
            ego.vx / 30.0,
            ego.vy / 30.0,
            ego.yaw / math.pi,
            ego.yaw_rate / 2.0,
        ]
        i += 6

        # Nearest agents
        ego = self._ego
        all_agents = self._agents + [
            ia.get_state_at(self._step * 0.1, ego_x=ego.x if ego else 0.0, ego_y=ego.y if ego else 0.0)
            for ia in self._incident_agents
        ]
        near = self._k_nearest(ego, all_agents, MAX_AGENTS)
        type_enc_map = {"traffic": 0.0, "incident": 1.0, "pedestrian": 0.5, "cyclist": 0.75}

        for j in range(MAX_AGENTS):
            if j < len(near):
                ag = near[j]
                dx = ag.x - ego.x
                dy = ag.y - ego.y
                ttc = self._ttc(ego, ag)
                obs[i:i+7] = [
                    dx / 100.0,
                    dy / 50.0,
                    (ag.vx - ego.vx) / 15.0,
                    (ag.vy - ego.vy) / 15.0,
                    (ag.yaw - ego.yaw) / math.pi,
                    type_enc_map.get(ag.agent_type, 0.0),
                    min(1.0, ttc / 10.0),
                ]
            i += 7

        # Constraints
        obs[i:i+5] = self._constraints.to_obs_vector()
        i += 5

        # Time progress
        obs[i] = self._step / self.max_steps
        return obs

    def _make_info(
        self,
        rc: Optional[RewardComponents] = None,
        incident_states: Optional[List[AgentState]] = None,
    ) -> dict:
        info: dict = {
            "step": self._step,
            "episode_reward": self._episode_reward,
            "ego_speed": self._ego.speed if self._ego else 0,
            "constraints": self._constraints.__dict__.copy(),
            "incident_type": self._initial_state.incident_type if self._initial_state else "unknown",
            "ticket_id": self._initial_state.ticket_id if self._initial_state else "",
        }
        if rc:
            info["reward_components"] = rc.to_dict()
        return info

    # ── State serialisation (for middleware → 3D sim) ─────────────────────────

    def _serialize_state(
        self,
        reward: float,
        done: bool,
        info: dict,
        incident_states: Optional[List[AgentState]] = None,
    ) -> dict:
        """Converts current episode state to JSON-serializable dict for the 3D sim."""
        agents_out = []
        for ag in self._agents:
            agents_out.append(ag.to_dict())
        for ag in (incident_states or []):
            agents_out.append(ag.to_dict())

        ego_dict = self._ego.to_dict() if self._ego else {}

        return {
            "type": "scene_update",
            "state": {
                "step": self._step,
                "timestamp": round(self._step * 0.1, 2),
                "ego": ego_dict,
                "agents": agents_out,
                "constraints": {
                    "weather": self._constraints.weather,
                    "friction": round(self._constraints.friction, 3),
                    "visibility_m": round(self._constraints.visibility_m, 1),
                    "traffic_density": round(self._constraints.traffic_density, 3),
                    "speed_limit_ms": round(self._constraints.speed_limit_ms, 2),
                },
                "reward": round(reward, 4),
                "episode_reward": round(self._episode_reward, 4),
                "done": done,
                "incident_type": info.get("incident_type", ""),
                "ticket_id": info.get("ticket_id", ""),
                "reward_components": info.get("reward_components", {}),
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_constraints_from_waymo(self, frame: dict) -> EnvironmentConstraints:
        """Extract live constraints from a Waymo frame dict."""
        return WaymoConstraints.from_waymo_frame(
            point_count=frame.get("pointCount", WaymoConstraints.BASELINE_POINT_COUNT),
            avg_intensity=frame.get("avgIntensity", 0.5),
            num_agents=len(frame.get("boxes", [])),
            weather_hint=frame.get("weather"),
        )

    def _k_nearest(
        self,
        ego: AgentState,
        agents: List[AgentState],
        k: int,
    ) -> List[AgentState]:
        def dist(a: AgentState) -> float:
            return math.sqrt((a.x - ego.x) ** 2 + (a.y - ego.y) ** 2)
        return sorted(agents, key=dist)[:k]

    def _find_lead(
        self,
        ego: AgentState,
        agents: List[AgentState],
    ) -> Optional[AgentState]:
        """Find closest agent ahead in the ego's heading direction."""
        cos_h = math.cos(ego.yaw)
        sin_h = math.sin(ego.yaw)
        best: Optional[AgentState] = None
        best_d = float("inf")
        for ag in agents:
            if ag.id == ego.id:
                continue
            dx = ag.x - ego.x
            dy = ag.y - ego.y
            # Dot product along heading
            ahead = dx * cos_h + dy * sin_h
            lateral = abs(-dx * sin_h + dy * cos_h)
            if ahead > 0 and lateral < 3.0:  # within lane width
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_d:
                    best_d = d
                    best = ag
        return best

    def _ttc(self, ego: AgentState, ag: AgentState) -> float:
        dx = ag.x - ego.x
        dy = ag.y - ego.y
        dvx = ag.vx - ego.vx
        dvy = ag.vy - ego.vy
        dot = dx * dvx + dy * dvy
        if dot >= 0:
            return float("inf")
        dist = math.sqrt(dx * dx + dy * dy)
        closing = abs(dot) / max(dist, 0.1)
        return dist / max(closing, 0.01)

    def _update_attention_weights(self, agents: List[AgentState]) -> None:
        """Softmax-weighted attention over inverse TTC — shown in 3D sim."""
        if not agents or not self._ego:
            return
        ttcs = np.array([self._ttc(self._ego, ag) for ag in agents])
        # High attention to close/fast-approaching agents
        scores = np.clip(10.0 / (ttcs + 0.1), 0, 10)
        weights = scores / (scores.sum() + 1e-8)
        for ag, w in zip(agents, weights):
            ag.attention_weight = float(w)

    def _ego_out_of_bounds(self) -> bool:
        if not self._ego:
            return True
        return abs(self._ego.x) > 200 or abs(self._ego.y) > 100
