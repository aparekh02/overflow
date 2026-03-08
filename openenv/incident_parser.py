"""
IncidentParser — converts natural-language incident tickets into structured
InitialState objects that seed the WaymoEnv.

Two modes:
  1. LLM-powered (Claude API) — full semantic understanding
  2. Rule-based fallback — regex extraction when no API key

Novel: The ticket becomes the environment's initial condition distribution,
ensuring the RL agent trains on realistic, historically-grounded scenarios
rather than random synthetic states. Each ticket produces a deterministic
scene that can be deterministically replayed or counterfactually explored.
"""

from __future__ import annotations

import math
import re
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .agents import AgentState, WaypointScript
from .constraints import EnvironmentConstraints, WaymoConstraints


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class IncidentTicket:
    """Raw incident ticket from an operator or report system."""
    ticket_id: str
    raw_text: str
    timestamp_utc: str = ""
    severity: str = "moderate"   # minor | moderate | severe | critical


@dataclass
class InvolvedActor:
    actor_id: str
    actor_type: str    # vehicle | pedestrian | cyclist
    initial_x: float
    initial_y: float
    initial_speed_ms: float
    initial_yaw_rad: float
    is_incident_agent: bool = False   # True = the "at-fault" agent to avoid
    size: tuple = (4.5, 2.0, 1.6)    # length, width, height


@dataclass
class InitialState:
    """Fully structured initial environment state from a parsed ticket."""
    ticket_id: str
    ego_x: float
    ego_y: float
    ego_speed_ms: float
    ego_yaw_rad: float
    actors: List[InvolvedActor]
    constraints: EnvironmentConstraints
    incident_type: str               # jaywalker | rear_end | intersection | lane_change | sideswipe
    scene_description: str
    max_episode_steps: int = 200     # 20 seconds at 10 Hz
    incident_script: Optional[WaypointScript] = None

    def get_ego_state(self) -> AgentState:
        return AgentState(
            id="review_agent",
            agent_type="review",
            x=self.ego_x,
            y=self.ego_y,
            vx=self.ego_speed_ms * math.cos(self.ego_yaw_rad),
            vy=self.ego_speed_ms * math.sin(self.ego_yaw_rad),
            yaw=self.ego_yaw_rad,
            yaw_rate=0.0,
            speed=self.ego_speed_ms,
            length=4.8, width=2.1, height=1.7,
        )

    def get_actor_states(self) -> List[AgentState]:
        states = []
        for a in self.actors:
            states.append(AgentState(
                id=a.actor_id,
                agent_type="incident" if a.is_incident_agent else "traffic",
                x=a.initial_x,
                y=a.initial_y,
                vx=a.initial_speed_ms * math.cos(a.initial_yaw_rad),
                vy=a.initial_speed_ms * math.sin(a.initial_yaw_rad),
                yaw=a.initial_yaw_rad,
                yaw_rate=0.0,
                speed=a.initial_speed_ms,
                length=a.size[0], width=a.size[1], height=a.size[2],
            ))
        return states


# ── Parser ───────────────────────────────────────────────────────────────────

class IncidentParser:
    """
    Parses incident tickets to InitialState.

    Usage:
        parser = IncidentParser(api_key=os.getenv("ANTHROPIC_API_KEY"))
        state = parser.parse(ticket)
    """

    # Default scenarios when parsing fails gracefully
    SCENARIO_TEMPLATES: Dict[str, dict] = {
        "jaywalker": {
            "ego_x": 0.0, "ego_y": 0.0, "ego_speed_ms": 11.2,  # 25 mph
            "actors": [
                {"id": "ped_1", "type": "pedestrian", "x": 18.0, "y": -3.0,
                 "speed": 1.2, "yaw": math.pi / 2, "incident": True,
                 "size": (0.8, 0.8, 1.8)},
                {"id": "car_1", "type": "vehicle", "x": -12.0, "y": 0.0,
                 "speed": 10.0, "yaw": 0.0, "incident": False,
                 "size": (4.5, 2.0, 1.6)},
            ],
            "incident_type": "jaywalker",
        },
        "rear_end": {
            "ego_x": 0.0, "ego_y": 0.0, "ego_speed_ms": 13.4,
            "actors": [
                {"id": "car_lead", "type": "vehicle", "x": 15.0, "y": 0.0,
                 "speed": 4.5, "yaw": 0.0, "incident": True,
                 "size": (4.8, 2.1, 1.7)},
                {"id": "car_follow", "type": "vehicle", "x": -12.0, "y": 0.0,
                 "speed": 14.0, "yaw": 0.0, "incident": False,
                 "size": (4.5, 2.0, 1.6)},
            ],
            "incident_type": "rear_end",
        },
        "intersection": {
            "ego_x": 0.0, "ego_y": 0.0, "ego_speed_ms": 8.0,
            "actors": [
                {"id": "car_cross", "type": "vehicle", "x": 25.0, "y": -20.0,
                 "speed": 12.0, "yaw": math.pi / 2, "incident": True,
                 "size": (4.6, 2.0, 1.7)},
                {"id": "ped_corner", "type": "pedestrian", "x": 22.0, "y": 8.0,
                 "speed": 0.0, "yaw": 0.0, "incident": False,
                 "size": (0.8, 0.8, 1.8)},
            ],
            "incident_type": "intersection",
        },
        "lane_change": {
            "ego_x": 0.0, "ego_y": 0.0, "ego_speed_ms": 15.6,
            "actors": [
                {"id": "car_merge", "type": "vehicle", "x": 8.0, "y": 4.0,
                 "speed": 16.0, "yaw": -0.3, "incident": True,
                 "size": (4.5, 2.0, 1.6)},
                {"id": "car_blind", "type": "vehicle", "x": -5.0, "y": 4.0,
                 "speed": 15.0, "yaw": 0.0, "incident": False,
                 "size": (5.0, 2.2, 2.0)},
            ],
            "incident_type": "lane_change",
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        if self._api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                print("[IncidentParser] anthropic package not found — using rule-based parser")

    def parse(self, ticket: IncidentTicket) -> InitialState:
        """Parse ticket → InitialState. Uses LLM if available, else rules."""
        if self._client is not None:
            return self._parse_with_llm(ticket)
        return self._parse_rule_based(ticket)

    # ── LLM parsing ──────────────────────────────────────────────────────────

    def _parse_with_llm(self, ticket: IncidentTicket) -> InitialState:
        prompt = f"""You are an autonomous vehicle incident analyst.
Parse this incident ticket and extract a structured JSON scene description.

TICKET:
{ticket.raw_text}

Return ONLY a JSON object with this exact schema:
{{
  "incident_type": "jaywalker|rear_end|intersection|lane_change|sideswipe",
  "weather": "clear|rain|heavy_rain|fog|snow|overcast",
  "speed_limit_mph": <number>,
  "ego_speed_mph": <number>,
  "ego_yaw_deg": <number>,
  "traffic_density": <0.0-1.0>,
  "actors": [
    {{
      "id": "<string>",
      "type": "vehicle|pedestrian|cyclist",
      "rel_x": <metres ahead of ego>,
      "rel_y": <metres left(+)/right(-) of ego>,
      "speed_mph": <number>,
      "yaw_deg": <degrees>,
      "is_incident_agent": <bool>
    }}
  ],
  "scene_description": "<one sentence summary>"
}}"""

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            # Extract JSON from response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return self._dict_to_initial_state(ticket, data)
        except Exception as e:
            print(f"[IncidentParser] LLM parse failed ({e}), falling back to rules")

        return self._parse_rule_based(ticket)

    # ── Rule-based parsing ───────────────────────────────────────────────────

    def _parse_rule_based(self, ticket: IncidentTicket) -> InitialState:
        text = ticket.raw_text.lower()

        # Incident type detection
        if any(k in text for k in ["jaywalk", "pedestrian crossing", "person crossing"]):
            incident_type = "jaywalker"
        elif any(k in text for k in ["rear-end", "rear end", "following too closely", "brake check"]):
            incident_type = "rear_end"
        elif any(k in text for k in ["intersection", "ran red light", "t-bone", "cross traffic"]):
            incident_type = "intersection"
        elif any(k in text for k in ["lane change", "merge", "cut off", "blind spot"]):
            incident_type = "lane_change"
        else:
            incident_type = "jaywalker"  # default

        # Weather extraction
        weather = "clear"
        for w in ["fog", "snow", "heavy rain", "rain", "overcast"]:
            if w in text:
                weather = w.replace(" ", "_")
                break

        # Speed limit
        speed_match = re.search(r'(\d+)\s*mph', text)
        speed_limit_mph = int(speed_match.group(1)) if speed_match else 30

        # Build initial state from template
        template = self.SCENARIO_TEMPLATES.get(incident_type, self.SCENARIO_TEMPLATES["jaywalker"])
        constraints = WaymoConstraints.from_incident_ticket({
            "weather": weather,
            "speed_limit_mph": speed_limit_mph,
            "traffic_density": 0.4,
        })

        actors = []
        for a in template["actors"]:
            actors.append(InvolvedActor(
                actor_id=a["id"],
                actor_type=a["type"],
                initial_x=a["x"],
                initial_y=a["y"],
                initial_speed_ms=a["speed"],
                initial_yaw_rad=a["yaw"],
                is_incident_agent=a["incident"],
                size=a.get("size", (4.5, 2.0, 1.6)),
            ))

        ego_speed = template["ego_speed_ms"]
        # Adjust ego speed from ticket
        ego_speed_match = re.search(r'traveling at (\d+)', text)
        if ego_speed_match:
            ego_speed = int(ego_speed_match.group(1)) * 0.44704

        incident_script = self._build_incident_script(incident_type, actors)

        return InitialState(
            ticket_id=ticket.ticket_id,
            ego_x=template["ego_x"],
            ego_y=template["ego_y"],
            ego_speed_ms=min(ego_speed, constraints.max_safe_speed_ms),
            ego_yaw_rad=0.0,
            actors=actors,
            constraints=constraints,
            incident_type=incident_type,
            scene_description=f"{incident_type.replace('_', ' ').title()} scenario in {weather} conditions",
            max_episode_steps=200,
            incident_script=incident_script,
        )

    def _dict_to_initial_state(self, ticket: IncidentTicket, data: dict) -> InitialState:
        """Convert LLM JSON output to InitialState."""
        constraints = WaymoConstraints.from_incident_ticket({
            "weather": data.get("weather", "clear"),
            "speed_limit_mph": data.get("speed_limit_mph", 30),
            "traffic_density": data.get("traffic_density", 0.4),
        })

        actors = []
        for a in data.get("actors", []):
            size = (4.5, 2.0, 1.6) if a.get("type") == "vehicle" else (0.8, 0.8, 1.8)
            actors.append(InvolvedActor(
                actor_id=a.get("id", f"agent_{len(actors)}"),
                actor_type=a.get("type", "vehicle"),
                initial_x=float(a.get("rel_x", 15.0)),
                initial_y=float(a.get("rel_y", 0.0)),
                initial_speed_ms=float(a.get("speed_mph", 20)) * 0.44704,
                initial_yaw_rad=math.radians(float(a.get("yaw_deg", 0))),
                is_incident_agent=bool(a.get("is_incident_agent", False)),
                size=size,
            ))

        ego_speed = float(data.get("ego_speed_mph", 25)) * 0.44704
        incident_type = data.get("incident_type", "jaywalker")
        incident_script = self._build_incident_script(incident_type, actors)

        return InitialState(
            ticket_id=ticket.ticket_id,
            ego_x=0.0,
            ego_y=0.0,
            ego_speed_ms=ego_speed,
            ego_yaw_rad=math.radians(float(data.get("ego_yaw_deg", 0))),
            actors=actors,
            constraints=constraints,
            incident_type=incident_type,
            scene_description=data.get("scene_description", ""),
            max_episode_steps=200,
            incident_script=incident_script,
        )

    def _build_incident_script(
        self,
        incident_type: str,
        actors: List[InvolvedActor],
    ) -> Optional[WaypointScript]:
        """
        Build a scripted trajectory for the incident agent.

        The script begins the moment the proximity trigger fires (t=0 of script
        = the instant ego enters trigger_distance). It defines what the incident
        actor DOES once triggered — e.g. the pedestrian starts crossing the road.
        """
        incident_actors = [a for a in actors if a.is_incident_agent]
        if not incident_actors:
            return None

        a = incident_actors[0]
        dt = 0.1

        timestamps, xs, ys, speeds, yaws = [], [], [], [], []

        if incident_type == "jaywalker":
            # Pedestrian walks perpendicular across the road (south to north, +y direction)
            # Crosses ~8m of road at 1.3 m/s — takes ~6 seconds
            cross_speed = 1.3
            cross_yaw = math.pi / 2  # heading north (+y)
            cross_distance = 9.0
            n_steps = int(cross_distance / cross_speed / dt) + 20  # extra frames after crossing

            x, y = a.initial_x, a.initial_y
            for i in range(n_steps):
                t = i * dt
                timestamps.append(t)
                xs.append(x)
                ys.append(y)
                dist_covered = cross_speed * t
                if dist_covered < cross_distance:
                    speeds.append(cross_speed)
                    yaws.append(cross_yaw)
                    y = a.initial_y + dist_covered
                    x = a.initial_x
                else:
                    # Crossed — stop on far side
                    speeds.append(0.0)
                    yaws.append(cross_yaw)
                    y = a.initial_y + cross_distance
                    x = a.initial_x

        elif incident_type == "intersection":
            # Cross-traffic car drives straight through intersection
            cross_speed = a.initial_speed_ms or 12.0
            cross_yaw = a.initial_yaw_rad
            n_steps = 50  # 5 seconds
            x, y = a.initial_x, a.initial_y
            for i in range(n_steps):
                timestamps.append(i * dt)
                xs.append(x)
                ys.append(y)
                speeds.append(cross_speed)
                yaws.append(cross_yaw)
                x += cross_speed * math.cos(cross_yaw) * dt
                y += cross_speed * math.sin(cross_yaw) * dt

        elif incident_type == "lane_change":
            # Merging car drifts laterally into ego lane
            fwd_speed = a.initial_speed_ms or 15.0
            n_steps = 40
            x, y = a.initial_x, a.initial_y
            for i in range(n_steps):
                timestamps.append(i * dt)
                xs.append(x)
                ys.append(y)
                # Lateral drift towards y=0 (ego lane)
                lateral_rate = -0.15 if y > 0 else 0.15
                yaw = math.atan2(lateral_rate, fwd_speed)
                speeds.append(fwd_speed)
                yaws.append(yaw)
                x += fwd_speed * dt
                y += lateral_rate * dt

        else:
            # Default: slow deceleration (rear-end risk)
            spd = a.initial_speed_ms or 4.5
            yaw = a.initial_yaw_rad
            n_steps = 60
            x, y = a.initial_x, a.initial_y
            for i in range(n_steps):
                timestamps.append(i * dt)
                xs.append(x)
                ys.append(y)
                speeds.append(max(0.0, spd - 0.05 * i))  # gradually slowing
                yaws.append(yaw)
                x += spd * math.cos(yaw) * dt
                y += spd * math.sin(yaw) * dt

        return WaypointScript(timestamps=timestamps, xs=xs, ys=ys, speeds=speeds, yaws=yaws)
