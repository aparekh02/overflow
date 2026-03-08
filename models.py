"""
Data models for the Overflow Environment.

An autonomous vehicle fleet oversight environment where an LLM agent
controls one car on a 2D road grid while other cars follow scripted rules.

Structured observation fields (cars, proximities, lane_occupancies) are
compatible with the Overflow frontend's CarState / AnomalyObservation types.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    class Action(BaseModel): pass
    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
    class State(BaseModel):
        episode_id: str = ""
        step_count: int = 0

# ── Structured sub-models (frontend-compatible) ─────────────────────────


class Position(BaseModel):
    """2D position on the road. x = longitudinal, y = lateral."""

    x: float = 0.0
    y: float = 0.0


class CarStateData(BaseModel):
    """
    Structured per-car snapshot — matches the frontend CarState interface.

    Frontend type:
        interface CarState {
            carId: number; lane: number;
            position: { x: number; y: number };
            speed: number; acceleration: number;
        }
    """

    carId: int
    lane: int
    position: Position
    speed: float
    acceleration: float = 0.0


class ProximityData(BaseModel):
    """Pairwise distance between two cars."""

    carA: int
    carB: int
    distance: float


class LaneOccupancyData(BaseModel):
    """Which cars are in a given lane."""

    lane: int
    carIds: List[int]


# ── OpenEnv core models ─────────────────────────────────────────────────


class OverflowAction(Action):
    """
    Action for the Overflow environment.

    The LLM agent outputs a driving decision and optional reasoning.
    """

    decision: str = Field(
        default="maintain",
        description="Driving decision: accelerate, brake, lane_change_left, lane_change_right, maintain",
    )
    reasoning: str = Field(
        default="",
        description="The LLM's chain-of-thought reasoning for this decision",
    )


class OverflowObservation(Observation):
    """
    Observation from the Overflow environment.

    Contains both:
    - Text fields (scene_description, incident_report) for the LLM to read.
    - Structured fields (cars, proximities, lane_occupancies) for the frontend
      to render, matching the Overflow frontend AnomalyObservation shape.
    """

    # ── Text (for the LLM) ──
    scene_description: str = Field(
        default="", description="Text description of the traffic scene"
    )
    incident_report: str = Field(
        default="", description="Observer's incident report, empty if no incident"
    )

    # ── Structured (for the frontend / viz) ──
    cars: List[CarStateData] = Field(
        default_factory=list, description="Structured state of every car"
    )
    proximities: List[ProximityData] = Field(
        default_factory=list, description="Pairwise proximity measurements"
    )
    lane_occupancies: List[LaneOccupancyData] = Field(
        default_factory=list, description="Per-lane vehicle occupancy"
    )


class OverflowState(State):
    """
    Internal state for the Overflow environment.
    """

    crash_count: int = Field(default=0, description="Number of crashes this episode")
    near_miss_count: int = Field(
        default=0, description="Number of near misses this episode"
    )
    cars_reached_goal: int = Field(
        default=0, description="Number of cars that reached their goal"
    )
    total_cars: int = Field(
        default=5, description="Total number of cars in the simulation"
    )
