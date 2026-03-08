"""
Overflow Environment Client.

Provides the client for connecting to an Overflow Environment server
via WebSocket for persistent sessions.
"""

from typing import Any, Dict, List

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    CarStateData,
    LaneOccupancyData,
    OverflowAction,
    OverflowObservation,
    OverflowState,
    Position,
    ProximityData,
)


class OverflowEnv(EnvClient[OverflowAction, OverflowObservation, OverflowState]):
    """
    WebSocket client for the Overflow Environment.

    Example:
        >>> with OverflowEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.scene_description)
        ...     print(result.observation.cars)  # structured car data
        ...     action = OverflowAction(decision="maintain", reasoning="Safe for now")
        ...     result = env.step(action)
    """

    def _step_payload(self, action: OverflowAction) -> Dict[str, Any]:
        """Convert OverflowAction to JSON payload for step request."""
        return {
            "decision": action.decision,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[OverflowObservation]:
        """Parse server response into StepResult[OverflowObservation]."""
        obs_data = payload.get("observation", {})

        # Parse structured car data
        cars = [
            CarStateData(
                carId=c["carId"],
                lane=c["lane"],
                position=Position(**c["position"]),
                speed=c["speed"],
                acceleration=c.get("acceleration", 0.0),
            )
            for c in obs_data.get("cars", [])
        ]

        proximities = [
            ProximityData(**p) for p in obs_data.get("proximities", [])
        ]

        lane_occupancies = [
            LaneOccupancyData(**lo) for lo in obs_data.get("lane_occupancies", [])
        ]

        observation = OverflowObservation(
            scene_description=obs_data.get("scene_description", ""),
            incident_report=obs_data.get("incident_report", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            cars=cars,
            proximities=proximities,
            lane_occupancies=lane_occupancies,
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> OverflowState:
        """Parse server response into OverflowState."""
        return OverflowState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            crash_count=payload.get("crash_count", 0),
            near_miss_count=payload.get("near_miss_count", 0),
            cars_reached_goal=payload.get("cars_reached_goal", 0),
            total_cars=payload.get("total_cars", 5),
        )
