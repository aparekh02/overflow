"""Overflow Environment — Autonomous vehicle fleet oversight for OpenEnv."""

try:
    from .client import OverflowEnv
except ImportError:
    OverflowEnv = None  # openenv-core not installed; training-only mode

from .models import (
    CarStateData,
    LaneOccupancyData,
    OverflowAction,
    OverflowObservation,
    OverflowState,
    Position,
    ProximityData,
)

__all__ = [
    "OverflowAction",
    "OverflowObservation",
    "OverflowState",
    "OverflowEnv",
    "CarStateData",
    "Position",
    "ProximityData",
    "LaneOccupancyData",
]
