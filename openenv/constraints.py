"""
WaymoConstraints — extracts real environmental constraints from Waymo scene data.

Novel: Instead of synthetic physics parameters, we derive friction, visibility,
and traffic density directly from the Waymo LiDAR observations, making the RL
environment grounded in the actual measured scene geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvironmentConstraints:
    """Physics + traffic constraints derived from Waymo scene data."""

    # Road physics
    friction: float = 0.85          # 0.3 (ice) → 1.0 (dry)
    visibility_m: float = 120.0     # metres — derived from point cloud density
    wind_speed_ms: float = 0.0

    # Traffic state
    traffic_density: float = 0.3    # 0 (empty) → 1.0 (gridlock)
    speed_limit_ms: float = 13.4    # 30 mph default

    # Weather category (for rendering effects)
    weather: str = "clear"          # clear | rain | fog | snow | overcast

    # Derived from Waymo point cloud stats
    point_density: float = 1.0      # relative to baseline (1.0 = nominal)
    avg_intensity: float = 0.5      # mean LiDAR return intensity
    num_agents: int = 0             # detected objects in scene

    # Dynamic multipliers applied to RL physics
    @property
    def braking_scale(self) -> float:
        """Scales braking distance — wet/icy roads reduce effectiveness."""
        return self.friction

    @property
    def visibility_scale(self) -> float:
        """0→1, scales how far agents can "see" in obs space."""
        return min(1.0, self.visibility_m / 150.0)

    @property
    def max_safe_speed_ms(self) -> float:
        """Friction-adjusted safe speed."""
        base = self.speed_limit_ms
        return base * math.sqrt(self.friction)

    def to_obs_vector(self) -> list[float]:
        """5-dim constraint vector included in every RL observation."""
        return [
            self.friction,
            self.visibility_scale,
            1.0 if self.weather in ("rain", "snow") else 0.0,
            1.0 if self.weather in ("fog",) else 0.0,
            self.traffic_density,
        ]


class WaymoConstraints:
    """
    Derives EnvironmentConstraints from raw Waymo frame data.

    Novel approach: LiDAR point cloud statistics (density, intensity distribution,
    return dropouts) act as implicit sensors for road and weather conditions —
    exactly how a deployed AV would infer conditions without explicit labels.
    """

    # Baseline point counts at nominal conditions (calibrated to mock data)
    BASELINE_POINT_COUNT = 42_000

    # Intensity thresholds for road surface classification
    WET_ROAD_INTENSITY_THRESHOLD = 0.25   # wet roads have low retro-reflectivity
    DRY_ROAD_INTENSITY_THRESHOLD = 0.40

    @classmethod
    def from_waymo_frame(
        cls,
        point_count: int,
        avg_intensity: float,
        num_agents: int,
        weather_hint: Optional[str] = None,
    ) -> EnvironmentConstraints:
        """
        Infer environmental constraints from a single Waymo frame.

        Args:
            point_count:   LiDAR returns this frame
            avg_intensity: mean intensity across all returns
            num_agents:    detected bounding boxes
            weather_hint:  optional label from Waymo metadata
        """
        # Visibility estimate from relative point density
        density_ratio = point_count / max(cls.BASELINE_POINT_COUNT, 1)
        visibility_m = cls._density_to_visibility(density_ratio)

        # Road friction from intensity distribution
        friction = cls._intensity_to_friction(avg_intensity)

        # Weather inference
        weather = weather_hint or cls._infer_weather(density_ratio, avg_intensity)

        # Wind: approximated from high-speed agents' heading spread (simplified here)
        wind_speed = 0.0

        # Traffic density: normalize agent count (assume >20 agents = dense)
        traffic_density = min(1.0, num_agents / 20.0)

        return EnvironmentConstraints(
            friction=friction,
            visibility_m=visibility_m,
            wind_speed_ms=wind_speed,
            traffic_density=traffic_density,
            weather=weather,
            point_density=density_ratio,
            avg_intensity=avg_intensity,
            num_agents=num_agents,
        )

    @classmethod
    def from_incident_ticket(cls, ticket_constraints: dict) -> EnvironmentConstraints:
        """Create constraints from a parsed incident ticket dict."""
        weather = ticket_constraints.get("weather", "clear")
        speed_limit_mph = ticket_constraints.get("speed_limit_mph", 30)

        friction_map = {
            "clear": 0.85,
            "overcast": 0.80,
            "rain": 0.55,
            "heavy_rain": 0.40,
            "fog": 0.75,
            "snow": 0.35,
            "ice": 0.25,
        }
        visibility_map = {
            "clear": 150.0,
            "overcast": 120.0,
            "rain": 80.0,
            "heavy_rain": 40.0,
            "fog": 20.0,
            "snow": 50.0,
            "ice": 120.0,
        }

        return EnvironmentConstraints(
            friction=friction_map.get(weather, 0.85),
            visibility_m=visibility_map.get(weather, 120.0),
            traffic_density=ticket_constraints.get("traffic_density", 0.4),
            speed_limit_ms=speed_limit_mph * 0.44704,
            weather=weather,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _density_to_visibility(density_ratio: float) -> float:
        """
        Maps LiDAR point density ratio to visibility in metres.
        Rain, fog, snow scatter the laser → fewer returns → lower density.
        """
        if density_ratio >= 0.95:
            return 150.0
        elif density_ratio >= 0.75:
            return 80.0 + 70.0 * (density_ratio - 0.75) / 0.20
        elif density_ratio >= 0.50:
            return 30.0 + 50.0 * (density_ratio - 0.50) / 0.25
        else:
            return max(10.0, density_ratio * 60.0)

    @staticmethod
    def _intensity_to_friction(avg_intensity: float) -> float:
        """
        Maps mean return intensity to road friction coefficient.
        Wet asphalt has lower specular reflection → lower intensity → lower friction.
        """
        # Linearly interpolate friction from intensity
        low_i, low_f = 0.15, 0.35  # very wet / icy
        high_i, high_f = 0.55, 0.90  # dry road
        t = max(0.0, min(1.0, (avg_intensity - low_i) / (high_i - low_i)))
        return low_f + t * (high_f - low_f)

    @staticmethod
    def _infer_weather(density_ratio: float, avg_intensity: float) -> str:
        if density_ratio < 0.5 and avg_intensity < 0.25:
            return "fog"
        elif density_ratio < 0.7 and avg_intensity < 0.30:
            return "rain"
        elif density_ratio < 0.85 and avg_intensity < 0.35:
            return "overcast"
        return "clear"
