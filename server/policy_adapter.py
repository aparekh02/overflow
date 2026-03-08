"""
Adapter between OverflowObservation (2D road grid) and the OpenENV policy
observation format (ego state + ticket matrix).

Nearby cars are converted to collision_risk tickets so TicketAttentionPolicy
can reason about them using the same mechanism it was designed for.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from ..policies.policy_spec import build_obs, build_ticket_vector, OBS_DIM
except ImportError:
    from policies.policy_spec import build_obs, build_ticket_vector, OBS_DIM


def overflow_obs_to_policy_obs(obs) -> np.ndarray:
    """OverflowObservation → 603-dim numpy vector for our policies."""
    cars = obs.cars
    if not cars:
        return np.zeros(OBS_DIM, dtype=np.float32)

    ego = next((c for c in cars if c.carId == 0), cars[0])
    ego_speed_ms = ego.speed / 4.5            # OverflowEnv speed units → m/s
    ego_x        = ego.position.x
    ego_y        = (ego.lane - 2) * 3.7      # lane → lateral metres

    ticket_vectors = []
    for car in cars:
        if car.carId == 0:
            continue
        rel_x    = car.position.x - ego.position.x
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


def policy_action_to_decision(action_vec: np.ndarray) -> tuple[str, str]:
    """Continuous [steer, throttle, brake] → (text decision, reasoning)."""
    steer, throttle, brake = float(action_vec[0]), float(action_vec[1]), float(action_vec[2])
    if abs(steer) > 0.35:
        decision  = "lane_change_left" if steer < 0 else "lane_change_right"
        reasoning = f"steer={steer:.2f}: lateral avoidance"
    elif brake > 0.25:
        decision  = "brake"
        reasoning = f"brake={brake:.2f}: closing gap"
    elif throttle > 0.20:
        decision  = "accelerate"
        reasoning = f"throttle={throttle:.2f}: clear ahead"
    else:
        decision  = "maintain"
        reasoning = f"s={steer:.2f} t={throttle:.2f} b={brake:.2f}: holding course"
    return decision, reasoning
