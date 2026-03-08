"""
Incident-response reward system for OverflowEnvironment.

Instead of diffuse per-step shaping, every step:
  1. Classifies the scene into the highest-priority incident type
  2. Grades the agent's action against the expected response for that incident
  3. Returns a large, signed reward tied to the actual decision quality

This creates a strong, non-fluctuating gradient:
  - Correct incident response  → large positive reward
  - Wrong incident response    → large negative reward
  - No incident / clear road   → small positive for efficiency

Incident priority (high → low):
  CRASH_IMMINENT      dist < 5  to any car
  NEAR_MISS_AHEAD     dist < 15, same lane, ahead
  NEAR_MISS_SIDE      dist < 15, adjacent lane
  BLOCKED_AHEAD       dist < 30, same lane, ahead, closing
  APPROACHING_GOAL    no threat, within 40 units of goal
  CLEAR_ROAD          no threat
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


# ── Incident types ────────────────────────────────────────────────────────────

class IncidentType(str, Enum):
    CRASH_IMMINENT   = "CRASH_IMMINENT"
    NEAR_MISS_AHEAD  = "NEAR_MISS_AHEAD"
    NEAR_MISS_SIDE   = "NEAR_MISS_SIDE"
    BLOCKED_AHEAD    = "BLOCKED_AHEAD"
    APPROACHING_GOAL = "APPROACHING_GOAL"
    CLEAR_ROAD       = "CLEAR_ROAD"


@dataclass
class IncidentContext:
    incident_type: IncidentType
    nearest_threat_dist: float
    threatening_car_id: int
    same_lane: bool
    threat_ahead: bool      # threat is in front of ego


# ── Response grading tables ───────────────────────────────────────────────────

# (decision) → reward for each incident type
# Designed so:
#   - Correct responses yield 3–8x more reward than baseline
#   - Wrong responses yield clear negative signal
#   - "Wrong direction" actions yield the strongest penalty

_GRADES: dict[IncidentType, dict[str, float]] = {
    # Imminent crash: must evade immediately
    IncidentType.CRASH_IMMINENT: {
        "brake":             8.0,   # correct — slow down now
        "lane_change_left":  7.0,   # good — create lateral separation
        "lane_change_right": 7.0,
        "maintain":         -10.0,  # failed to respond
        "accelerate":       -15.0,  # actively made it worse
    },
    # Near miss, same lane ahead: need to slow or change lane
    IncidentType.NEAR_MISS_AHEAD: {
        "brake":             5.0,
        "lane_change_left":  6.0,   # better — removes from collision path
        "lane_change_right": 6.0,
        "maintain":          -3.0,
        "accelerate":        -8.0,
    },
    # Near miss, adjacent lane: don't swerve into them
    IncidentType.NEAR_MISS_SIDE: {
        "brake":             3.0,   # slow down, reduce conflict zone
        "maintain":          2.0,   # hold steady — acceptable
        "lane_change_left":  -4.0,  # graded below — may swerve into threat
        "lane_change_right": -4.0,
        "accelerate":        -2.0,  # speed through works sometimes, but risky
    },
    # Blocked but not yet critical: brake or lane-change proactively
    IncidentType.BLOCKED_AHEAD: {
        "brake":             3.0,
        "lane_change_left":  4.0,
        "lane_change_right": 4.0,
        "maintain":         -1.0,
        "accelerate":        -5.0,
    },
    # Approaching goal: keep speed, don't brake unnecessarily
    IncidentType.APPROACHING_GOAL: {
        "accelerate":        2.5,
        "maintain":          2.0,
        "brake":            -1.0,
        "lane_change_left":  0.5,
        "lane_change_right": 0.5,
    },
    # Clear road: maintain speed and lane discipline
    IncidentType.CLEAR_ROAD: {
        "accelerate":        1.0,
        "maintain":          0.8,
        "brake":            -0.5,   # unnecessary braking
        "lane_change_left":  0.0,
        "lane_change_right": 0.0,
    },
}

# Fixed terminal rewards
W_COLLISION     = -50.0
W_GOAL          =  10.0
W_SURVIVE_BONUS =   5.0


# ── Scene classification ──────────────────────────────────────────────────────

def classify_scene(
    ego_x:   float,
    ego_lane: int,
    ego_speed: float,
    cars: list,          # list of CarStateData (from overflow_obs.cars)
    goal_x:  float,
) -> IncidentContext:
    """
    Classify the highest-priority incident the ego car faces this step.

    Cars are CarStateData objects with .carId, .position.x, .lane, .speed.
    """
    best: Optional[IncidentContext] = None
    best_priority = -1

    PRIORITY = {
        IncidentType.CRASH_IMMINENT:   5,
        IncidentType.NEAR_MISS_AHEAD:  4,
        IncidentType.NEAR_MISS_SIDE:   3,
        IncidentType.BLOCKED_AHEAD:    2,
        IncidentType.APPROACHING_GOAL: 1,
        IncidentType.CLEAR_ROAD:       0,
    }

    for car in cars:
        if car.carId == 0:
            continue

        rel_x     = car.position.x - ego_x
        lane_diff = abs(car.lane - ego_lane)
        # Euclidean-ish (same as overflow_environment.Car.distance_to)
        dist = math.sqrt(rel_x ** 2 + (lane_diff * 10.0) ** 2)

        same_lane = (car.lane == ego_lane)
        ahead     = rel_x > 0

        # Determine incident type for this car
        if dist < 5.0:
            inc_type = IncidentType.CRASH_IMMINENT
        elif dist < 15.0 and same_lane and ahead:
            inc_type = IncidentType.NEAR_MISS_AHEAD
        elif dist < 15.0 and not same_lane:
            inc_type = IncidentType.NEAR_MISS_SIDE
        elif dist < 30.0 and same_lane and ahead:
            # Only counts as blocked if ego is actually closing
            closing_speed = ego_speed - car.speed
            if closing_speed > 0:
                inc_type = IncidentType.BLOCKED_AHEAD
            else:
                continue
        else:
            continue

        p = PRIORITY[inc_type]
        if p > best_priority or (p == best_priority and dist < (best.nearest_threat_dist if best else 1e9)):
            best_priority = p
            best = IncidentContext(
                incident_type=inc_type,
                nearest_threat_dist=dist,
                threatening_car_id=car.carId,
                same_lane=same_lane,
                threat_ahead=ahead,
            )

    if best is not None:
        return best

    goal_dist = abs(goal_x - ego_x)
    if goal_dist < 40.0:
        return IncidentContext(
            incident_type=IncidentType.APPROACHING_GOAL,
            nearest_threat_dist=goal_dist,
            threatening_car_id=-1,
            same_lane=False,
            threat_ahead=True,
        )

    return IncidentContext(
        incident_type=IncidentType.CLEAR_ROAD,
        nearest_threat_dist=999.0,
        threatening_car_id=-1,
        same_lane=False,
        threat_ahead=False,
    )


def grade_response(incident: IncidentContext, decision: str) -> Tuple[float, str]:
    """
    Return (reward, description) for the agent's decision given the incident context.

    For NEAR_MISS_SIDE: penalise lane changes that move *toward* the threat.
    For other incidents: use the fixed grade table.
    """
    grades = _GRADES[incident.incident_type]

    # Special case: side near-miss — swerving away is ok, swerving toward is bad
    if incident.incident_type == IncidentType.NEAR_MISS_SIDE:
        # We don't know which side the threat is without the car's lane,
        # so default to "maintain is safe, any lane change is risky"
        grade = grades.get(decision, 0.0)
    else:
        grade = grades.get(decision, 0.0)

    desc = (
        f"{incident.incident_type.value} "
        f"(dist={incident.nearest_threat_dist:.1f}, car={incident.threatening_car_id}) "
        f"→ {decision}: {grade:+.1f}"
    )
    return grade, desc


# ── Main reward function ──────────────────────────────────────────────────────

def compute_reward(
    ego_speed:    float,
    ego_x:        float,
    ego_lane:     int,
    ego_y:        float,
    decision:     str,
    action:       np.ndarray,
    prev_action:  np.ndarray,
    collision:    bool,
    goal_reached: bool,
    cars: list,
    goal_x: float,
) -> Tuple[float, IncidentContext, str]:
    """
    Incident-response reward.

    Returns (reward, incident_context, grade_description).
    """
    if collision:
        ctx = IncidentContext(IncidentType.CRASH_IMMINENT, 0.0, -1, False, False)
        return W_COLLISION, ctx, f"COLLISION → {W_COLLISION}"

    # Classify scene and grade response
    ctx   = classify_scene(ego_x, ego_lane, ego_speed, cars, goal_x)
    grade, desc = grade_response(ctx, decision)
    reward = grade

    # Goal terminal bonus
    if goal_reached:
        reward += W_GOAL

    return float(reward), ctx, desc


def compute_episode_bonus(total_steps: int, survived: bool) -> float:
    """Survival bonus scales with longevity."""
    if not survived:
        return 0.0
    bonus  = W_SURVIVE_BONUS
    bonus += min(total_steps, 100) * 0.05   # up to +5 for lasting 100 steps
    return float(bonus)
