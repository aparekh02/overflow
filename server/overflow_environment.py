"""
Overflow Environment Implementation.

A 2D road grid with N cars. One car (Car 0) is the LLM agent, others follow
scripted rules. An observer checks for collisions each step. The environment
returns text observations describing the traffic scene and rewards based on safety.

Observations carry both text (for the LLM) and structured data (for the frontend).
"""

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import (
    CarStateData,
    LaneOccupancyData,
    OverflowAction,
    OverflowObservation,
    OverflowState,
    Position,
    ProximityData,
)

# --- Constants ---
NUM_LANES = 3
ROAD_LENGTH = 200
NUM_CARS = 5
MAX_STEPS = 100
CRASH_DISTANCE = 5.0
NEAR_MISS_DISTANCE = 15.0
LANE_WIDTH = 3.7  # metres — matches frontend's makeCar convention

# Reward values
REWARD_CRASH = -5.0
REWARD_NEAR_MISS = -1.0
REWARD_SAFE_STEP = 0.5
REWARD_REACHED_GOAL = 3.0
REWARD_REASONING_MAX = 0.3

# Speed bounds
MIN_SPEED = 20
MAX_SPEED = 90
SPEED_DELTA = 5


@dataclass
class Car:
    """Represents a car on the road grid."""

    car_id: int
    lane: int  # 1-indexed: 1, 2, or 3
    position: float
    speed: float
    goal_position: float
    is_agent: bool = False
    reached_goal: bool = False
    prev_speed: float = 0.0  # speed last step, for acceleration calc

    def distance_to(self, other: "Car") -> float:
        """Euclidean-ish distance considering lane and position."""
        lane_diff = abs(self.lane - other.lane) * 10.0  # lanes are ~10 units apart
        pos_diff = abs(self.position - other.position)
        return math.sqrt(lane_diff**2 + pos_diff**2)

    @property
    def acceleration(self) -> float:
        """Speed delta since last step."""
        return self.speed - self.prev_speed

    def to_state_data(self) -> CarStateData:
        """Convert to frontend-compatible CarStateData."""
        return CarStateData(
            carId=self.car_id,
            lane=self.lane,
            position=Position(x=self.position, y=self.lane * LANE_WIDTH),
            speed=self.speed,
            acceleration=self.acceleration,
        )


def _parse_decision(action: OverflowAction) -> str:
    """Extract a valid decision from the action, being forgiving about format."""
    valid = {"accelerate", "brake", "lane_change_left", "lane_change_right", "maintain"}

    # Try the decision field directly
    decision = action.decision.strip().lower().replace(" ", "_")
    if decision in valid:
        return decision

    # Try to extract from free text (the LLM might wrap it in tags)
    text = f"{action.decision} {action.reasoning}".lower()

    # Check for <action>...</action> tags
    match = re.search(r"<action>\s*(\w+)\s*</action>", text)
    if match:
        candidate = match.group(1).strip().replace(" ", "_")
        if candidate in valid:
            return candidate

    # Check for keywords anywhere (ordered: most specific first to avoid ambiguity)
    for v in ["lane_change_left", "lane_change_right", "accelerate", "brake", "maintain"]:
        if v in text:
            return v

    return "maintain"


def _compute_reasoning_bonus(reasoning: str) -> float:
    """
    Compute a small reasoning quality bonus (0.0 to 0.3).

    Gives a minor reward for providing structured reasoning, kept low
    so driving performance remains the dominant training signal.
    """
    if not reasoning:
        return 0.0

    score = 0.0
    lower = reasoning.lower()

    # Small bonus for providing any reasoning at all
    if len(reasoning) > 20:
        score += 0.1

    # Bonus for structured reasoning (not just keyword stuffing)
    if "<think>" in lower or "because" in lower:
        score += 0.1
    if any(word in lower for word in ["therefore", "so i should", "best option", "i will"]):
        score += 0.1

    return min(score, REWARD_REASONING_MAX)


def _scripted_car_action(car: Car, all_cars: List[Car], rng: random.Random) -> str:
    """
    Simple scripted AI for non-agent cars.

    Rules:
    - If car ahead in same lane is close (< 20 units): brake
    - If speed is low and random chance: accelerate
    - Otherwise: maintain
    """
    # Find nearest car ahead in same lane
    nearest_ahead_dist = float("inf")
    for other in all_cars:
        if other.car_id == car.car_id:
            continue
        if other.lane == car.lane and other.position > car.position:
            dist = other.position - car.position
            if dist < nearest_ahead_dist:
                nearest_ahead_dist = dist

    if nearest_ahead_dist < 20:
        return "brake"

    if car.speed < 60 and rng.random() < 0.1:
        return "accelerate"

    # Occasionally change lanes to make traffic more dynamic
    if rng.random() < 0.05:
        if car.lane > 1 and rng.random() < 0.5:
            return "lane_change_left"
        elif car.lane < NUM_LANES:
            return "lane_change_right"

    return "maintain"


def _apply_action(car: Car, decision: str) -> None:
    """Apply a driving decision to a car, mutating it in place."""
    if decision == "accelerate":
        car.speed = min(car.speed + SPEED_DELTA, MAX_SPEED)
    elif decision == "brake":
        car.speed = max(car.speed - SPEED_DELTA, MIN_SPEED)
    elif decision == "lane_change_left":
        if car.lane > 1:
            car.lane -= 1
    elif decision == "lane_change_right":
        if car.lane < NUM_LANES:
            car.lane += 1
    # "maintain" — no change


def _generate_scene_description(agent_car: Car, cars: List[Car]) -> str:
    """Generate a text description of the current traffic scene."""
    lines = [
        f"You are Car 0 in lane {agent_car.lane}, position {agent_car.position:.0f}, speed {agent_car.speed:.0f}.",
        f"Goal: reach position {agent_car.goal_position:.0f}.",
        "Nearby cars:",
    ]

    for car in cars:
        if car.car_id == agent_car.car_id:
            continue

        detail = f"- Car {car.car_id}: lane {car.lane}, position {car.position:.0f}, speed {car.speed:.0f}"

        # Add context about relative position
        if car.lane == agent_car.lane:
            pos_diff = car.position - agent_car.position
            if pos_diff > 0:
                detail += f" [AHEAD IN YOUR LANE - {pos_diff:.0f} units away]"
            else:
                detail += f" [BEHIND IN YOUR LANE - {abs(pos_diff):.0f} units away]"

        if car.reached_goal:
            detail += " [REACHED GOAL]"

        lines.append(detail)

    return "\n".join(lines)


def _build_structured_data(
    cars: List[Car],
    proximity_pairs: List[ProximityData],
) -> tuple[List[CarStateData], List[LaneOccupancyData]]:
    """Build structured arrays for the observation."""
    cars_data = [c.to_state_data() for c in cars]

    # Lane occupancies
    lane_map: dict[int, list[int]] = {}
    for car in cars:
        if not car.reached_goal:
            lane_map.setdefault(car.lane, []).append(car.car_id)
    lane_occupancies = [
        LaneOccupancyData(lane=lane, carIds=ids)
        for lane, ids in sorted(lane_map.items())
    ]

    return cars_data, lane_occupancies


class OverflowEnvironment(Environment):
    """
    Autonomous vehicle fleet oversight environment.

    A 2D road grid with N cars. Car 0 is the LLM agent, others follow
    scripted rules. The observer detects crashes and near-misses and
    computes rewards based on safety.
    """

    def __init__(self):
        super().__init__()
        self._state = OverflowState(episode_id=str(uuid4()))
        self._cars: List[Car] = []
        self._rng = random.Random()
        self._done = False

    def _build_observation(
        self,
        incident_report: str,
        reward: float,
        proximities: Optional[List[ProximityData]] = None,
    ) -> OverflowObservation:
        """Build a full observation with text + structured data."""
        agent = self._cars[0]
        scene = _generate_scene_description(agent, self._cars)
        prox = proximities or []
        cars_data, lane_occ = _build_structured_data(self._cars, prox)

        return OverflowObservation(
            scene_description=scene,
            incident_report=incident_report,
            done=self._done,
            reward=reward,
            cars=cars_data,
            proximities=prox,
            lane_occupancies=lane_occ,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OverflowObservation:
        """Reset the environment: create road and spawn cars."""
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        self._state = OverflowState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            crash_count=0,
            near_miss_count=0,
            cars_reached_goal=0,
            total_cars=NUM_CARS,
        )
        self._done = False

        # Spawn cars with random positions, speeds, lanes, and goals
        self._cars = []

        for i in range(NUM_CARS):
            # Ensure no two cars spawn within crash distance
            for _attempt in range(100):
                lane = self._rng.randint(1, NUM_LANES)
                position = float(self._rng.randint(10, 80))
                too_close = False
                for existing in self._cars:
                    lane_diff = abs(lane - existing.lane) * 10.0
                    pos_diff = abs(position - existing.position)
                    dist = math.sqrt(lane_diff**2 + pos_diff**2)
                    if dist < CRASH_DISTANCE * 2:
                        too_close = True
                        break
                if not too_close:
                    break

            speed = float(self._rng.randint(40, 70))
            goal = float(self._rng.randint(160, 195))

            self._cars.append(
                Car(
                    car_id=i,
                    lane=lane,
                    position=position,
                    speed=speed,
                    goal_position=goal,
                    is_agent=(i == 0),
                    prev_speed=speed,  # no delta on first step
                )
            )

        return self._build_observation(incident_report="", reward=0.0)

    def step(
        self,
        action: OverflowAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OverflowObservation:
        """Execute one simulation step."""
        if self._done:
            return self._build_observation(
                incident_report="Episode is over. Call reset() to start a new one.",
                reward=0.0,
            )

        self._state.step_count += 1
        reward = 0.0
        incidents = []

        # Snapshot previous speeds for acceleration tracking
        for car in self._cars:
            car.prev_speed = car.speed

        # 1. Parse and apply the agent's action to Car 0
        decision = _parse_decision(action)
        _apply_action(self._cars[0], decision)

        # 2. Compute and apply scripted actions for Cars 1-N
        for car in self._cars[1:]:
            if car.reached_goal:
                continue
            scripted_decision = _scripted_car_action(car, self._cars, self._rng)
            _apply_action(car, scripted_decision)

        # 3. Move all cars forward based on speed (speed is in units/step, scaled down)
        for car in self._cars:
            if car.reached_goal:
                continue
            car.position += car.speed * 0.1  # scale factor for reasonable movement

        # 4. Collision detection (pairwise)
        agent_crash = False
        proximity_list: List[ProximityData] = []
        active_cars = [c for c in self._cars if not c.reached_goal]
        agent_id = self._cars[0].car_id
        for i in range(len(active_cars)):
            for j in range(i + 1, len(active_cars)):
                dist = active_cars[i].distance_to(active_cars[j])
                involves_agent = active_cars[i].car_id == agent_id or active_cars[j].car_id == agent_id
                if dist < CRASH_DISTANCE:
                    self._state.crash_count += 1
                    proximity_list.append(
                        ProximityData(
                            carA=active_cars[i].car_id,
                            carB=active_cars[j].car_id,
                            distance=round(dist, 2),
                        )
                    )
                    incidents.append(
                        f"CRASH between Car {active_cars[i].car_id} and Car {active_cars[j].car_id}! "
                        f"(distance: {dist:.1f})"
                    )
                    if involves_agent:
                        agent_crash = True
                elif dist < NEAR_MISS_DISTANCE:
                    self._state.near_miss_count += 1
                    # Only penalize near misses involving the agent
                    if involves_agent:
                        reward += REWARD_NEAR_MISS
                    proximity_list.append(
                        ProximityData(
                            carA=active_cars[i].car_id,
                            carB=active_cars[j].car_id,
                            distance=round(dist, 2),
                        )
                    )
                    incidents.append(
                        f"NEAR MISS between Car {active_cars[i].car_id} and Car {active_cars[j].car_id} "
                        f"(distance: {dist:.1f})"
                    )

        if agent_crash:
            reward += REWARD_CRASH
            self._done = True
        else:
            # 5. Goal check for agent car
            agent = self._cars[0]
            if agent.position >= agent.goal_position:
                agent.reached_goal = True
                self._state.cars_reached_goal += 1
                reward += REWARD_REACHED_GOAL
                incidents.append(
                    f"Car 0 reached its goal at position {agent.goal_position:.0f}!"
                )
                self._done = True

            # Check goal for scripted cars too (for state tracking)
            for car in self._cars[1:]:
                if not car.reached_goal and car.position >= car.goal_position:
                    car.reached_goal = True
                    self._state.cars_reached_goal += 1

            # 6. Safe step bonus (no crash, agent still active)
            if not self._done:
                reward += REWARD_SAFE_STEP

        # 7. Reasoning quality bonus
        reasoning_bonus = _compute_reasoning_bonus(action.reasoning)
        reward += reasoning_bonus

        # 8. Max steps check
        if self._state.step_count >= MAX_STEPS and not self._done:
            self._done = True
            incidents.append(f"Maximum steps ({MAX_STEPS}) reached.")

        incident_report = (
            "\n".join(incidents) if incidents else "Observer: No incidents this step."
        )

        return self._build_observation(
            incident_report=incident_report,
            reward=reward,
            proximities=proximity_list,
        )

    @property
    def state(self) -> OverflowState:
        """Get the current environment state."""
        return self._state
