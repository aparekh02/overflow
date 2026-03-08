# Overflow Environment — Low-Level Design Document

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File-by-File Breakdown](#2-file-by-file-breakdown)
3. [Data Models (Wire Format)](#3-data-models-wire-format)
4. [Simulation Internals](#4-simulation-internals)
5. [Step-by-Step Execution Pipeline](#5-step-by-step-execution-pipeline)
6. [Distance and Collision Model](#6-distance-and-collision-model)
7. [Reward Function — Complete Breakdown](#7-reward-function--complete-breakdown)
8. [Scripted Car AI](#8-scripted-car-ai)
9. [Action Parsing — How LLM Output Becomes a Decision](#9-action-parsing--how-llm-output-becomes-a-decision)
10. [Observation Text Format](#10-observation-text-format)
11. [Server Protocol — What Training Scripts Must Send](#11-server-protocol--what-training-scripts-must-send)
12. [Training Integration — GRPO / TRL](#12-training-integration--grpo--trl)
13. [Episode Dynamics and RL Characteristics](#13-episode-dynamics-and-rl-characteristics)
14. [Configuration Constants](#14-configuration-constants)
15. [Docker and Deployment](#15-docker-and-deployment)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Training Script (GRPO)                │
│  calls reset(), reads observation, calls step(action)   │
└────────────────────────┬────────────────────────────────┘
                         │ WebSocket (persistent session)
                         │ JSON messages over ws://host:8000/ws
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Server (app.py)                     │
│  create_app(OverflowEnvironment, OverflowAction,        │
│             OverflowObservation)                         │
│                                                         │
│  Endpoints:                                             │
│    WS  /ws       ← primary (stateful session)           │
│    POST /reset   ← HTTP fallback                        │
│    POST /step    ← HTTP fallback                        │
│    GET  /state   ← HTTP fallback                        │
│    GET  /health  ← health check                         │
│    GET  /schema  ← JSON schemas for action/obs/state    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         OverflowEnvironment (pure Python)                │
│                                                         │
│  Internal state:                                        │
│    _cars: List[Car]     (5 cars, car 0 = agent)         │
│    _state: OverflowState (episode tracking)             │
│    _rng: random.Random  (seeded per episode)            │
│    _done: bool                                          │
│                                                         │
│  Methods:                                               │
│    reset(seed, episode_id) → OverflowObservation        │
│    step(OverflowAction)    → OverflowObservation        │
│    state (property)        → OverflowState              │
└─────────────────────────────────────────────────────────┘
```

**Key invariant**: The training loop calls `reset()`. The LLM agent only calls `step()` via the training harness. Agents can never reset — if they could undo consequences, training breaks.

**Session model**: Each WebSocket connection gets its own `OverflowEnvironment` instance. The `create_app` function receives the class (factory), not an instance. When a WebSocket connects, the server instantiates a fresh environment for that session.

---

## 2. File-by-File Breakdown

### `models.py` — Pydantic data models

Defines three classes inheriting from OpenEnv core types:

| Class | Parent | Purpose |
|-------|--------|---------|
| `OverflowAction(Action)` | `openenv.core.env_server.types.Action` | What the LLM sends each step |
| `OverflowObservation(Observation)` | `openenv.core.env_server.types.Observation` | What the environment returns |
| `OverflowState(State)` | `openenv.core.env_server.types.State` | Internal state exposed via `/state` |

All three are Pydantic `BaseModel` subclasses. The parent classes provide `metadata: Dict[str, Any]` (on Action and Observation) and `episode_id: str`, `step_count: int` (on State). The parent `Observation` provides `done: bool` and `reward: float | None`.

### `server/overflow_environment.py` — All game logic

Contains:
- `Car` dataclass — per-car state (id, lane, position, speed, goal, is_agent, reached_goal)
- `_parse_decision()` — tolerant action parser
- `_compute_reasoning_bonus()` — reasoning quality scorer
- `_scripted_car_action()` — NPC car AI
- `_apply_action()` — mutates a car's speed/lane
- `_generate_scene_description()` — builds the text observation
- `OverflowEnvironment(Environment)` — the main class with `reset()`, `step()`, `state`

### `server/app.py` — FastAPI wiring

Introspects `create_app` to determine if it expects a factory (class) or an instance. Passes `OverflowEnvironment`, `OverflowAction`, `OverflowObservation` to `create_app`. The resulting `app` object is what uvicorn serves.

### `client.py` — WebSocket client

`OverflowEnv(EnvClient[OverflowAction, OverflowObservation, OverflowState])` with three required methods:
- `_step_payload(action)` — serializes `OverflowAction` to `{"decision": ..., "reasoning": ...}`
- `_parse_result(payload)` — deserializes server JSON into `StepResult[OverflowObservation]`
- `_parse_state(payload)` — deserializes server JSON into `OverflowState`

### `__init__.py` — Public API

Exports: `OverflowAction`, `OverflowObservation`, `OverflowState`, `OverflowEnv`.

---

## 3. Data Models (Wire Format)

### OverflowAction — What the training script sends to `/step`

```json
{
  "action": {
    "decision": "brake",
    "reasoning": "Car 3 is ahead in my lane, 15 units away, going slower. I should brake."
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `decision` | `str` | No | `"maintain"` | One of: `accelerate`, `brake`, `lane_change_left`, `lane_change_right`, `maintain` |
| `reasoning` | `str` | No | `""` | Free-text chain-of-thought. Affects reward via reasoning bonus (0.0–2.0). |

The `decision` field is parsed tolerantly — see Section 9.

### OverflowObservation — What the server returns

Each observation carries **both** text (for the LLM) and structured data (for the frontend/viz).

```json
{
  "observation": {
    "scene_description": "You are Car 0 in lane 2, position 45, speed 60.\n...",
    "incident_report": "Observer: No incidents this step.",
    "done": false,
    "reward": 1.45,
    "cars": [
      {"carId": 0, "lane": 2, "position": {"x": 45.0, "y": 7.4}, "speed": 60.0, "acceleration": 5.0},
      {"carId": 1, "lane": 1, "position": {"x": 43.0, "y": 3.7}, "speed": 55.0, "acceleration": 0.0}
    ],
    "proximities": [
      {"carA": 0, "carB": 1, "distance": 10.5}
    ],
    "lane_occupancies": [
      {"lane": 1, "carIds": [1]},
      {"lane": 2, "carIds": [0]}
    ],
    "metadata": {}
  },
  "reward": 1.45,
  "done": false
}
```

#### Text fields (for the LLM)

| Field | Type | Description |
|-------|------|-------------|
| `scene_description` | `str` | Multi-line text describing all cars. This is what the LLM reads. |
| `incident_report` | `str` | Observer output. Either `"Observer: No incidents this step."` or a list of CRASH/NEAR MISS events. |

#### Structured fields (for the frontend — compatible with Overflow frontend types)

| Field | Type | Frontend equivalent |
|-------|------|---------------------|
| `cars` | `CarStateData[]` | `CarState[]` — `{carId, lane, position: {x, y}, speed, acceleration}` |
| `proximities` | `ProximityData[]` | `{carA, carB, distance}[]` — pairwise distances for close cars |
| `lane_occupancies` | `LaneOccupancyData[]` | `{lane, carIds}[]` — which cars are in each lane |

Position `y` is computed as `lane * 3.7` (lane width in metres), matching the frontend's `makeCar` convention.

#### Common fields

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | `true` if episode ended (crash, goal reached, or max steps). |
| `reward` | `float` | Scalar reward for this step. Sum of all reward components. |

The `reward` and `done` appear both inside `observation` and at the top level of the response (OpenEnv convention).

### OverflowState — What `/state` returns

```json
{
  "episode_id": "a1b2c3d4-...",
  "step_count": 17,
  "crash_count": 0,
  "near_miss_count": 23,
  "cars_reached_goal": 1,
  "total_cars": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | `str` | UUID for this episode. Set on `reset()`. |
| `step_count` | `int` | How many `step()` calls have been made. |
| `crash_count` | `int` | Cumulative crash events (each pair counts as 1). |
| `near_miss_count` | `int` | Cumulative near-miss events (each pair counts as 1). |
| `cars_reached_goal` | `int` | How many cars (including scripted) reached their goal. |
| `total_cars` | `int` | Always 5. |

---

## 4. Simulation Internals

### The Road

- 3 lanes, numbered 1, 2, 3 (1 = leftmost, 3 = rightmost)
- Road length: ~200 position units
- No wrapping — cars move forward from low positions toward high positions
- Lanes are conceptually 10 units apart for distance calculations

### Car State

Each car is a `Car` dataclass:

```python
@dataclass
class Car:
    car_id: int          # 0 = agent, 1–4 = scripted
    lane: int            # 1, 2, or 3
    position: float      # 0.0 to ~200.0 (along the road)
    speed: float         # 20.0 to 90.0
    goal_position: float # 160.0 to 195.0
    is_agent: bool       # True only for car 0
    reached_goal: bool   # True once position >= goal_position
```

### Initialization (reset)

On `reset(seed=N)`:
1. A `random.Random(seed)` RNG is created (deterministic replays if same seed).
2. 5 cars are spawned:
   - **Lane**: random 1–3
   - **Position**: random 10–80 (spread across the first half of the road)
   - **Speed**: random 40–70
   - **Goal**: random 160–195
3. No two cars occupy the same 10-unit segment in the same lane at spawn (deconflicted via `(lane, position // 10)` hash).
4. Car 0 is the agent. Cars 1–4 are scripted.

### Movement

Each step, every active (non-goal-reached) car moves forward:

```
car.position += car.speed * 0.1
```

This means a car at speed 60 moves 6.0 units per step. At that rate, traversing the ~120-unit gap from starting zone (10–80) to goal zone (160–195) takes roughly 20 steps. Faster cars (speed 90) move 9.0 units/step and reach goals sooner.

---

## 5. Step-by-Step Execution Pipeline

When `step(action)` is called, the following happens **in this exact order**:

```
1. GUARD: if episode is already done → return stale observation with reward=0.0
2. INCREMENT step_count
3. PARSE the agent's action → one of {accelerate, brake, lane_change_left, lane_change_right, maintain}
4. APPLY action to Car 0 (mutate speed or lane)
5. COMPUTE scripted actions for Cars 1–4 and APPLY them
6. MOVE all active cars forward: position += speed * 0.1
7. COLLISION DETECTION (pairwise over all active cars):
   - distance < 5.0 → CRASH (reward -5.0, episode ends)
   - distance < 15.0 → NEAR MISS (reward -1.0 per pair)
8. If no crash:
   a. Check if Car 0 reached its goal → reward +3.0, episode ends
   b. Check if scripted cars reached their goals (state tracking only)
   c. If episode not ending → SAFE STEP bonus: reward +0.5
9. REASONING BONUS: score the reasoning text → reward +0.0 to +2.0
10. MAX STEPS CHECK: if step_count >= 100 → episode ends
11. BUILD observation text and incident report
12. RETURN OverflowObservation(scene_description, incident_report, done, reward)
```

**Important ordering detail**: Actions are applied (step 4–5) **before** movement (step 6). This means the agent's speed/lane change takes effect for this step's movement. Collision detection (step 7) happens **after** movement, on the new positions.

**Reward accumulation within a step**: A single step's reward is the **sum** of all applicable components. For example, if there are 2 near-miss pairs and the agent is still alive with good reasoning, the reward could be: `(-1.0 * 2) + 0.5 + 1.5 = -1.0`.

---

## 6. Distance and Collision Model

Distance between two cars uses a weighted Euclidean formula:

```python
def distance_to(self, other):
    lane_diff = abs(self.lane - other.lane) * 10.0
    pos_diff = abs(self.position - other.position)
    return sqrt(lane_diff**2 + pos_diff**2)
```

**Implications**:
- Two cars in the **same lane** at positions 45 and 50: distance = 5.0 (exactly at crash threshold)
- Two cars in **adjacent lanes** (e.g., lane 1 and lane 2) at the same position: distance = 10.0 (near miss, not crash)
- Two cars **two lanes apart** at the same position: distance = 20.0 (safe, no incident)
- Two cars in adjacent lanes, 10 units apart longitudinally: distance = sqrt(100 + 100) ≈ 14.1 (near miss)

**Key insight for the agent**: Lane changes provide safety via the 10-unit lane multiplier. Staying in the same lane as another car is the primary crash risk. The agent should use lane changes proactively to maintain distance from cars in its lane.

### Collision detection scope

Detection is **pairwise over ALL active cars**, not just agent-involving pairs. If Car 2 and Car 3 crash, the episode still ends with -5.0 reward. This means the agent is implicitly responsible for the overall traffic flow — it should avoid creating situations where its actions cause chain reactions among scripted cars.

---

## 7. Reward Function — Complete Breakdown

### Per-step reward components

| Component | Value | Condition | Stacks? |
|-----------|-------|-----------|---------|
| **Crash** | -5.0 | Any pair distance < 5.0 | Once (episode ends) |
| **Near miss** | -1.0 | Per pair with distance < 15.0 | Yes, per pair (can be -2.0, -3.0, etc.) |
| **Safe step** | +0.5 | No crash and episode not ending this step | Once per step |
| **Goal reached** | +3.0 | Car 0's position >= goal_position | Once (episode ends) |
| **Reasoning bonus** | +0.0 to +2.0 | Based on reasoning text quality | Once per step |

### Reasoning bonus scoring

The bonus has three sub-components capped at 2.0 total:

**Length bonus** (up to 0.5):
- `len > 20` chars → +0.2
- `len > 50` chars → +0.15
- `len > 100` chars → +0.15

**Keyword awareness** (up to 1.0):
Each keyword found → +0.2, capped at 1.0. Keywords: `ahead`, `behind`, `lane`, `speed`, `distance`, `safe`, `danger`, `collision`, `brake`, `gap`, `close`, `slow`, `fast`, `goal`, `position`.

**Structure bonus** (up to 0.5):
- Contains `<think>` or `because` → +0.25
- Contains `therefore`, `so i should`, `best option`, or `i will` → +0.25

### Typical reward ranges per step

| Scenario | Typical reward |
|----------|---------------|
| Safe step, no reasoning | +0.5 |
| Safe step, decent reasoning | +1.0 to +2.0 |
| Safe step, excellent reasoning | +2.0 to +2.5 |
| 1 near miss, decent reasoning | -0.5 to +0.5 |
| 2 near misses, decent reasoning | -1.5 to -0.5 |
| Crash (any) | -5.0 + reasoning bonus |
| Goal reached, good reasoning | +3.0 + reasoning bonus |

### Episode return (total reward) characteristics

Based on testing with seed=42:
- A "maintain" strategy with decent reasoning gets ~1.1 per step × ~17 steps ≈ 18.7 total, minus near-miss penalties
- Aggressive "accelerate" strategies reach the goal faster but accumulate more near misses
- Smart strategies that use lane changes and braking to avoid near misses can maximize total reward

---

## 8. Scripted Car AI

Cars 1–4 use `_scripted_car_action(car, all_cars, rng)`:

```
1. Find the nearest car AHEAD in the SAME LANE
2. If that car is < 20 units ahead → "brake"
3. Else if speed < 60 and 10% random chance → "accelerate"
4. Else if 5% random chance → lane change (random left/right, respecting boundaries)
5. Else → "maintain"
```

**Characteristics**:
- Scripted cars are mostly passive — they maintain speed
- They brake reactively when blocked (but only for same-lane, ahead)
- They rarely change lanes (5% per step), making their behavior somewhat predictable
- They never intentionally avoid the agent — only react to cars directly ahead
- They can accumulate near misses and crashes among themselves

This creates an environment where a smart agent can learn to navigate around largely predictable but occasionally erratic traffic.

---

## 9. Action Parsing — How LLM Output Becomes a Decision

The parser `_parse_decision(action)` is intentionally forgiving. It tries three strategies in order:

### Strategy 1: Direct field match
```python
decision = action.decision.strip().lower().replace(" ", "_")
# If it's one of {accelerate, brake, lane_change_left, lane_change_right, maintain} → use it
```

### Strategy 2: XML tag extraction
```python
text = f"{action.decision} {action.reasoning}".lower()
match = re.search(r"<action>\s*(\w+)\s*</action>", text)
# If found and valid → use it
```

This handles LLM outputs like:
```
decision: "think about it"
reasoning: "<think>Car ahead is close</think><action>brake</action>"
```

### Strategy 3: Keyword scan
```python
for v in {"accelerate", "brake", "lane_change_left", "lane_change_right", "maintain"}:
    if v in text:
        return v
```

This handles outputs like `decision: "I want to accelerate now"`.

### Fallback
If nothing matches → `"maintain"` (safe default).

**For training scripts**: The cleanest format is to put the exact decision string in the `decision` field. The tolerant parsing is there so that LLMs in early training (before they learn the format) still produce valid actions rather than crashing.

---

## 10. Observation Text Format

The `scene_description` field is a multi-line string that the LLM reads as its input. Example:

```
You are Car 0 in lane 2, position 45, speed 60.
Goal: reach position 180.
Nearby cars:
- Car 1: lane 1, position 43, speed 55
- Car 2: lane 3, position 48, speed 70
- Car 3: lane 2, position 65, speed 50 [AHEAD IN YOUR LANE - 20 units away]
- Car 4: lane 1, position 30, speed 65
```

**Annotations added**:
- `[AHEAD IN YOUR LANE - N units away]` — same lane, ahead of agent
- `[BEHIND IN YOUR LANE - N units away]` — same lane, behind agent
- `[REACHED GOAL]` — car has finished

The `incident_report` is separate:
- No incidents: `"Observer: No incidents this step."`
- With incidents: One line per event, e.g.:
  ```
  NEAR MISS between Car 0 and Car 3 (distance: 12.5)
  Car 0 reached its goal at position 180!
  ```

---

## 11. Server Protocol — What Training Scripts Must Send

### WebSocket Protocol (Primary — for training)

Connect to `ws://host:8000/ws`. All messages are JSON.

#### Reset

**Send:**
```json
{"type": "reset", "data": {"seed": 42}}
```

`data` can include `seed` (int) and/or `episode_id` (str). Both are optional.

**Receive:**
```json
{
  "type": "observation",
  "data": {
    "observation": {
      "scene_description": "You are Car 0 in lane 3, position 24, speed 40.\n...",
      "incident_report": "",
      "done": false,
      "reward": 0.0,
      "metadata": {}
    },
    "reward": 0.0,
    "done": false
  }
}
```

#### Step

**Send:**
```json
{
  "type": "step",
  "data": {
    "decision": "brake",
    "reasoning": "Car ahead is close, braking to maintain safe distance."
  }
}
```

**Receive:**
```json
{
  "type": "observation",
  "data": {
    "observation": {
      "scene_description": "You are Car 0 in lane 3, position 27, speed 35.\n...",
      "incident_report": "Observer: No incidents this step.",
      "done": false,
      "reward": 2.25,
      "metadata": {}
    },
    "reward": 2.25,
    "done": false
  }
}
```

#### State

**Send:**
```json
{"type": "state"}
```

**Receive:**
```json
{
  "type": "state",
  "data": {
    "episode_id": "a1b2c3d4-...",
    "step_count": 7,
    "crash_count": 0,
    "near_miss_count": 3,
    "cars_reached_goal": 0,
    "total_cars": 5
  }
}
```

#### Close

**Send:**
```json
{"type": "close"}
```

### HTTP Protocol (Fallback — for simple testing)

Note: The HTTP API creates a **new environment instance per endpoint** in factory mode. The `/reset` and `/step` calls hit separate instances. Use WebSocket for stateful multi-step episodes.

```
POST /reset     Body: {"seed": 42}              → {"observation": {...}, "reward": 0.0, "done": false}
POST /step      Body: {"action": {"decision": "brake", "reasoning": "..."}}  → {"observation": {...}, "reward": ..., "done": ...}
GET  /state     → {"episode_id": ..., "step_count": ..., ...}
GET  /health    → {"status": "healthy"}
GET  /schema    → {"action": {...}, "observation": {...}, "state": {...}}
```

### Using the Python Client

```python
from overflow_env import OverflowEnv, OverflowAction

with OverflowEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=42)
    # result is StepResult[OverflowObservation]
    # result.observation.scene_description  — the text for the LLM
    # result.observation.incident_report    — observer output
    # result.reward                         — float
    # result.done                           — bool

    while not result.done:
        # Feed scene_description to LLM, get decision + reasoning back
        llm_decision, llm_reasoning = call_llm(result.observation.scene_description)

        action = OverflowAction(decision=llm_decision, reasoning=llm_reasoning)
        result = env.step(action)

    # Episode over
    state = env.state()
    print(f"Steps: {state.step_count}, Crashes: {state.crash_count}")
```

---

## 12. Training Integration — GRPO / TRL

### System prompt for the LLM

The training script should set a system prompt like:

```
You are an autonomous vehicle controller. Each turn you receive a traffic scene description.
You must output a driving decision and your reasoning.

Available decisions: accelerate, brake, lane_change_left, lane_change_right, maintain

Output format:
<think>Your reasoning about the traffic situation</think>
<action>your_decision</action>
```

### What the training loop does each episode

```python
# 1. Reset environment
result = env.reset(seed=episode_seed)

# 2. Build initial prompt
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": result.observation.scene_description}
]

trajectory_rewards = []

# 3. Loop until done
while not result.done:
    # 3a. Get LLM completion
    completion = model.generate(messages)  # the text the LLM produces

    # 3b. Parse LLM output into action
    #     The environment's parser is tolerant, but for clean training
    #     you might also parse on the client side
    action = OverflowAction(
        decision=extract_decision(completion),
        reasoning=completion  # pass full text as reasoning
    )

    # 3c. Step
    result = env.step(action)
    trajectory_rewards.append(result.reward)

    # 3d. Append to conversation for next turn
    messages.append({"role": "assistant", "content": completion})
    messages.append({"role": "user", "content": (
        result.observation.scene_description + "\n" +
        result.observation.incident_report
    )})

# 4. Compute episode return for GRPO
episode_return = sum(trajectory_rewards)
```

### GRPO reward signal

For GRPO (Group Relative Policy Optimization), the reward signal is the **episode return** — the sum of all per-step rewards across the episode. The environment is designed so that:

- **Positive episode returns** (agent reached goal safely with good reasoning) indicate good behavior
- **Negative episode returns** (crashes, many near misses) indicate bad behavior
- The **reasoning bonus** provides per-step reward shaping that encourages the LLM to explain its thinking, which improves interpretability and can speed up learning

### Constructing the reward for TRL

If using TRL's `OnlineDPOTrainer` or `GRPOTrainer`:

```python
# Per-step reward is already in result.reward
# For token-level reward (assign to last token of each turn):
rewards_per_turn = trajectory_rewards  # list of floats, one per step

# For episode-level reward (assign to last token of episode):
episode_reward = sum(trajectory_rewards)
```

---

## 13. Episode Dynamics and RL Characteristics

### Episode length distribution

| Scenario | Typical length |
|----------|---------------|
| Aggressive accelerate → goal | 12–20 steps |
| Moderate maintain → goal | 18–30 steps |
| Conservative braking | 30–50+ steps |
| Crash (bad luck or bad driving) | 5–15 steps |
| Max steps timeout | 100 steps |

### What makes this environment learnable

1. **Clear signal**: Crashes give -5.0, goals give +3.0. The agent quickly learns that crashing is bad and reaching the goal is good.

2. **Gradual improvement**: Near misses (-1.0 each) provide intermediate signal. An agent that learns to avoid near misses gets higher returns than one that just avoids crashes.

3. **Speed-accuracy tradeoff**: Accelerating reaches the goal faster (more +3.0 episodes) but increases crash/near-miss risk. The optimal policy is to accelerate when safe and brake/change lanes when needed.

4. **Reasoning is rewarded**: The reasoning bonus (up to +2.0/step) means that over a 20-step episode, reasoning alone can contribute up to +40.0. This incentivizes the LLM to produce structured, situation-aware reasoning.

5. **Stochasticity**: Scripted cars have random elements (10% accelerate, 5% lane change). This means the same seed produces the same episode, but different seeds produce different traffic patterns, forcing the agent to generalize.

6. **All-pairs collision**: The agent is rewarded/punished for the entire traffic system, not just its own car. This means the agent must be aware of the overall traffic flow.

### Typical learning progression

1. **Random policy**: Mostly "maintain", occasional random actions. Episode return: 0 to 15 (depending on luck).
2. **Basic safety**: Agent learns to brake when car ahead is close. Fewer crashes, more goals. Episode return: 10 to 25.
3. **Strategic driving**: Agent learns to change lanes proactively, accelerate when clear, brake early. Episode return: 20 to 40.
4. **Optimized reasoning**: Agent produces structured reasoning with relevant keywords, maximizing the reasoning bonus. Episode return: 30 to 60.

### Reproducibility

Passing `seed=N` to `reset()` produces deterministic initial conditions and scripted car behavior (since the `random.Random` instance is seeded). The same seed + same agent actions = same trajectory. This is critical for GRPO, which compares multiple rollouts of the same prompt.

---

## 14. Configuration Constants

All constants are defined at the top of `server/overflow_environment.py`:

```python
NUM_LANES = 3              # Number of road lanes
ROAD_LENGTH = 200          # Conceptual road length (units)
NUM_CARS = 5               # Total cars (1 agent + 4 scripted)
MAX_STEPS = 100            # Maximum steps before forced termination
CRASH_DISTANCE = 5.0       # Distance threshold for crash
NEAR_MISS_DISTANCE = 15.0  # Distance threshold for near miss

REWARD_CRASH = -5.0        # Reward for any crash
REWARD_NEAR_MISS = -1.0    # Reward per near-miss pair
REWARD_SAFE_STEP = 0.5     # Reward for surviving a step
REWARD_REACHED_GOAL = 3.0  # Reward for reaching goal
REWARD_REASONING_MAX = 2.0 # Maximum reasoning quality bonus

MIN_SPEED = 20             # Minimum car speed
MAX_SPEED = 90             # Maximum car speed
SPEED_DELTA = 5            # Speed change per accelerate/brake
```

To tune difficulty:
- **Easier**: Increase `CRASH_DISTANCE` and `NEAR_MISS_DISTANCE`, decrease `NUM_CARS`, widen starting positions
- **Harder**: Decrease distances, increase `NUM_CARS`, narrow starting positions, increase `MAX_SPEED`
- **Longer episodes**: Increase `ROAD_LENGTH` or decrease starting speeds
- **More reasoning incentive**: Increase `REWARD_REASONING_MAX`

---

## 15. Docker and Deployment

### Local development

```bash
uvicorn overflow_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker build

```bash
# From the overflow_env/ directory:
docker build -t overflow-env:latest -f server/Dockerfile .
docker run -p 8000:8000 overflow-env:latest
```

The Dockerfile uses a multi-stage build:
1. **Builder stage**: Installs dependencies with `uv sync` into a `.venv`
2. **Runtime stage**: Copies the `.venv` and source code, runs uvicorn

Base image: `ghcr.io/meta-pytorch/openenv-base:latest`

### Push to HuggingFace Spaces

```bash
openenv push --repo-id username/overflow-env
```

### Connect from training script

```python
# Local
env = OverflowEnv(base_url="http://localhost:8000")

# Docker
env = OverflowEnv.from_docker_image("overflow-env:latest")

# HuggingFace Space
env = OverflowEnv.from_env("username/overflow-env")
```

### openenv.yaml manifest

```yaml
spec_version: 1
name: overflow_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

This tells OpenEnv tooling how to find and run the environment.
