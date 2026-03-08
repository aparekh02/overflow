# OpenENV — Waymo Incident RL Environment

A novel RL environment that **bidirectionally connects a live 3D Waymo simulation** with a Gymnasium-compatible multi-agent RL system. Incident tickets become RL episodes; real Waymo sensor data becomes physics constraints; and a review agent learns optimal avoidance policies visualized in real-time 3D.

---

## Architecture

```
INCIDENT TICKET (natural language)
         │
    IncidentParser ← Claude API (or rule-based fallback)
         │
   OPENENV CORE (Python)
     ├── WaymoConstraints  ← real LiDAR point density → friction, visibility
     ├── MultiAgent System
     │     ├── ReviewAgent  (RL ego, bicycle model + constraint-adaptive physics)
     │     ├── TrafficAgents (IDM with Waymo-constraint-adaptive headways)
     │     └── IncidentAgent (scripted replay of original incident)
     ├── RewardEngine      (safety + efficiency + comfort + incident prevention)
     └── Renderer          (scene → JSON)
              │ WebSocket ws://localhost:8765 (bidirectional)
         MIDDLEWARE
              │
    3D SIM (React + Three.js)
     ├── AgentOverlay   — RL agents over Waymo LiDAR point cloud
     ├── IncidentPanel  — ticket submission UI
     └── RLStatsPanel   — reward curves, constraint bars, action breakdown
```

## What Makes This Novel

1. **Incident-seeded RL**: Instead of random initial states, real incident reports create realistic, historically-grounded scenarios. The RL agent trains on the exact geometry and conditions of real near-misses.

2. **Waymo constraints as RL physics**: LiDAR point density → visibility estimate. Mean return intensity → road friction coefficient. These ground the RL physics in real sensor measurements rather than hand-tuned parameters.

3. **Bidirectional 3D-RL coupling**: The 3D visualization isn't passive telemetry — it's an active part of the training loop. Users can:
   - Submit incident tickets from the browser
   - Pause/resume to inspect agent decisions
   - Inject ghost actions (manual overrides) for counterfactual exploration
   - Forward live Waymo frame data to update RL physics constraints

4. **Constraint-adaptive reward**: Identical behaviour earns different rewards based on inferred conditions. Smooth braking in rain is rewarded more than on dry roads.

5. **Attention-weighted visualization**: The 3D overlay shows which agents the review agent is attending to (softmax over inverse TTC), making the policy interpretable.

---

## Setup

```bash
# Python backend
pip install -r openenv/requirements.txt

# 3D simulation frontend
cd 3D_sim && npm install
```

## Running

```bash
# Terminal 1: Python RL backend + WebSocket server
python openenv/run_demo.py

# Terminal 2: 3D simulation
cd 3D_sim && npm run dev
```

Then open the 3D sim, click **OpenENV ●** in the top-right, and use the **Incident Panel** (bottom-right) to submit tickets.

## Headless Training

```bash
python openenv/run_demo.py --headless --episodes 500
```

## Integrating a Trained Policy (Stable-Baselines3)

```python
from stable_baselines3 import PPO
from openenv import WaymoEnv
from openenv.middleware import run_env_with_middleware
import asyncio

env = WaymoEnv(render_mode="websocket")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Visualize in 3D
asyncio.run(run_env_with_middleware(env, policy=model))
```

## Message Protocol (WebSocket)

### Browser → Python
| type | payload | effect |
|------|---------|--------|
| `load_incident` | `{ticket: {id, text, severity}}` | Parse ticket, start RL episode |
| `pause` | — | Freeze simulation |
| `resume` | — | Continue simulation |
| `reset` | — | New episode, same scenario |
| `ghost_action` | `{action: [steer, throttle, brake]}` | Manual override for 1 step |
| `waymo_frame` | `{pointCount, boxes, avgIntensity}` | Update live physics constraints |

### Python → Browser
| type | payload | effect |
|------|---------|--------|
| `scene_update` | Full `RLSceneState` | Update 3D agent positions |
| `episode_start` | `{incident_type, constraints}` | Reset UI for new episode |
| `episode_end` | `{total_reward, steps}` | Show episode summary |

## Observation Space (82-dim)

| Slice | Dimensions | Description |
|-------|-----------|-------------|
| 0–5 | 6 | Ego state: x, y, vx, vy, yaw, yaw_rate (normalised) |
| 6–75 | 70 | 10 nearest agents × [rel_x, rel_y, rel_vx, rel_vy, rel_yaw, type, ttc] |
| 76–80 | 5 | Waymo constraints: friction, visibility, rain, fog, traffic_density |
| 81 | 1 | Episode progress (step / max_steps) |

## Action Space (3-dim)

| Dim | Range | Description |
|-----|-------|-------------|
| 0 | [-1, 1] | Steering (left → right) |
| 1 | [0, 1] | Throttle |
| 2 | [0, 1] | Brake |
