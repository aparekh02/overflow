---
title: Overflow Environment Server
emoji: 🚗
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Overflow Environment

An autonomous vehicle fleet oversight environment for [OpenEnv](https://github.com/meta-pytorch/OpenEnv).

## Overview

A 2D road grid with N cars. One car (Car 0) is controlled by an LLM agent, while other cars follow simple scripted driving rules. An observer detects crashes and near-misses each step and computes rewards based on safety.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

```python
from overflow_env import OverflowEnv, OverflowAction

async with OverflowEnv(base_url="http://localhost:8000") as env:
    result = await env.reset()
    print(result.observation.scene_description)

    action = OverflowAction(decision="maintain", reasoning="Road is clear ahead.")
    result = await env.step(action)
    print(result.observation.incident_report)
    print(f"Reward: {result.reward}, Done: {result.done}")
```

## Action Space

| Decision | Effect |
|----------|--------|
| `accelerate` | Increase speed by 5 |
| `brake` | Decrease speed by 5 |
| `lane_change_left` | Move to left lane |
| `lane_change_right` | Move to right lane |
| `maintain` | Keep current speed and lane |

## Reward Structure

| Event | Reward |
|-------|--------|
| Crash (distance < 5) | -5.0 |
| Near miss (distance < 15) | -1.0 |
| Safe step toward goal | +0.5 |
| Reached goal | +3.0 |
| Reasoning quality bonus | +0.0 to +0.3 |

## Environment Details

- **Road**: 3 lanes, ~200 units long
- **Cars**: 5 total (1 agent + 4 scripted)
- **Max steps**: 100 per episode
- **Speed range**: 20–90 units
