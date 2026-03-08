---
title: APEX-Agents Environment Server
emoji: 📋
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# APEX-Agents Environment

A professional task evaluation environment for [OpenEnv](https://github.com/meta-pytorch/OpenEnv), wrapping the [Mercor APEX-Agents](https://huggingface.co/datasets/mercor/apex-agents) benchmark.

## Overview

480 professional tasks across three domains:
- **Law** — legal analysis, contract review, regulatory compliance
- **Investment Banking** — financial modeling, deal analysis, market assessment
- **Management Consulting** — strategy, operations, organizational design

Single-step episodes: the agent receives a task prompt, submits a response, and is graded by an LLM judge against rubric criteria.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set judge API key
export JUDGE_API_KEY="your-openai-api-key"

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

```python
from apex_agents_env import ApexAgentsEnv, ApexAction

with ApexAgentsEnv(base_url="http://localhost:8000") as env:
    result = env.reset(domain="law")
    print(result.observation.task_prompt)

    action = ApexAction(response="Based on the applicable statutes...")
    result = env.step(action)
    print(f"Score: {result.observation.criteria_passed}/{result.observation.criteria_total}")
    print(f"Reward: {result.reward}")
```

## Reward Modes

| Mode | Formula | Range |
|------|---------|-------|
| **Capped** | `passed / total` | [0, 1] |
| **Uncapped** | `(passed / total) * (1 + log2(1 + tokens/256))` | [0, ~6+] |

Uncapped mode: correctness is multiplicative — wrong answers always get 0 regardless of length. Logarithmic token scaling rewards thoroughness with diminishing returns.

## Reset Options

| Parameter | Description |
|-----------|-------------|
| `task_id` | Select a specific task by ID |
| `domain` | Filter by domain: `law`, `investment_banking`, `consulting` |
| `reward_mode` | `capped` (default) or `uncapped` |
| `judge_model` | LLM judge model (default: `gpt-4o-mini`) |
| `seed` | Random seed for reproducibility |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JUDGE_API_KEY` | API key for the judge LLM | `$OPENAI_API_KEY` |
| `JUDGE_MODEL` | Default judge model | `gpt-4o-mini` |
| `JUDGE_BASE_URL` | Custom API base URL | OpenAI default |
