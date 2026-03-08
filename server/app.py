"""
Overflow OpenENV — Continuous PPO Training + Live Incident Management Dashboard

Training loop runs in a background thread.
Every step:
  - Classifies the scene into an incident type
  - Grades the agent's action against that incident
  - Records the incident in a live feed

Dashboard:
  Left:  2D road canvas + live incident feed (step-by-step decisions)
  Right: Episode reward curve + incident response accuracy chart

GET  /         → HTML dashboard
GET  /api/state → JSON snapshot
GET  /api/stream → SSE (0.5s heartbeats + episode/incident events)
POST /api/mode  → switch reward mode {capped, uncapped}
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch

# ── Absolute-import fallback ──────────────────────────────────────────────────
try:
    from ..training.overflow_gym_env import OverflowGymEnv
    from ..training.curriculum import CurriculumManager
    from ..training.reward import compute_episode_bonus, IncidentType
    from ..training.ppo_trainer import RolloutBuffer
    from ..policies.ticket_attention_policy import TicketAttentionPolicy
    from ..policies.policy_spec import OBS_DIM
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.overflow_gym_env import OverflowGymEnv
    from training.curriculum import CurriculumManager
    from training.reward import compute_episode_bonus, IncidentType
    from training.ppo_trainer import RolloutBuffer
    from policies.ticket_attention_policy import TicketAttentionPolicy
    from policies.policy_spec import OBS_DIM

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Reward mode config ────────────────────────────────────────────────────────

REWARD_CAP      =  8.0    # capped mode: ceiling per step (matches max correct response)
TOKEN_SCALE     =  0.002  # uncapped: reward += tokens * TOKEN_SCALE
MAX_TOKEN_BONUS = 10.0    # uncapped mode max bonus per step

# ── Shared state ──────────────────────────────────────────────────────────────

@dataclass
class CarSnapshot:
    car_id: int
    x: float
    y: float
    lane: int
    speed: float

@dataclass
class IncidentEvent:
    step: int
    episode: int
    incident_type: str
    decision: str
    reward: float
    grade_desc: str

@dataclass
class TrainingState:
    # Road
    cars: List[CarSnapshot]      = field(default_factory=list)
    ego_x: float                 = 0.0
    ego_lane: int                = 2
    goal_x: float                = 180.0
    # Training metrics
    total_steps: int             = 0
    n_updates: int               = 0
    n_episodes: int              = 0
    episode_reward: float        = 0.0
    episode_steps: int           = 0
    mean_reward_100: float       = 0.0
    mean_ep_len: float           = 0.0
    stage: int                   = 1
    stage_name: str              = "Survival"
    reward_mode: str             = "capped"
    # Incident stats
    incident_feed: List[Dict]    = field(default_factory=list)   # last 100 events
    incident_counts: Dict        = field(default_factory=dict)   # {type: count}
    correct_responses: int       = 0
    total_responses: int         = 0
    # History for charts
    reward_history: List[float]  = field(default_factory=list)   # per-episode cumulative
    cumulative_reward: List[float] = field(default_factory=list) # running net total
    accuracy_history: List[float]= field(default_factory=list)   # per-episode correct%
    episode_history: List[Dict]  = field(default_factory=list)
    # PPO
    last_pg_loss: float          = 0.0
    last_vf_loss: float          = 0.0
    last_entropy: float          = 0.0
    running: bool                = True
    error: Optional[str]         = None


_state      = TrainingState()
_state_lock = threading.Lock()
_sse_queue: Deque[str] = deque(maxlen=100)

# Per-episode counters (reset each episode)
_ep_correct  = 0
_ep_total    = 0

# Which decisions count as "correct" for each incident type
_CORRECT_RESPONSES: Dict[str, set] = {
    IncidentType.CRASH_IMMINENT.value:   {"brake", "lane_change_left", "lane_change_right"},
    IncidentType.NEAR_MISS_AHEAD.value:  {"brake", "lane_change_left", "lane_change_right"},
    IncidentType.NEAR_MISS_SIDE.value:   {"brake", "maintain"},
    IncidentType.BLOCKED_AHEAD.value:    {"brake", "lane_change_left", "lane_change_right"},
    IncidentType.APPROACHING_GOAL.value: {"accelerate", "maintain"},
    IncidentType.CLEAR_ROAD.value:       {"accelerate", "maintain"},
}


def _is_correct(incident_type: str, decision: str) -> bool:
    return decision in _CORRECT_RESPONSES.get(incident_type, set())


def get_reward_mode() -> str:
    with _state_lock:
        return _state.reward_mode

def set_reward_mode(mode: str) -> None:
    with _state_lock:
        _state.reward_mode = mode

def apply_reward_mode(base_reward: float, token_count: int = 0) -> float:
    mode = get_reward_mode()
    if mode == "uncapped":
        bonus = min(token_count * TOKEN_SCALE, MAX_TOKEN_BONUS)
        return base_reward + bonus
    else:
        # Only cap positive rewards — keep penalties intact
        return min(base_reward, REWARD_CAP) if base_reward > 0 else base_reward

def _push_sse(data: dict) -> None:
    _sse_queue.append(json.dumps(data))


# ── Training thread ───────────────────────────────────────────────────────────

def _training_loop() -> None:
    global _ep_correct, _ep_total

    try:
        policy    = TicketAttentionPolicy(obs_dim=OBS_DIM)
        env       = OverflowGymEnv()
        curriculum = CurriculumManager()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

        GAMMA = 0.99; GAE_LAMBDA = 0.95; CLIP = 0.2
        ENT_COEF = 0.02; VF_COEF = 0.5; MAX_GRAD = 0.5
        N_STEPS = 512; BATCH_SIZE = 128; N_EPOCHS = 6

        buf = RolloutBuffer(N_STEPS, OBS_DIM, device)

        ep_rewards:    deque = deque(maxlen=100)
        ep_lengths:    deque = deque(maxlen=100)
        ep_accuracies: deque = deque(maxlen=100)

        obs, _ = env.reset()
        ep_reward = 0.0; ep_steps = 0
        total_steps = 0; n_updates = 0; n_episodes = 0
        _ep_correct = 0; _ep_total = 0
        t0 = time.time()

        while True:
            buf.reset()
            policy.eval()

            for _ in range(N_STEPS):
                curriculum.step(env._sim_time)

                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    act_mean, val = policy(obs_t.unsqueeze(0))
                act_mean = act_mean.squeeze(0)
                val      = val.squeeze(0)

                dist   = torch.distributions.Normal(act_mean, torch.ones_like(act_mean) * 0.3)
                action = dist.sample().clamp(-1, 1)
                logp   = dist.log_prob(action).sum()

                next_obs, base_reward, term, trunc, info = env.step(action.cpu().numpy())

                reward = apply_reward_mode(base_reward)

                buf.add(obs, action.cpu().numpy(), reward, float(val), float(logp), float(term or trunc))

                obs        = next_obs
                ep_reward += reward
                ep_steps  += 1
                total_steps += 1

                # ── Incident tracking ────────────────────────────────────
                inc_type = info.get("incident_type", "CLEAR_ROAD")
                decision = info.get("decision", "maintain")
                correct  = _is_correct(inc_type, decision)
                _ep_correct += int(correct)
                _ep_total   += 1

                evt = {
                    "step":          total_steps,
                    "episode":       n_episodes + 1,
                    "incident_type": inc_type,
                    "decision":      decision,
                    "reward":        round(reward, 2),
                    "correct":       correct,
                    "grade_desc":    info.get("grade_desc", ""),
                }

                # Pull car positions from env internals
                cars = []
                if hasattr(env._env, "_cars"):
                    for c in env._env._cars:
                        cars.append(CarSnapshot(
                            car_id=c.car_id, x=c.position,
                            y=(c.lane - 2) * 3.7, lane=c.lane, speed=c.speed,
                        ))
                goal_x = 180.0
                if hasattr(env._env, "_cars") and env._env._cars:
                    agent_car = next((c for c in env._env._cars if c.car_id == 0), None)
                    if agent_car:
                        goal_x = agent_car.goal_position

                with _state_lock:
                    _state.total_steps    = total_steps
                    _state.episode_reward = round(ep_reward, 2)
                    _state.episode_steps  = ep_steps
                    _state.goal_x         = goal_x
                    _state.correct_responses += int(correct)
                    _state.total_responses   += 1
                    _state.incident_counts[inc_type] = _state.incident_counts.get(inc_type, 0) + 1
                    _state.incident_feed.append(evt)
                    if len(_state.incident_feed) > 100:
                        _state.incident_feed = _state.incident_feed[-100:]
                    if cars:
                        _state.cars = cars
                        ego = next((c for c in cars if c.car_id == 0), None)
                        if ego:
                            _state.ego_x = ego.x
                            _state.ego_lane = ego.lane

                # Push incident event to SSE (only threatening incidents or every 10 steps)
                if inc_type not in (IncidentType.CLEAR_ROAD.value,) or total_steps % 10 == 0:
                    _push_sse({"type": "incident", "data": evt})

                # ── Episode end ──────────────────────────────────────────
                if term or trunc:
                    bonus = compute_episode_bonus(
                        total_steps=ep_steps,
                        survived=not info.get("collision", False),
                    )
                    ep_reward += bonus
                    n_episodes += 1
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(ep_steps)

                    acc = (_ep_correct / _ep_total * 100) if _ep_total > 0 else 0.0
                    ep_accuracies.append(acc)
                    advanced = curriculum.record_episode_reward(ep_reward)

                    outcome = (
                        "crash"   if info.get("collision")    else
                        "goal"    if info.get("goal_reached") else "timeout"
                    )
                    ep_rec = {
                        "episode":     n_episodes,
                        "steps":       ep_steps,
                        "reward":      round(ep_reward, 2),
                        "outcome":     outcome,
                        "stage":       curriculum.current_stage,
                        "accuracy":    round(acc, 1),
                        "reward_mode": get_reward_mode(),
                    }

                    with _state_lock:
                        _state.n_episodes       = n_episodes
                        _state.stage            = curriculum.current_stage
                        _state.stage_name       = curriculum.config.name
                        _state.mean_reward_100  = round(float(np.mean(ep_rewards)), 2)
                        _state.mean_ep_len      = round(float(np.mean(ep_lengths)), 1)
                        _state.reward_history.append(round(ep_reward, 2))
                        prev_cum = _state.cumulative_reward[-1] if _state.cumulative_reward else 0.0
                        _state.cumulative_reward.append(round(prev_cum + ep_reward, 2))
                        _state.accuracy_history.append(round(acc, 1))
                        _state.episode_history.append(ep_rec)

                    _push_sse({"type": "episode", "data": ep_rec})

                    obs, _ = env.reset()
                    ep_reward = 0.0; ep_steps = 0
                    _ep_correct = 0; _ep_total = 0

            # ── PPO update ────────────────────────────────────────────────
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _, last_val = policy(obs_t.unsqueeze(0))
            buf.compute_returns(float(last_val), GAMMA, GAE_LAMBDA)

            policy.train()
            all_obs  = buf.obs;  all_acts = buf.acts;  old_logp = buf.logp
            adv      = buf.ret - buf.val
            adv      = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret      = buf.ret;  old_val  = buf.val
            indices  = torch.randperm(N_STEPS, device=device)
            pg_ls, vf_ls, ents = [], [], []

            for _ in range(N_EPOCHS):
                for start in range(0, N_STEPS, BATCH_SIZE):
                    idx     = indices[start: start + BATCH_SIZE]
                    am, val = policy(all_obs[idx])
                    val     = val.squeeze(-1)
                    dist    = torch.distributions.Normal(am, torch.ones_like(am) * 0.3)
                    logp    = dist.log_prob(all_acts[idx]).sum(dim=-1)
                    ent     = dist.entropy().sum(dim=-1).mean()
                    ratio   = torch.exp(logp - old_logp[idx])
                    pg_loss = torch.max(-adv[idx]*ratio, -adv[idx]*ratio.clamp(1-CLIP, 1+CLIP)).mean()
                    vc      = old_val[idx] + (val - old_val[idx]).clamp(-CLIP, CLIP)
                    vf_loss = 0.5*torch.max((val-ret[idx])**2, (vc-ret[idx])**2).mean()
                    loss    = pg_loss + VF_COEF*vf_loss - ENT_COEF*ent
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
                    optimizer.step()
                    pg_ls.append(float(pg_loss)); vf_ls.append(float(vf_loss)); ents.append(float(ent))

            n_updates += 1
            with _state_lock:
                _state.n_updates    = n_updates
                _state.last_pg_loss = round(float(np.mean(pg_ls)), 5)
                _state.last_vf_loss = round(float(np.mean(vf_ls)), 5)
                _state.last_entropy = round(float(np.mean(ents)), 5)
                _state.steps_per_sec = round(total_steps / max(time.time() - t0, 1.0), 1)

            _push_sse({"type": "update", "data": {
                "n_updates": n_updates, "total_steps": total_steps,
                "mean_reward": round(float(np.mean(ep_rewards)) if ep_rewards else 0.0, 2),
                "stage": curriculum.current_stage,
                "pg_loss": round(float(np.mean(pg_ls)), 5),
                "vf_loss": round(float(np.mean(vf_ls)), 5),
                "entropy": round(float(np.mean(ents)), 5),
            }})

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        with _state_lock:
            _state.running = False
            _state.error   = f"{exc}\n\n{tb}"
        print(f"[Training] FATAL: {exc}\n{tb}", flush=True)


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Overflow OpenENV")

_training_thread: Optional[threading.Thread] = None

@app.on_event("startup")
def _start_training():
    global _training_thread
    _training_thread = threading.Thread(target=_training_loop, daemon=True)
    _training_thread.start()

@app.get("/health")
def health():
    return {"status": "ok"}

class ModeRequest(BaseModel):
    mode: str

@app.post("/api/mode")
def set_mode(req: ModeRequest):
    if req.mode not in ("capped", "uncapped"):
        return {"error": "mode must be capped or uncapped"}
    set_reward_mode(req.mode)
    return {"mode": req.mode}

@app.post("/api/restart")
def restart_training():
    global _state, _training_thread, _ep_correct, _ep_total
    with _state_lock:
        mode = _state.reward_mode  # preserve current mode selection
        _state = TrainingState()
        _state.reward_mode = mode
    _ep_correct = 0
    _ep_total = 0
    _sse_queue.clear()
    # Start a fresh training thread (old one is daemon and will die)
    _training_thread = threading.Thread(target=_training_loop, daemon=True)
    _training_thread.start()
    return {"status": "restarted"}

@app.get("/api/state")
def get_state():
    with _state_lock:
        s = _state
        acc = round(s.correct_responses / s.total_responses * 100, 1) if s.total_responses > 0 else 0.0
        return {
            "total_steps":       s.total_steps,
            "n_updates":         s.n_updates,
            "n_episodes":        s.n_episodes,
            "episode_reward":    s.episode_reward,
            "episode_steps":     s.episode_steps,
            "mean_reward":       s.mean_reward_100,
            "mean_ep_len":       s.mean_ep_len,
            "stage":             s.stage,
            "stage_name":        s.stage_name,
            "reward_mode":       s.reward_mode,
            "steps_per_sec":     s.steps_per_sec if hasattr(s, "steps_per_sec") else 0,
            "pg_loss":           s.last_pg_loss,
            "vf_loss":           s.last_vf_loss,
            "entropy":           s.last_entropy,
            "reward_history":    s.reward_history,
            "cumulative_reward": s.cumulative_reward,
            "accuracy_history":  s.accuracy_history,
            "episode_history":   s.episode_history[-50:],
            "incident_feed":     s.incident_feed[-30:],
            "incident_counts":   s.incident_counts,
            "response_accuracy": acc,
            "cars":              [asdict(c) for c in s.cars],
            "ego_x":             s.ego_x,
            "ego_lane":          s.ego_lane,
            "goal_x":            s.goal_x,
            "running":           s.running,
            "error":             s.error,
        }

@app.get("/api/stream")
async def sse_stream():
    async def gen():
        last_idx = len(_sse_queue)
        while True:
            current = list(_sse_queue)
            for msg in current[last_idx:]:
                yield f"data: {msg}\n\n"
            last_idx = len(current)

            with _state_lock:
                s = _state
                acc = round(s.correct_responses / s.total_responses * 100, 1) if s.total_responses > 0 else 0.0
                snap = {
                    "type": "tick",
                    "data": {
                        "total_steps":       s.total_steps,
                        "episode_reward":    s.episode_reward,
                        "episode_steps":     s.episode_steps,
                        "stage":             s.stage,
                        "stage_name":        s.stage_name,
                        "reward_mode":       s.reward_mode,
                        "response_accuracy": acc,
                        "cars":              [asdict(c) for c in s.cars],
                        "ego_x":             s.ego_x,
                        "goal_x":            s.goal_x,
                    }
                }
            yield f"data: {json.dumps(snap)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── HTML Dashboard ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Overflow OpenENV — Incident Management Training</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #08080f; color: #ddd; font-family: 'Courier New', monospace; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

header { background: #101018; border-bottom: 1px solid #252535; padding: 8px 18px; display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
header h1 { font-size: 13px; color: #7eb8ff; letter-spacing: 3px; }
.badge { padding: 2px 9px; border-radius: 10px; font-size: 10px; font-weight: bold; }
.badge-ok  { background:#0e2a0e; color:#4caf50; border:1px solid #4caf50; }
.badge-err { background:#2a0e0e; color:#f44336; border:1px solid #f44336; }
.mode-row { margin-left:auto; display:flex; gap:6px; align-items:center; }
.mbtn { padding:3px 12px; border-radius:7px; border:1px solid #333; background:#151522; color:#888; cursor:pointer; font-size:11px; }
.mbtn.on { background:#0e2a40; color:#7eb8ff; border-color:#7eb8ff; }

main { flex:1; display:flex; min-height:0; }

/* LEFT */
.left { flex: 0 0 54%; display:flex; flex-direction:column; border-right:1px solid #252535; min-height:0; }
.road-wrap { flex:0 0 170px; padding:10px 12px 0; }
.road-wrap canvas { width:100%; height:150px; display:block; background:#111118; border:1px solid #252535; border-radius:6px; }

.metrics-row { flex:0 0 auto; display:grid; grid-template-columns:repeat(4,1fr); gap:6px; padding:8px 12px; }
.mc { background:#101018; border:1px solid #252535; border-radius:6px; padding:8px; }
.mc-label { font-size:9px; color:#556; text-transform:uppercase; letter-spacing:1px; }
.mc-val { font-size:16px; color:#7eb8ff; font-weight:bold; margin-top:1px; }
.mc-sub { font-size:9px; color:#445; margin-top:1px; }

.stage-row { flex:0 0 auto; padding:0 12px 6px; }
.stage-inner { background:#101018; border:1px solid #252535; border-radius:6px; padding:8px 10px; }
.stage-title { font-size:9px; color:#556; letter-spacing:2px; text-transform:uppercase; }
.stage-name { font-size:12px; color:#7eb8ff; margin-top:2px; }
.pips { display:flex; gap:5px; margin-top:6px; }
.pip { flex:1; height:5px; border-radius:3px; background:#252535; transition:background .4s; }
.pip.done { background:#4caf50; }
.pip.active { background:#7eb8ff; }

/* Incident feed */
.feed-wrap { flex:1; display:flex; flex-direction:column; min-height:0; padding:0 12px 10px; }
.feed-title { font-size:9px; color:#556; letter-spacing:2px; text-transform:uppercase; margin-bottom:5px; display:flex; justify-content:space-between; }
.feed-title span { color:#7eb8ff; }
.feed { flex:1; overflow-y:auto; background:#101018; border:1px solid #252535; border-radius:6px; font-size:10px; }
.feed-row { display:grid; grid-template-columns:50px 1fr 90px 60px 50px; gap:0; padding:4px 8px; border-bottom:1px solid #181820; align-items:center; }
.feed-row:hover { background:#14141e; }
.feed-hdr { background:#131320; color:#556; font-size:9px; letter-spacing:1px; position:sticky; top:0; }
.inc-type { font-size:9px; font-weight:bold; }
.inc-crash    { color:#f44336; }
.inc-near     { color:#ff9800; }
.inc-blocked  { color:#ffeb3b; }
.inc-approach { color:#4caf50; }
.inc-clear    { color:#556; }
.dec  { color:#aaa; }
.rw-pos { color:#4caf50; }
.rw-neg { color:#f44336; }
.correct-y { color:#4caf50; }
.correct-n { color:#f44336; }

/* RIGHT */
.right { flex:1; display:flex; flex-direction:column; min-height:0; padding:10px 12px; gap:8px; }
.chart-wrap { flex:1; display:flex; flex-direction:column; min-height:0; }
.chart-title { font-size:9px; color:#556; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px; }
canvas.chart { flex:1; min-height:0; display:block; background:#101018; border:1px solid #252535; border-radius:6px; }
.ep-wrap { flex:0 0 180px; display:flex; flex-direction:column; min-height:0; }
.ep-table { flex:1; overflow-y:auto; background:#101018; border:1px solid #252535; border-radius:6px; }
table { width:100%; border-collapse:collapse; font-size:10px; }
th { padding:5px 8px; color:#556; text-align:left; border-bottom:1px solid #252535; position:sticky; top:0; background:#131320; font-size:9px; }
td { padding:4px 8px; border-bottom:1px solid #13131a; }
.c-crash { color:#f44336; }
.c-goal  { color:#4caf50; }
.c-tout  { color:#ff9800; }

.ppo-row { display:flex; gap:6px; flex-wrap:wrap; flex-shrink:0; }
.pstat { background:#101018; border:1px solid #252535; border-radius:5px; padding:4px 10px; font-size:10px; }
.pstat span { color:#7eb8ff; }

.rbtn { padding:3px 12px; border-radius:7px; border:1px solid #f44336; background:#1a0e0e; color:#f44336; cursor:pointer; font-size:11px; font-weight:bold; letter-spacing:1px; transition:all .2s; }
.rbtn:hover { background:#2a1515; border-color:#ff6659; color:#ff6659; }

.info-note { background:#101018; border:1px solid #252535; border-radius:6px; padding:6px 10px; font-size:9px; color:#778; line-height:1.5; flex-shrink:0; }
.info-note strong { color:#ffeb3b; font-size:9px; }
</style>
</head>
<body>
<header>
  <h1>OVERFLOW OPENENV — INCIDENT MANAGEMENT</h1>
  <span id="sbadge" class="badge badge-ok">TRAINING</span>
  <div class="mode-row">
    <span style="font-size:10px;color:#556">REWARD:</span>
    <button class="mbtn on" id="bcap" onclick="setMode('capped')">CAPPED</button>
    <button class="mbtn" id="bunc" onclick="setMode('uncapped')">UNCAPPED (LLM tokens)</button>
    <span style="width:1px;height:18px;background:#252535;margin:0 4px"></span>
    <button class="rbtn" onclick="restartTraining()">RESTART</button>
  </div>
</header>

<main>
  <!-- LEFT -->
  <div class="left">
    <div class="road-wrap">
      <canvas id="road" width="800" height="150"></canvas>
    </div>

    <div class="metrics-row">
      <div class="mc"><div class="mc-label">Steps</div><div class="mc-val" id="m-st">0</div><div class="mc-sub" id="m-ep">ep 0</div></div>
      <div class="mc"><div class="mc-label">Mean Reward</div><div class="mc-val" id="m-mr">0.00</div><div class="mc-sub" id="m-er">ep: 0.00</div></div>
      <div class="mc"><div class="mc-label">Response Acc.</div><div class="mc-val" id="m-acc">0%</div><div class="mc-sub">correct actions</div></div>
      <div class="mc"><div class="mc-label">PPO Updates</div><div class="mc-val" id="m-up">0</div><div class="mc-sub" id="m-st2">stage 1</div></div>
    </div>

    <div class="stage-row">
      <div class="stage-inner">
        <div class="stage-title">CURRICULUM — <span id="sname">Survival</span></div>
        <div class="pips"><div class="pip active" id="p1"></div><div class="pip" id="p2"></div><div class="pip" id="p3"></div><div class="pip" id="p4"></div></div>
      </div>
    </div>

    <div class="feed-wrap">
      <div class="feed-title">INCIDENT FEED — LIVE DECISIONS <span id="acc-badge">ACC 0%</span></div>
      <div class="feed" id="feed">
        <div class="feed-row feed-hdr">
          <div>STEP</div><div>INCIDENT</div><div>DECISION</div><div>REWARD</div><div>OK?</div>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT -->
  <div class="right">
    <div class="chart-wrap" style="flex:0 0 42%">
      <div class="chart-title" id="rw-title">EPISODE REWARD — CAPPED | mean: 0.00 | net: 0.00</div>
      <canvas class="chart" id="rwChart"></canvas>
    </div>
    <div class="info-note">
      <strong>Why does net reward start negative?</strong>
      Early in training the agent has a random policy — it crashes frequently and earns large negative penalties
      (collision = heavy punishment). The cumulative net drops as these early failures stack up. As PPO learns to
      avoid crashes and respond correctly to incidents, per-episode rewards turn positive and the net climbs back.
      The crossover point where net goes positive is when the agent's learned gains have fully offset its early
      exploration cost — a sign that training is working.
    </div>
    <div class="chart-wrap" style="flex:0 0 28%">
      <div class="chart-title">RESPONSE ACCURACY % (per episode)</div>
      <canvas class="chart" id="accChart"></canvas>
    </div>
    <div class="ep-wrap">
      <div class="chart-title">EPISODE LOG</div>
      <div class="ep-table">
        <table><thead><tr><th>#</th><th>Steps</th><th>Reward</th><th>Outcome</th><th>Acc%</th><th>Stage</th></tr></thead>
        <tbody id="eptbl"></tbody></table>
      </div>
    </div>
    <div class="ppo-row">
      <div class="pstat">PG: <span id="pg">—</span></div>
      <div class="pstat">VF: <span id="vf">—</span></div>
      <div class="pstat">Ent: <span id="ent">—</span></div>
      <div class="pstat">Mode: <span id="mode-lbl">capped</span></div>
    </div>
  </div>
</main>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let S = {
  cars:[], reward_history:[], cumulative_reward:[], accuracy_history:[], episode_history:[],
  incident_feed:[], incident_counts:{}, stage:1, reward_mode:'capped',
  response_accuracy:0, total_steps:0, n_episodes:0, n_updates:0,
  ego_x:0, goal_x:180, episode_reward:0, episode_steps:0,
  mean_reward:0, pg_loss:0, vf_loss:0, entropy:0, stage_name:'Survival',
};

// ── Road canvas ──────────────────────────────────────────────────────────
const roadC = document.getElementById('road');
const roadX = roadC.getContext('2d');
const N_LANES = 3, LANE_H = 40, ROAD_TOP = (150 - N_LANES*LANE_H)/2;

function laneY(lane) { return ROAD_TOP + (lane-1)*LANE_H + LANE_H/2; }
function carPx(x) {
  const w = roadC.offsetWidth || 800;
  return w*0.3 + (x - S.ego_x)*0.8;
}

function drawRoad() {
  const w = roadC.offsetWidth || 800, h = 150;
  roadC.width = w;
  const ctx = roadX;
  ctx.clearRect(0,0,w,h);
  // Road bg
  ctx.fillStyle='#15151f';
  ctx.fillRect(0, ROAD_TOP, w, N_LANES*LANE_H);
  // Lane lines
  ctx.setLineDash([18,12]); ctx.strokeStyle='#2a2a40'; ctx.lineWidth=1;
  for(let i=1;i<N_LANES;i++){
    const y=ROAD_TOP+i*LANE_H;
    ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();
  }
  ctx.setLineDash([]);
  // Road edges
  ctx.strokeStyle='#4a4a6a'; ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(0,ROAD_TOP);ctx.lineTo(w,ROAD_TOP);ctx.stroke();
  ctx.beginPath();ctx.moveTo(0,ROAD_TOP+N_LANES*LANE_H);ctx.lineTo(w,ROAD_TOP+N_LANES*LANE_H);ctx.stroke();
  // Goal marker
  const gx = carPx(S.goal_x);
  if(gx>0 && gx<w){
    ctx.strokeStyle='rgba(76,175,80,0.6)'; ctx.lineWidth=2; ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(gx,ROAD_TOP);ctx.lineTo(gx,ROAD_TOP+N_LANES*LANE_H);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='rgba(76,175,80,0.8)'; ctx.font='bold 9px Courier New'; ctx.textAlign='center';
    ctx.fillText('GOAL',gx,ROAD_TOP-3);
  }
  // Cars
  for(const car of (S.cars||[])){
    const cx=carPx(car.x), cy=laneY(car.lane);
    if(cx<-60||cx>w+60) continue;
    const isEgo=car.car_id===0;
    ctx.save(); ctx.translate(cx,cy);
    ctx.fillStyle=isEgo?'#1a3a6a':'#2a1e10';
    ctx.strokeStyle=isEgo?'#7eb8ff':'#ff9800';
    ctx.lineWidth=isEgo?2:1;
    ctx.beginPath(); ctx.roundRect(-18,-10,36,20,4); ctx.fill(); ctx.stroke();
    ctx.fillStyle=isEgo?'#7eb8ff':'#ff9800';
    ctx.font=isEgo?'bold 8px Courier New':'7px Courier New';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(isEgo?'EGO':'C'+car.car_id,0,0);
    // Speed indicator
    ctx.fillStyle='rgba(255,255,255,0.3)'; ctx.font='6px Courier New'; ctx.textBaseline='top';
    ctx.fillText(Math.round(car.speed),0,11);
    ctx.restore();
  }
}

// ── Chart drawing ────────────────────────────────────────────────────────
// Shows ALL data from t=0 to current t. X-axis scales to fit all episodes.
// Draws: raw (faint), MA (bright), global mean (dashed yellow).
function drawLineChart(canvasId, data, color, label, yMin, yMax, showZero) {
  const canvas = document.getElementById(canvasId);
  const w = canvas.offsetWidth||400, h = canvas.offsetHeight||160;
  canvas.width=w; canvas.height=h;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,w,h);
  if(!data||data.length<2){
    ctx.fillStyle='#445'; ctx.font='11px Courier New'; ctx.textAlign='center';
    ctx.fillText('Waiting for episodes...', w/2, h/2);
    return;
  }
  const pad={t:10,r:10,b:24,l:52};
  const pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  const mn = yMin!==undefined?yMin:Math.min(...data);
  const mx = yMax!==undefined?yMax:Math.max(...data);
  const rng = mx-mn||1;
  const n   = data.length;
  const xOf = i => pad.l + i*(pw/(n-1||1));
  const yOf = v => pad.t + (mx-v)/rng*ph;

  // Grid lines + Y labels
  ctx.strokeStyle='#1a1a28'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad.t+ph*(i/4);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+pw,y);ctx.stroke();
    ctx.fillStyle='#445'; ctx.font='8px Courier New'; ctx.textAlign='right';
    ctx.fillText((mx-rng*(i/4)).toFixed(1), pad.l-3, y+3);
  }

  // Zero line
  if(showZero && mn<0 && mx>0){
    const zy=yOf(0);
    ctx.strokeStyle='#3a3a50'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(pad.l,zy);ctx.lineTo(pad.l+pw,zy);ctx.stroke();
    ctx.setLineDash([]);
  }

  // MA window: adaptive
  const MA = Math.max(5, Math.min(30, Math.floor(n/10)));
  const ma = data.map((_,i)=>{
    const sl=data.slice(Math.max(0,i-MA+1),i+1);
    return sl.reduce((a,b)=>a+b,0)/sl.length;
  });

  // Global mean (horizontal dashed line)
  const globalMean = data.reduce((a,b)=>a+b,0)/n;
  const gy = yOf(globalMean);
  ctx.strokeStyle='rgba(255,235,59,0.6)'; ctx.lineWidth=1; ctx.setLineDash([6,4]);
  ctx.beginPath();ctx.moveTo(pad.l,gy);ctx.lineTo(pad.l+pw,gy);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='rgba(255,235,59,0.8)'; ctx.font='bold 9px Courier New'; ctx.textAlign='left';
  ctx.fillText('\u03bc='+globalMean.toFixed(2), pad.l+4, gy-5);

  // Raw line (faint)
  ctx.strokeStyle=color+'33'; ctx.lineWidth=1; ctx.beginPath();
  data.forEach((v,i)=>{ i?ctx.lineTo(xOf(i),yOf(v)):ctx.moveTo(xOf(i),yOf(v)); });
  ctx.stroke();

  // Smoothed MA line
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  ma.forEach((v,i)=>{ i?ctx.lineTo(xOf(i),yOf(v)):ctx.moveTo(xOf(i),yOf(v)); });
  ctx.stroke();

  // X-axis: t=0 on left, current episode on right
  ctx.fillStyle='#445'; ctx.font='8px Courier New';
  ctx.textAlign='left';  ctx.fillText('t=0', pad.l, h-4);
  ctx.textAlign='right'; ctx.fillText('t='+n, pad.l+pw, h-4);
  ctx.textAlign='center';ctx.fillText(label, pad.l+pw/2, h-4);
}

// Reward chart with dual Y-axes: per-episode reward (left) + cumulative net (right)
function drawRewardChart() {
  const canvas = document.getElementById('rwChart');
  const w = canvas.offsetWidth||400, h = canvas.offsetHeight||160;
  canvas.width=w; canvas.height=h;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,w,h);
  const data = S.reward_history||[];
  const cumul = S.cumulative_reward||[];
  if(!data||data.length<2){
    ctx.fillStyle='#445'; ctx.font='11px Courier New'; ctx.textAlign='center';
    ctx.fillText('Waiting for episodes...', w/2, h/2);
    return;
  }
  const pad={t:10,r:52,b:24,l:52};
  const pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  const n = data.length;
  const xOf = i => pad.l + i*(pw/(n-1||1));

  // Left Y: per-episode reward
  const mn1 = Math.min(...data);
  const mx1 = Math.max(...data);
  const rng1 = mx1-mn1||1;
  const yOf1 = v => pad.t + (mx1-v)/rng1*ph;

  // Right Y: cumulative net reward
  const mn2 = cumul.length? Math.min(...cumul) : 0;
  const mx2 = cumul.length? Math.max(...cumul) : 1;
  const rng2 = mx2-mn2||1;
  const yOf2 = v => pad.t + (mx2-v)/rng2*ph;

  // Grid lines + left Y labels
  ctx.strokeStyle='#1a1a28'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad.t+ph*(i/4);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+pw,y);ctx.stroke();
    ctx.fillStyle='#556'; ctx.font='8px Courier New'; ctx.textAlign='right';
    ctx.fillText((mx1-rng1*(i/4)).toFixed(1), pad.l-3, y+3);
  }
  // Right Y labels (cumulative)
  for(let i=0;i<=4;i++){
    const y=pad.t+ph*(i/4);
    ctx.fillStyle='#ff985580'; ctx.font='8px Courier New'; ctx.textAlign='left';
    ctx.fillText((mx2-rng2*(i/4)).toFixed(0), pad.l+pw+3, y+3);
  }

  // Zero line
  if(mn1<0 && mx1>0){
    const zy=yOf1(0);
    ctx.strokeStyle='#3a3a50'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(pad.l,zy);ctx.lineTo(pad.l+pw,zy);ctx.stroke();
    ctx.setLineDash([]);
  }

  // Global mean (horizontal dashed line)
  const globalMean = data.reduce((a,b)=>a+b,0)/n;
  const gy = yOf1(globalMean);
  ctx.strokeStyle='rgba(255,235,59,0.6)'; ctx.lineWidth=1; ctx.setLineDash([6,4]);
  ctx.beginPath();ctx.moveTo(pad.l,gy);ctx.lineTo(pad.l+pw,gy);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='rgba(255,235,59,0.9)'; ctx.font='bold 9px Courier New'; ctx.textAlign='left';
  ctx.fillText('\u03bc='+globalMean.toFixed(2), pad.l+4, gy-5);

  // MA
  const MA = Math.max(5, Math.min(30, Math.floor(n/10)));
  const ma = data.map((_,i)=>{
    const sl=data.slice(Math.max(0,i-MA+1),i+1);
    return sl.reduce((a,b)=>a+b,0)/sl.length;
  });

  // Raw per-episode line (faint blue)
  ctx.strokeStyle='#7eb8ff33'; ctx.lineWidth=1; ctx.beginPath();
  data.forEach((v,i)=>{ i?ctx.lineTo(xOf(i),yOf1(v)):ctx.moveTo(xOf(i),yOf1(v)); });
  ctx.stroke();

  // Smoothed MA per-episode (bright blue)
  ctx.strokeStyle='#7eb8ff'; ctx.lineWidth=2; ctx.beginPath();
  ma.forEach((v,i)=>{ i?ctx.lineTo(xOf(i),yOf1(v)):ctx.moveTo(xOf(i),yOf1(v)); });
  ctx.stroke();

  // Cumulative net reward (orange, right axis)
  if(cumul.length>=2){
    ctx.strokeStyle='#ff9855'; ctx.lineWidth=2; ctx.beginPath();
    cumul.forEach((v,i)=>{ i?ctx.lineTo(xOf(i),yOf2(v)):ctx.moveTo(xOf(i),yOf2(v)); });
    ctx.stroke();
  }

  // X-axis
  ctx.fillStyle='#445'; ctx.font='8px Courier New';
  ctx.textAlign='left';  ctx.fillText('t=0', pad.l, h-4);
  ctx.textAlign='right'; ctx.fillText('t='+n, pad.l+pw, h-4);

  // Legend
  ctx.font='8px Courier New'; ctx.textAlign='center';
  ctx.fillStyle='#7eb8ff'; ctx.fillText('\u25CF per-ep', pad.l+pw*0.3, h-4);
  ctx.fillStyle='#ff9855'; ctx.fillText('\u25CF net cumul.', pad.l+pw*0.7, h-4);

  // Update title
  const net = cumul.length? cumul[cumul.length-1] : 0;
  const modeLabel = S.reward_mode.toUpperCase();
  document.getElementById('rw-title').textContent =
    'EPISODE REWARD \u2014 '+modeLabel+' | \u03bc: '+globalMean.toFixed(2)+' | net: '+net.toFixed(2);
}

// ── Incident feed ────────────────────────────────────────────────────────
const INC_COLORS = {
  'CRASH_IMMINENT':  'inc-crash',
  'NEAR_MISS_AHEAD': 'inc-near',
  'NEAR_MISS_SIDE':  'inc-near',
  'BLOCKED_AHEAD':   'inc-blocked',
  'APPROACHING_GOAL':'inc-approach',
  'CLEAR_ROAD':      'inc-clear',
};
const INC_SHORT = {
  'CRASH_IMMINENT':'CRASH!','NEAR_MISS_AHEAD':'NM-AHEAD','NEAR_MISS_SIDE':'NM-SIDE',
  'BLOCKED_AHEAD':'BLOCKED','APPROACHING_GOAL':'GOAL-NEAR','CLEAR_ROAD':'CLEAR',
};

function renderFeed(feed) {
  const el = document.getElementById('feed');
  // Keep header row, add/update feed rows
  const header = el.querySelector('.feed-hdr');
  el.innerHTML = '';
  if(header) el.appendChild(header);
  else {
    const h = document.createElement('div');
    h.className='feed-row feed-hdr';
    h.innerHTML='<div>STEP</div><div>INCIDENT</div><div>DECISION</div><div>REWARD</div><div>OK?</div>';
    el.appendChild(h);
  }
  const items = [...feed].reverse();
  for(const e of items){
    const row = document.createElement('div');
    row.className='feed-row';
    const tc = INC_COLORS[e.incident_type]||'inc-clear';
    const ts = INC_SHORT[e.incident_type]||e.incident_type;
    const rwc = e.reward>=0?'rw-pos':'rw-neg';
    const okc = e.correct?'correct-y':'correct-n';
    const okl = e.correct?'YES':'NO';
    row.innerHTML = `
      <div style="color:#556">${e.step}</div>
      <div class="inc-type ${tc}">${ts}</div>
      <div class="dec">${e.decision}</div>
      <div class="${rwc}">${e.reward>0?'+':''}${e.reward}</div>
      <div class="${okc}">${okl}</div>`;
    el.appendChild(row);
  }
}

// ── Episode table ────────────────────────────────────────────────────────
function renderEpTable(hist) {
  const tb = document.getElementById('eptbl');
  const rows = [...hist].reverse().slice(0,40).map(ep=>{
    const oc = ep.outcome==='crash'?'c-crash':ep.outcome==='goal'?'c-goal':'c-tout';
    const rw = ep.reward>=0?`<span class="c-goal">+${ep.reward}</span>`:`<span class="c-crash">${ep.reward}</span>`;
    return `<tr><td>${ep.episode}</td><td>${ep.steps}</td><td>${rw}</td>
      <td class="${oc}">${ep.outcome.toUpperCase()}</td>
      <td>${ep.accuracy||0}%</td><td>${ep.stage}</td></tr>`;
  }).join('');
  tb.innerHTML = rows;
}

// ── UI update ─────────────────────────────────────────────────────────────
function updateUI() {
  document.getElementById('m-st').textContent = S.total_steps.toLocaleString();
  document.getElementById('m-ep').textContent = `ep ${S.n_episodes}`;
  document.getElementById('m-mr').textContent = S.mean_reward.toFixed(2);
  document.getElementById('m-er').textContent = `ep: ${S.episode_reward.toFixed(2)}`;
  document.getElementById('m-acc').textContent = `${S.response_accuracy}%`;
  document.getElementById('m-up').textContent = S.n_updates;
  document.getElementById('m-st2').textContent = `stage ${S.stage}`;
  document.getElementById('sname').textContent = S.stage_name||'Survival';
  document.getElementById('pg').textContent = S.pg_loss||'—';
  document.getElementById('vf').textContent = S.vf_loss||'—';
  document.getElementById('ent').textContent = S.entropy||'—';
  document.getElementById('mode-lbl').textContent = S.reward_mode;
  document.getElementById('acc-badge').textContent = `ACC ${S.response_accuracy}%`;

  // Stage pips
  const st=S.stage||1;
  for(let i=1;i<=4;i++){
    const p=document.getElementById('p'+i);
    p.className='pip'+(i<st?' done':i===st?' active':'');
  }
  // Mode buttons
  document.getElementById('bcap').className='mbtn'+(S.reward_mode==='capped'?' on':'');
  document.getElementById('bunc').className='mbtn'+(S.reward_mode==='uncapped'?' on':'');

  if(S.error){
    document.getElementById('sbadge').textContent='ERROR';
    document.getElementById('sbadge').className='badge badge-err';
  }
}

// ── Render all ────────────────────────────────────────────────────────────
function renderAll() {
  drawRoad();
  drawRewardChart();
  drawLineChart('accChart', S.accuracy_history, '#4caf50', 'Episodes', 0, 100, false);
  renderFeed(S.incident_feed||[]);
  renderEpTable(S.episode_history||[]);
  updateUI();
}

// ── Mode switch ────────────────────────────────────────────────────────────
function setMode(mode) {
  fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})})
    .then(r=>r.json()).then(d=>{S.reward_mode=d.mode;updateUI();});
}

// ── Restart training ──────────────────────────────────────────────────────
function restartTraining() {
  if(!confirm('Restart training from scratch? All progress will be reset.')) return;
  fetch('/api/restart',{method:'POST'}).then(r=>r.json()).then(()=>{
    S.reward_history=[]; S.cumulative_reward=[]; S.accuracy_history=[];
    S.episode_history=[]; S.incident_feed=[]; S.incident_counts={};
    S.total_steps=0; S.n_episodes=0; S.n_updates=0;
    S.episode_reward=0; S.episode_steps=0; S.mean_reward=0;
    S.response_accuracy=0;
    renderAll();
  });
}

// ── SSE for fast events ───────────────────────────────────────────────────
const evtSrc = new EventSource('/api/stream');
evtSrc.onmessage = (e) => {
  try {
    const msg = JSON.parse(e.data);
    if(msg.type==='incident'){
      if(!S.incident_feed) S.incident_feed=[];
      S.incident_feed.push(msg.data);
      if(S.incident_feed.length>100) S.incident_feed=S.incident_feed.slice(-100);
      renderFeed(S.incident_feed);
    } else if(msg.type==='episode'){
      if(!S.episode_history) S.episode_history=[];
      S.episode_history.push(msg.data);
      if(!S.reward_history) S.reward_history=[];
      S.reward_history.push(msg.data.reward);
      if(!S.cumulative_reward) S.cumulative_reward=[];
      const prevNet = S.cumulative_reward.length? S.cumulative_reward[S.cumulative_reward.length-1] : 0;
      S.cumulative_reward.push(prevNet + msg.data.reward);
      if(!S.accuracy_history) S.accuracy_history=[];
      S.accuracy_history.push(msg.data.accuracy||0);
      renderEpTable(S.episode_history);
      drawRewardChart();
      drawLineChart('accChart', S.accuracy_history, '#4caf50', 'Episodes', 0, 100, false);
    } else if(msg.type==='tick'){
      Object.assign(S, msg.data);
      drawRoad(); updateUI();
    } else if(msg.type==='update'){
      Object.assign(S, msg.data);
      updateUI();
    }
  } catch(_){}
};

// ── Poll every 3s for full state sync ─────────────────────────────────────
function poll() {
  fetch('/api/state').then(r=>r.json()).then(s=>{Object.assign(S,s);renderAll();}).catch(()=>{});
}
setInterval(poll, 3000);
poll();

// ── Road animation loop ────────────────────────────────────────────────────
(function loop(){ drawRoad(); requestAnimationFrame(loop); })();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
