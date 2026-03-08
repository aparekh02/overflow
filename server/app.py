"""
Overflow OpenENV — Continuous PPO Training + Live Web Dashboard

Runs PPO training in a background thread and exposes:
  GET /          → live HTML dashboard (road animation + reward charts)
  GET /api/state → JSON snapshot of current training state
  GET /api/stream → SSE stream of state updates
  POST /api/mode  → switch reward mode {capped, uncapped}

Reward modes:
  capped   — standard shaped reward, capped at REWARD_CAP per step
  uncapped — base reward + token_bonus (scales with LLM reasoning token count)
             frontier models that produce more reasoning tokens earn more reward
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
    from ..training.overflow_gym_env import OverflowGymEnv, _obs_to_vector, _action_to_decision
    from ..training.curriculum import CurriculumManager, STAGES
    from ..training.reward import compute_reward, compute_episode_bonus, W_COLLISION
    from ..training.ppo_trainer import PPOTrainer, RolloutBuffer
    from ..policies.flat_mlp_policy import FlatMLPPolicy
    from ..policies.ticket_attention_policy import TicketAttentionPolicy
    from ..policies.policy_spec import OBS_DIM
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.overflow_gym_env import OverflowGymEnv, _obs_to_vector, _action_to_decision
    from training.curriculum import CurriculumManager, STAGES
    from training.reward import compute_reward, compute_episode_bonus, W_COLLISION
    from training.ppo_trainer import PPOTrainer, RolloutBuffer
    from policies.flat_mlp_policy import FlatMLPPolicy
    from policies.ticket_attention_policy import TicketAttentionPolicy
    from policies.policy_spec import OBS_DIM

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Reward mode config ────────────────────────────────────────────────────────

REWARD_CAP     = 2.0          # per-step ceiling in capped mode
TOKEN_SCALE    = 0.001        # uncapped: reward += tokens * TOKEN_SCALE
MAX_TOKEN_BONUS = 5.0         # uncapped mode bonus ceiling per step

# ── Shared training state (updated by training thread, read by API) ───────────

@dataclass
class CarSnapshot:
    car_id: int
    x: float
    y: float
    lane: int
    speed: float

@dataclass
class EpisodeRecord:
    episode: int
    steps: int
    reward: float
    outcome: str   # "crash" | "goal" | "timeout"
    stage: int
    reward_mode: str

@dataclass
class TrainingState:
    # Current frame
    cars: List[CarSnapshot] = field(default_factory=list)
    ego_x: float = 0.0
    ego_lane: int = 2
    # Live metrics
    total_steps: int = 0
    n_updates: int = 0
    n_episodes: int = 0
    episode_reward: float = 0.0
    episode_steps: int = 0
    mean_reward_100: float = 0.0
    mean_ep_len: float = 0.0
    stage: int = 1
    stage_name: str = "Survival"
    reward_mode: str = "capped"
    steps_per_sec: float = 0.0
    # History (capped at 500 for the chart)
    reward_history: List[float] = field(default_factory=list)
    episode_history: List[Dict] = field(default_factory=list)
    # PPO update metrics
    last_pg_loss: float = 0.0
    last_vf_loss: float = 0.0
    last_entropy: float = 0.0
    running: bool = True
    error: Optional[str] = None


_state = TrainingState()
_state_lock = threading.Lock()
_sse_queue: Deque[str] = deque(maxlen=50)

# ── Reward mode switch (thread-safe) ─────────────────────────────────────────

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
        return min(base_reward, REWARD_CAP) if base_reward > 0 else base_reward


# ── Training thread ───────────────────────────────────────────────────────────

def _push_sse(data: dict) -> None:
    _sse_queue.append(json.dumps(data))

def _snapshot_cars(overflow_obs) -> List[CarSnapshot]:
    snaps = []
    if not overflow_obs or not overflow_obs.cars:
        return snaps
    for c in overflow_obs.cars:
        snaps.append(CarSnapshot(
            car_id=c.carId,
            x=c.position.x,
            y=c.position.y if hasattr(c.position, "y") else (c.lane - 2) * 3.7,
            lane=c.lane,
            speed=c.speed,
        ))
    return snaps

def _training_loop() -> None:
    global _state
    try:
        # Build policy + env + curriculum
        policy = TicketAttentionPolicy(obs_dim=OBS_DIM)
        env = OverflowGymEnv()
        curriculum = CurriculumManager()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)

        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

        # PPO hyperparams
        GAMMA      = 0.99
        GAE_LAMBDA = 0.95
        CLIP       = 0.2
        ENT_COEF   = 0.02
        VF_COEF    = 0.5
        MAX_GRAD   = 0.5
        N_STEPS    = 512
        BATCH_SIZE = 128
        N_EPOCHS   = 6

        buf = RolloutBuffer(N_STEPS, OBS_DIM, device)

        ep_rewards: Deque[float] = deque(maxlen=100)
        ep_lengths: Deque[int]   = deque(maxlen=100)

        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        total_steps = 0
        n_updates = 0
        n_episodes = 0
        t0 = time.time()

        while True:
            buf.reset()
            policy.eval()

            # ── Collect rollout ──────────────────────────────────────────────
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

                # Apply reward mode
                reward = apply_reward_mode(base_reward)

                buf.add(obs, action.cpu().numpy(), reward, float(val), float(logp), float(term or trunc))

                obs        = next_obs
                ep_reward += reward
                ep_steps  += 1
                total_steps += 1

                # Update live car positions from OverflowEnvironment._cars
                cars = []
                overflow_env = env._env  # OverflowEnvironment instance
                if hasattr(overflow_env, "_cars"):
                    for c in overflow_env._cars:
                        cars.append(CarSnapshot(
                            car_id=c.car_id,
                            x=c.position,
                            y=(c.lane - 2) * 3.7,
                            lane=c.lane,
                            speed=c.speed,
                        ))

                with _state_lock:
                    _state.total_steps = total_steps
                    _state.episode_reward = ep_reward
                    _state.episode_steps = ep_steps
                    _state.steps_per_sec = total_steps / max(time.time() - t0, 1.0)
                    if cars:
                        _state.cars = cars
                        ego = next((c for c in cars if c.car_id == 0), None)
                        if ego:
                            _state.ego_x = ego.x
                            _state.ego_lane = ego.lane

                if term or trunc:
                    bonus = compute_episode_bonus(
                        total_steps=ep_steps,
                        survived=not info.get("collision", False),
                    )
                    ep_reward += bonus
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(ep_steps)
                    n_episodes += 1

                    advanced = curriculum.record_episode_reward(ep_reward)
                    outcome = (
                        "crash" if info.get("collision") else
                        ("goal" if info.get("goal_reached") else "timeout")
                    )

                    ep_rec = {
                        "episode":     n_episodes,
                        "steps":       ep_steps,
                        "reward":      round(ep_reward, 3),
                        "outcome":     outcome,
                        "stage":       curriculum.current_stage,
                        "reward_mode": get_reward_mode(),
                    }

                    with _state_lock:
                        _state.n_episodes = n_episodes
                        _state.stage = curriculum.current_stage
                        _state.stage_name = curriculum.config.name
                        _state.mean_reward_100 = float(np.mean(ep_rewards))
                        _state.mean_ep_len = float(np.mean(ep_lengths))
                        _state.reward_history.append(round(ep_reward, 3))
                        if len(_state.reward_history) > 500:
                            _state.reward_history = _state.reward_history[-500:]
                        _state.episode_history.append(ep_rec)
                        if len(_state.episode_history) > 200:
                            _state.episode_history = _state.episode_history[-200:]

                    _push_sse({"type": "episode", "data": ep_rec})

                    obs, _ = env.reset()
                    ep_reward = 0.0
                    ep_steps  = 0

            # ── PPO update ───────────────────────────────────────────────────
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _, last_val = policy(obs_t.unsqueeze(0))
            buf.compute_returns(float(last_val), GAMMA, GAE_LAMBDA)

            policy.train()

            all_obs      = buf.obs
            all_acts     = buf.acts
            old_logp     = buf.logp
            adv          = buf.ret - buf.val
            adv          = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret          = buf.ret
            old_val      = buf.val
            indices      = torch.randperm(N_STEPS, device=device)

            pg_losses, vf_losses, entropies = [], [], []

            for _ in range(N_EPOCHS):
                for start in range(0, N_STEPS, BATCH_SIZE):
                    idx = indices[start: start + BATCH_SIZE]
                    act_mean, val = policy(all_obs[idx])
                    val = val.squeeze(-1)

                    dist     = torch.distributions.Normal(act_mean, torch.ones_like(act_mean) * 0.3)
                    logp     = dist.log_prob(all_acts[idx]).sum(dim=-1)
                    entropy  = dist.entropy().sum(dim=-1).mean()

                    ratio    = torch.exp(logp - old_logp[idx])
                    pg_loss  = torch.max(-adv[idx] * ratio, -adv[idx] * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()

                    val_clip = old_val[idx] + (val - old_val[idx]).clamp(-CLIP, CLIP)
                    vf_loss  = 0.5 * torch.max((val - ret[idx]) ** 2, (val_clip - ret[idx]) ** 2).mean()

                    loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
                    optimizer.step()

                    pg_losses.append(float(pg_loss))
                    vf_losses.append(float(vf_loss))
                    entropies.append(float(entropy))

            n_updates += 1
            with _state_lock:
                _state.n_updates     = n_updates
                _state.last_pg_loss  = round(float(np.mean(pg_losses)), 5)
                _state.last_vf_loss  = round(float(np.mean(vf_losses)), 5)
                _state.last_entropy  = round(float(np.mean(entropies)), 5)

            _push_sse({
                "type": "update",
                "data": {
                    "n_updates":   n_updates,
                    "total_steps": total_steps,
                    "mean_reward": round(float(np.mean(ep_rewards)) if ep_rewards else 0.0, 3),
                    "stage":       curriculum.current_stage,
                    "pg_loss":     round(float(np.mean(pg_losses)), 5),
                    "vf_loss":     round(float(np.mean(vf_losses)), 5),
                    "entropy":     round(float(np.mean(entropies)), 5),
                }
            })

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        with _state_lock:
            _state.running = False
            _state.error = f"{exc}\n\n{tb}"
        print(f"[Training] ERROR: {exc}\n{tb}", flush=True)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Overflow OpenENV")

@app.on_event("startup")
def _start_training():
    t = threading.Thread(target=_training_loop, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {"status": "ok"}


class ModeRequest(BaseModel):
    mode: str   # "capped" or "uncapped"

@app.post("/api/mode")
def set_mode(req: ModeRequest):
    if req.mode not in ("capped", "uncapped"):
        return JSONResponse({"error": "mode must be 'capped' or 'uncapped'"}, status_code=400)
    set_reward_mode(req.mode)
    return {"mode": req.mode}


@app.get("/api/state")
def get_state():
    with _state_lock:
        s = _state
        return {
            "total_steps":    s.total_steps,
            "n_updates":      s.n_updates,
            "n_episodes":     s.n_episodes,
            "episode_reward": round(s.episode_reward, 3),
            "episode_steps":  s.episode_steps,
            "mean_reward":    round(s.mean_reward_100, 3),
            "mean_ep_len":    round(s.mean_ep_len, 1),
            "stage":          s.stage,
            "stage_name":     s.stage_name,
            "reward_mode":    s.reward_mode,
            "steps_per_sec":  round(s.steps_per_sec, 1),
            "pg_loss":        s.last_pg_loss,
            "vf_loss":        s.last_vf_loss,
            "entropy":        s.last_entropy,
            "reward_history": s.reward_history[-200:],
            "episode_history": s.episode_history[-50:],
            "cars":           [asdict(c) for c in s.cars],
            "ego_x":          s.ego_x,
            "ego_lane":       s.ego_lane,
            "running":        s.running,
            "error":          s.error,
        }


@app.get("/api/stream")
async def sse_stream():
    async def generator():
        last_idx = len(_sse_queue)
        while True:
            current = list(_sse_queue)
            for msg in current[last_idx:]:
                yield f"data: {msg}\n\n"
            last_idx = len(current)
            # Also send a heartbeat state snapshot every 2s
            with _state_lock:
                s = _state
                snap = {
                    "type": "tick",
                    "data": {
                        "total_steps":    s.total_steps,
                        "episode_reward": round(s.episode_reward, 3),
                        "episode_steps":  s.episode_steps,
                        "stage":          s.stage,
                        "stage_name":     s.stage_name,
                        "reward_mode":    s.reward_mode,
                        "cars":           [asdict(c) for c in s.cars],
                        "ego_x":          s.ego_x,
                    }
                }
            yield f"data: {json.dumps(snap)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(generator(), media_type="text/event-stream")


# ── HTML Dashboard ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Overflow OpenENV — Live Training</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0a0f; color: #e0e0e0; font-family: 'Courier New', monospace; height: 100vh; display: flex; flex-direction: column; }
  header { background: #12121a; border-bottom: 1px solid #2a2a40; padding: 10px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; color: #7eb8ff; letter-spacing: 2px; }
  .badge { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; }
  .badge-running { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
  .badge-error { background: #3a1a1a; color: #f44336; border: 1px solid #f44336; }
  .mode-toggle { margin-left: auto; display: flex; gap: 8px; align-items: center; }
  .mode-btn { padding: 4px 14px; border-radius: 8px; border: 1px solid #444; background: #1a1a2a; color: #aaa; cursor: pointer; font-size: 12px; transition: all 0.2s; }
  .mode-btn.active { background: #1a3a5a; color: #7eb8ff; border-color: #7eb8ff; }
  main { flex: 1; display: flex; gap: 0; overflow: hidden; }
  .left-panel { flex: 0 0 55%; display: flex; flex-direction: column; padding: 16px; gap: 12px; border-right: 1px solid #2a2a40; overflow: hidden; }
  .right-panel { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow: hidden; }
  .panel-title { font-size: 11px; color: #667; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
  canvas { border-radius: 8px; }
  #road-canvas { width: 100%; height: 200px; background: #111118; border: 1px solid #2a2a40; border-radius: 8px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .metric-card { background: #12121a; border: 1px solid #2a2a40; border-radius: 8px; padding: 10px; }
  .metric-label { font-size: 10px; color: #667; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 18px; color: #7eb8ff; font-weight: bold; margin-top: 2px; }
  .metric-sub { font-size: 10px; color: #556; margin-top: 2px; }
  .stage-bar { background: #12121a; border: 1px solid #2a2a40; border-radius: 8px; padding: 12px; }
  .stage-stages { display: flex; gap: 6px; margin-top: 8px; }
  .stage-pip { flex: 1; height: 6px; border-radius: 3px; background: #2a2a40; transition: background 0.5s; }
  .stage-pip.active { background: #7eb8ff; }
  .stage-pip.done { background: #4caf50; }
  #reward-canvas { width: 100%; flex: 1; min-height: 160px; background: #12121a; border: 1px solid #2a2a40; border-radius: 8px; }
  .episode-table { flex: 1; overflow-y: auto; background: #12121a; border: 1px solid #2a2a40; border-radius: 8px; min-height: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { padding: 6px 10px; color: #667; text-align: left; border-bottom: 1px solid #2a2a40; position: sticky; top: 0; background: #1a1a28; }
  td { padding: 5px 10px; border-bottom: 1px solid #1a1a28; }
  tr:hover td { background: #1a1a2a; }
  .outcome-crash { color: #f44336; }
  .outcome-goal  { color: #4caf50; }
  .outcome-timeout { color: #ff9800; }
  .ppo-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .ppo-stat { background: #12121a; border: 1px solid #2a2a40; border-radius: 6px; padding: 6px 12px; font-size: 11px; }
  .ppo-stat span { color: #7eb8ff; }
  .error-box { background: #2a0a0a; border: 1px solid #f44336; border-radius: 8px; padding: 12px; font-size: 11px; color: #f44336; white-space: pre-wrap; overflow-y: auto; max-height: 200px; }
</style>
</head>
<body>
<header>
  <h1>OVERFLOW OPENENV</h1>
  <span id="status-badge" class="badge badge-running">TRAINING</span>
  <div class="mode-toggle">
    <span style="font-size:11px;color:#667">REWARD MODE:</span>
    <button class="mode-btn active" id="btn-capped" onclick="setMode('capped')">CAPPED</button>
    <button class="mode-btn" id="btn-uncapped" onclick="setMode('uncapped')">UNCAPPED (LLM)</button>
  </div>
</header>
<main>
  <!-- LEFT: road + metrics + stage -->
  <div class="left-panel">
    <div>
      <div class="panel-title">Road View — Live</div>
      <canvas id="road-canvas"></canvas>
    </div>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Total Steps</div>
        <div class="metric-value" id="m-steps">0</div>
        <div class="metric-sub" id="m-sps">0 sps</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Episodes</div>
        <div class="metric-value" id="m-eps">0</div>
        <div class="metric-sub" id="m-eplen">avg len: 0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Mean Reward (100)</div>
        <div class="metric-value" id="m-reward">0.00</div>
        <div class="metric-sub" id="m-curr-r">ep: 0.00</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">PPO Updates</div>
        <div class="metric-value" id="m-updates">0</div>
        <div class="metric-sub">policy gradient</div>
      </div>
    </div>
    <div class="stage-bar">
      <div class="panel-title">Curriculum Stage — <span id="stage-name">Survival</span></div>
      <div class="stage-stages">
        <div class="stage-pip active" id="pip-1"></div>
        <div class="stage-pip" id="pip-2"></div>
        <div class="stage-pip" id="pip-3"></div>
        <div class="stage-pip" id="pip-4"></div>
      </div>
    </div>
    <div>
      <div class="panel-title">PPO Losses</div>
      <div class="ppo-row">
        <div class="ppo-stat">Policy: <span id="pg-loss">—</span></div>
        <div class="ppo-stat">Value: <span id="vf-loss">—</span></div>
        <div class="ppo-stat">Entropy: <span id="entropy">—</span></div>
      </div>
    </div>
    <div id="error-section" style="display:none">
      <div class="panel-title" style="color:#f44336">Error</div>
      <div class="error-box" id="error-text"></div>
    </div>
  </div>

  <!-- RIGHT: reward chart + episode table -->
  <div class="right-panel">
    <div style="flex:0 0 auto">
      <div class="panel-title">Reward History</div>
    </div>
    <canvas id="reward-canvas"></canvas>
    <div style="flex:0 0 auto">
      <div class="panel-title">Episode Log</div>
    </div>
    <div class="episode-table">
      <table>
        <thead><tr>
          <th>#</th><th>Steps</th><th>Reward</th><th>Outcome</th><th>Stage</th><th>Mode</th>
        </tr></thead>
        <tbody id="ep-tbody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let state = { cars: [], reward_history: [], episode_history: [], stage: 1, reward_mode: 'capped' };

// ── Road canvas ────────────────────────────────────────────────────────────
const roadCanvas = document.getElementById('road-canvas');
const roadCtx    = roadCanvas.getContext('2d');
const N_LANES    = 3;
const LANE_H     = 40;
const ROAD_PAD   = 20;

function resizeRoad() {
  roadCanvas.width  = roadCanvas.offsetWidth;
  roadCanvas.height = roadCanvas.offsetHeight;
}
resizeRoad();
window.addEventListener('resize', resizeRoad);

function laneY(lane) {
  // lane 1..3, top = lane 1
  const totalH = N_LANES * LANE_H;
  const offsetY = (roadCanvas.height - totalH) / 2;
  return offsetY + (lane - 1) * LANE_H + LANE_H / 2;
}

function carX(x, egoX) {
  // Center ego at 30% of canvas
  const w = roadCanvas.width;
  const scale = 0.8;   // pixels per unit
  return w * 0.3 + (x - egoX) * scale;
}

function drawRoad() {
  const w = roadCanvas.width, h = roadCanvas.height;
  roadCtx.clearRect(0, 0, w, h);

  // Road background
  const totalH = N_LANES * LANE_H;
  const offsetY = (h - totalH) / 2;
  roadCtx.fillStyle = '#1a1a28';
  roadCtx.fillRect(0, offsetY, w, totalH);

  // Lane dividers
  roadCtx.setLineDash([20, 15]);
  roadCtx.strokeStyle = '#3a3a50';
  roadCtx.lineWidth = 1;
  for (let i = 1; i < N_LANES; i++) {
    const y = offsetY + i * LANE_H;
    roadCtx.beginPath();
    roadCtx.moveTo(0, y);
    roadCtx.lineTo(w, y);
    roadCtx.stroke();
  }
  roadCtx.setLineDash([]);

  // Road edges
  roadCtx.strokeStyle = '#5a5a80';
  roadCtx.lineWidth = 2;
  roadCtx.beginPath(); roadCtx.moveTo(0, offsetY); roadCtx.lineTo(w, offsetY); roadCtx.stroke();
  roadCtx.beginPath(); roadCtx.moveTo(0, offsetY + totalH); roadCtx.lineTo(w, offsetY + totalH); roadCtx.stroke();

  // Cars
  const egoX = state.ego_x || 0;
  for (const car of state.cars || []) {
    const cx = carX(car.x, egoX);
    if (cx < -50 || cx > w + 50) continue;
    const cy = laneY(car.lane);
    const isEgo = car.car_id === 0;

    // Car body
    roadCtx.save();
    roadCtx.translate(cx, cy);
    const cw = 36, ch = 18;
    roadCtx.fillStyle = isEgo ? '#2a5a9a' : '#3a2a1a';
    roadCtx.strokeStyle = isEgo ? '#7eb8ff' : '#ff9800';
    roadCtx.lineWidth = isEgo ? 2 : 1;
    roadCtx.beginPath();
    roadCtx.roundRect(-cw/2, -ch/2, cw, ch, 4);
    roadCtx.fill();
    roadCtx.stroke();

    // Label
    roadCtx.fillStyle = isEgo ? '#7eb8ff' : '#ff9800';
    roadCtx.font = isEgo ? 'bold 9px Courier New' : '8px Courier New';
    roadCtx.textAlign = 'center';
    roadCtx.textBaseline = 'middle';
    roadCtx.fillText(isEgo ? 'EGO' : `C${car.car_id}`, 0, 0);
    roadCtx.restore();
  }
}

// ── Reward chart ────────────────────────────────────────────────────────────
const rwCanvas = document.getElementById('reward-canvas');
const rwCtx    = rwCanvas.getContext('2d');

function resizeRw() {
  rwCanvas.width  = rwCanvas.offsetWidth;
  rwCanvas.height = rwCanvas.offsetHeight;
}
resizeRw();
window.addEventListener('resize', resizeRw);

function drawRewardChart() {
  const w = rwCanvas.width, h = rwCanvas.height;
  rwCtx.clearRect(0, 0, w, h);

  const hist = state.reward_history || [];
  if (hist.length < 2) {
    rwCtx.fillStyle = '#667';
    rwCtx.font = '12px Courier New';
    rwCtx.textAlign = 'center';
    rwCtx.fillText('Waiting for episodes...', w/2, h/2);
    return;
  }

  const pad = { top: 10, right: 10, bottom: 30, left: 50 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;

  const minR = Math.min(...hist);
  const maxR = Math.max(...hist);
  const rangeR = maxR - minR || 1;

  const xScale = pw / (hist.length - 1);
  const yScale = ph / rangeR;

  // Grid
  rwCtx.strokeStyle = '#1e1e2e';
  rwCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + ph * (i / 4);
    rwCtx.beginPath(); rwCtx.moveTo(pad.left, y); rwCtx.lineTo(pad.left + pw, y); rwCtx.stroke();
    const val = maxR - rangeR * (i / 4);
    rwCtx.fillStyle = '#556';
    rwCtx.font = '9px Courier New';
    rwCtx.textAlign = 'right';
    rwCtx.fillText(val.toFixed(1), pad.left - 4, y + 3);
  }

  // Zero line
  if (minR < 0 && maxR > 0) {
    const zy = pad.top + (maxR / rangeR) * ph;
    rwCtx.strokeStyle = '#3a3a50';
    rwCtx.lineWidth = 1;
    rwCtx.setLineDash([4, 4]);
    rwCtx.beginPath(); rwCtx.moveTo(pad.left, zy); rwCtx.lineTo(pad.left + pw, zy); rwCtx.stroke();
    rwCtx.setLineDash([]);
  }

  // Moving average (window=10)
  const MA = 10;
  const ma = hist.map((_, i) => {
    const sl = hist.slice(Math.max(0, i - MA + 1), i + 1);
    return sl.reduce((a, b) => a + b, 0) / sl.length;
  });

  // Raw line
  rwCtx.strokeStyle = 'rgba(126,184,255,0.25)';
  rwCtx.lineWidth = 1;
  rwCtx.beginPath();
  hist.forEach((v, i) => {
    const x = pad.left + i * xScale;
    const y = pad.top + (maxR - v) * yScale;
    i === 0 ? rwCtx.moveTo(x, y) : rwCtx.lineTo(x, y);
  });
  rwCtx.stroke();

  // MA line
  rwCtx.strokeStyle = '#7eb8ff';
  rwCtx.lineWidth = 2;
  rwCtx.beginPath();
  ma.forEach((v, i) => {
    const x = pad.left + i * xScale;
    const y = pad.top + (maxR - v) * yScale;
    i === 0 ? rwCtx.moveTo(x, y) : rwCtx.lineTo(x, y);
  });
  rwCtx.stroke();

  // X axis label
  rwCtx.fillStyle = '#556';
  rwCtx.font = '9px Courier New';
  rwCtx.textAlign = 'center';
  rwCtx.fillText(`Episodes (${hist.length})`, pad.left + pw / 2, h - 6);
}

// ── Episode table ───────────────────────────────────────────────────────────
function updateEpisodeTable() {
  const tbody = document.getElementById('ep-tbody');
  const episodes = (state.episode_history || []).slice().reverse().slice(0, 50);
  tbody.innerHTML = episodes.map(ep => `
    <tr>
      <td>${ep.episode}</td>
      <td>${ep.steps}</td>
      <td style="color:${ep.reward >= 0 ? '#4caf50' : '#f44336'}">${ep.reward.toFixed(2)}</td>
      <td class="outcome-${ep.outcome}">${ep.outcome.toUpperCase()}</td>
      <td>${ep.stage}</td>
      <td style="color:#ff9800">${ep.mode || ep.reward_mode || 'capped'}</td>
    </tr>
  `).join('');
}

// ── Metric update ───────────────────────────────────────────────────────────
function updateMetrics(s) {
  document.getElementById('m-steps').textContent = s.total_steps?.toLocaleString() || '0';
  document.getElementById('m-sps').textContent   = `${(s.steps_per_sec||0).toFixed(0)} sps`;
  document.getElementById('m-eps').textContent    = s.n_episodes || '0';
  document.getElementById('m-eplen').textContent  = `avg len: ${(s.mean_ep_len||0).toFixed(0)}`;
  document.getElementById('m-reward').textContent = (s.mean_reward||0).toFixed(2);
  document.getElementById('m-curr-r').textContent = `ep: ${(s.episode_reward||0).toFixed(2)}`;
  document.getElementById('m-updates').textContent = s.n_updates || '0';
  document.getElementById('pg-loss').textContent  = s.pg_loss ?? '—';
  document.getElementById('vf-loss').textContent  = s.vf_loss ?? '—';
  document.getElementById('entropy').textContent  = s.entropy ?? '—';
  document.getElementById('stage-name').textContent = s.stage_name || 'Survival';

  const stage = s.stage || 1;
  for (let i = 1; i <= 4; i++) {
    const pip = document.getElementById(`pip-${i}`);
    pip.className = 'stage-pip' + (i < stage ? ' done' : (i === stage ? ' active' : ''));
  }

  // Mode buttons
  const mode = s.reward_mode || 'capped';
  document.getElementById('btn-capped').className   = 'mode-btn' + (mode === 'capped' ? ' active' : '');
  document.getElementById('btn-uncapped').className = 'mode-btn' + (mode === 'uncapped' ? ' active' : '');

  if (s.error) {
    document.getElementById('error-section').style.display = 'block';
    document.getElementById('error-text').textContent = s.error;
    document.getElementById('status-badge').textContent = 'ERROR';
    document.getElementById('status-badge').className = 'badge badge-error';
  }
}

// ── Mode switch ─────────────────────────────────────────────────────────────
function setMode(mode) {
  fetch('/api/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  }).then(r => r.json()).then(d => {
    state.reward_mode = d.mode;
    updateMetrics(state);
  });
}

// ── Poll /api/state every 2s (SSE as supplement) ───────────────────────────
function poll() {
  fetch('/api/state').then(r => r.json()).then(s => {
    Object.assign(state, s);
    drawRoad();
    drawRewardChart();
    updateMetrics(s);
    updateEpisodeTable();
  }).catch(() => {});
}
setInterval(poll, 2000);
poll();

// ── SSE for fast episode events ─────────────────────────────────────────────
const evtSrc = new EventSource('/api/stream');
evtSrc.onmessage = (e) => {
  try {
    const msg = JSON.parse(e.data);
    if (msg.type === 'episode') {
      if (!state.episode_history) state.episode_history = [];
      state.episode_history.push(msg.data);
      if (!state.reward_history) state.reward_history = [];
      state.reward_history.push(msg.data.reward);
      updateEpisodeTable();
      drawRewardChart();
    } else if (msg.type === 'tick') {
      Object.assign(state, msg.data);
      drawRoad();
    } else if (msg.type === 'update') {
      Object.assign(state, msg.data);
      updateMetrics(state);
    }
  } catch(err) {}
};

// Render loop for smooth road animation
function renderLoop() {
  drawRoad();
  requestAnimationFrame(renderLoop);
}
renderLoop();
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
