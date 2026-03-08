"""
SimMiddleware — bidirectional WebSocket bridge between WaymoEnv (Python) and
the React/Three.js 3D simulation (browser).

Novel: The middleware is not one-directional telemetry — it's a two-way
coupling layer. The 3D sim can:
  • Submit incident tickets → starts new RL episode
  • Freeze/unfreeze the simulation → enables counterfactual exploration
  • Inject a "ghost action" → manual override of review agent for demonstration
  • Stream Waymo parquet frame stats → update live environment constraints

Message protocol (all JSON):
  Browser → Python:
    { "type": "load_incident",  "ticket": { "id": "...", "text": "..." } }
    { "type": "pause" }
    { "type": "resume" }
    { "type": "reset" }
    { "type": "waymo_frame",   "data": { "pointCount": 42000, "boxes": [...] } }
    { "type": "ghost_action",  "action": [steer, throttle, brake] }

  Python → Browser:
    { "type": "scene_update",  "state": { ... } }
    { "type": "episode_start", "incident_type": "...", "constraints": { ... } }
    { "type": "episode_end",   "total_reward": ..., "steps": ... }
    { "type": "error",         "message": "..." }
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("[SimMiddleware] 'websockets' package not found. Install with: pip install websockets")

logger = logging.getLogger("openenv.middleware")


class SimMiddleware:
    """
    Async WebSocket server. One instance serves one WaymoEnv.

    Usage:
        env = WaymoEnv(render_mode="websocket")
        middleware = SimMiddleware(env, host="localhost", port=8765)
        asyncio.run(middleware.serve())   # blocking — run in background task
    """

    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 8765

    def __init__(
        self,
        env: Any,  # WaymoEnv
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets package required: pip install websockets")

        self._env = env
        self._host = host
        self._port = port
        self._clients: Set[WebSocketServerProtocol] = set()
        self._paused: bool = False
        self._ghost_action: Optional[Any] = None
        self._shutdown_event = asyncio.Event()

        # Register ourselves with the env
        env.attach_middleware(self)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Start the WebSocket server. This coroutine runs until shutdown()."""
        logger.info(f"[SimMiddleware] Listening on ws://{self._host}:{self._port}")
        print(f"[SimMiddleware] 3D sim can connect at ws://{self._host}:{self._port}")

        async with websockets.serve(self._handler, self._host, self._port):
            await self._shutdown_event.wait()

        logger.info("[SimMiddleware] Server shut down.")

    async def shutdown(self) -> None:
        """Gracefully stop the server."""
        self._shutdown_event.set()
        for ws in list(self._clients):
            await ws.close()

    # ── Outbound: broadcast to all 3D sim clients ─────────────────────────────

    async def broadcast_state(self, state: dict) -> None:
        """Send scene state to all connected browser clients."""
        if not self._clients:
            return
        message = json.dumps(state, default=_json_default)
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def broadcast_episode_start(self, incident_type: str, constraints: dict) -> None:
        await self.broadcast_state({
            "type": "episode_start",
            "incident_type": incident_type,
            "constraints": constraints,
        })

    async def broadcast_episode_end(self, total_reward: float, steps: int) -> None:
        await self.broadcast_state({
            "type": "episode_end",
            "total_reward": round(total_reward, 4),
            "steps": steps,
        })

    # ── Inbound: handle messages from 3D sim ──────────────────────────────────

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        logger.info(f"[SimMiddleware] Client connected: {ws.remote_address}")

        # Announce connection
        await ws.send(json.dumps({
            "type": "connected",
            "message": "OpenENV middleware ready",
            "host": self._host,
            "port": self._port,
        }))

        # Push current scene state immediately so 3D sim shows agents right away
        if hasattr(self._env, '_ego') and self._env._ego is not None:
            try:
                state = self._env._serialize_state(0.0, False, self._env._make_info())
                await ws.send(json.dumps(state, default=_json_default))
            except Exception:
                pass

        try:
            async for raw in ws:
                await self._handle_message(ws, raw)
        except Exception as e:
            logger.warning(f"[SimMiddleware] Client error: {e}")
        finally:
            self._clients.discard(ws)
            logger.info(f"[SimMiddleware] Client disconnected: {ws.remote_address}")

    async def _handle_message(self, ws: WebSocketServerProtocol, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            await ws.send(json.dumps({"type": "error", "message": f"Invalid JSON: {e}"}))
            return

        msg_type = msg.get("type", "")

        if msg_type == "load_incident":
            await self._on_load_incident(ws, msg)

        elif msg_type == "pause":
            self._paused = True
            logger.info("[SimMiddleware] Simulation paused")
            await ws.send(json.dumps({"type": "paused"}))

        elif msg_type == "resume":
            self._paused = False
            logger.info("[SimMiddleware] Simulation resumed")
            await ws.send(json.dumps({"type": "resumed"}))

        elif msg_type == "reset":
            await self._on_reset(ws, msg)

        elif msg_type == "waymo_frame":
            # Live Waymo data from the browser (parquet frame stats)
            frame_data = msg.get("data", {})
            self._env.update_waymo_frame(frame_data)
            logger.debug(f"[SimMiddleware] Waymo frame update: {frame_data.get('pointCount', 0)} pts")

        elif msg_type == "ghost_action":
            # Manual action injection from browser
            action = msg.get("action")
            if action and len(action) == 3:
                import numpy as np
                self._ghost_action = np.array(action, dtype=np.float32)
                logger.info(f"[SimMiddleware] Ghost action set: {action}")

        elif msg_type == "clear_ghost":
            self._ghost_action = None

        elif msg_type == "ping":
            await ws.send(json.dumps({"type": "pong"}))

        else:
            await ws.send(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            }))

    async def _on_load_incident(self, ws: WebSocketServerProtocol, msg: dict) -> None:
        from .incident_parser import IncidentTicket
        ticket_data = msg.get("ticket", {})
        ticket = IncidentTicket(
            ticket_id=ticket_data.get("id", "ws_ticket"),
            raw_text=ticket_data.get("text", ""),
            severity=ticket_data.get("severity", "moderate"),
        )
        try:
            obs, info = self._env.reset(options={"ticket": ticket})
            await self.broadcast_episode_start(
                incident_type=info.get("incident_type", "unknown"),
                constraints=info.get("constraints", {}),
            )
            logger.info(f"[SimMiddleware] Episode started: {info.get('incident_type')}")
        except Exception as e:
            await ws.send(json.dumps({"type": "error", "message": str(e)}))

    async def _on_reset(self, ws: WebSocketServerProtocol, msg: dict) -> None:
        try:
            obs, info = self._env.reset()
            await ws.send(json.dumps({
                "type": "episode_start",
                "incident_type": info.get("incident_type", "unknown"),
                "constraints": info.get("constraints", {}),
            }))
        except Exception as e:
            await ws.send(json.dumps({"type": "error", "message": str(e)}))

    # ── Ghost action access ───────────────────────────────────────────────────

    @property
    def is_paused(self) -> bool:
        return self._paused

    def consume_ghost_action(self):
        """Returns a one-shot ghost action if set, then clears it."""
        action = self._ghost_action
        self._ghost_action = None
        return action

    @property
    def client_count(self) -> int:
        return len(self._clients)


# ── JSON serialiser ───────────────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    """Handle non-serialisable types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── Convenience: run env + middleware together ────────────────────────────────

async def run_env_with_middleware(
    env: Any,
    policy: Any = None,
    host: str = SimMiddleware.DEFAULT_HOST,
    port: int = SimMiddleware.DEFAULT_PORT,
    max_episodes: int = 100,
    seed_ticket: Any = None,
) -> None:
    """
    Runs the environment loop alongside the WebSocket server.

    Each episode replays the SAME incident (seed_ticket) from the same
    start state so the review agent can learn across repeated attempts.

    policy: callable(obs) → action. If None, uses random policy.
    """
    middleware = SimMiddleware(env, host, port)

    async def env_loop():
        ep = 0
        while ep < max_episodes:
            # Always reset with the same seed ticket → deterministic episode start
            reset_opts = {"ticket": seed_ticket} if seed_ticket is not None else None
            obs, info = env.reset(options=reset_opts)

            await middleware.broadcast_episode_start(
                info.get("incident_type", "unknown"),
                info.get("constraints", {}),
            )
            logger.info(
                f"[Episode {ep+1}] type={info.get('incident_type')}  "
                f"weather={info.get('constraints', {}).get('weather', '?')}"
            )

            done = False
            while not done:
                if middleware.is_paused:
                    await asyncio.sleep(0.05)
                    continue

                # Ghost action = manual override from 3D sim (one shot)
                action = middleware.consume_ghost_action()
                if action is None:
                    if policy is not None:
                        action, _ = policy.predict(obs)
                    else:
                        action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 10 Hz pacing — matches Waymo frame rate
                await asyncio.sleep(0.1)

            await middleware.broadcast_episode_end(
                info.get("episode_reward", 0.0),
                info.get("step", 0),
            )
            logger.info(
                f"[Episode {ep+1}] done  reward={info.get('episode_reward', 0):+.2f}  "
                f"steps={info.get('step', 0)}"
            )
            ep += 1

            # Brief pause between episodes so user can see the outcome
            await asyncio.sleep(1.5)

    # Run WebSocket server and env loop concurrently
    await asyncio.gather(
        middleware.serve(),
        env_loop(),
    )
