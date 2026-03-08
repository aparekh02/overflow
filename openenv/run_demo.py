"""
run_demo.py — Launch OpenENV with the 3D simulation middleware.

Usage:
    python openenv/run_demo.py

    Then open the 3D sim (npm run dev in 3D_sim/) and click "OpenENV" mode.
    Submit an incident ticket to start an RL episode with live 3D visualization.

Options:
    --headless    Run without WebSocket server (just train)
    --episode N   Number of episodes (default 100)
    --port PORT   WebSocket port (default 8765)
"""

import asyncio
import argparse
import logging
import os
import subprocess
import sys
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv import WaymoEnv, SimMiddleware
from openenv.incident_parser import IncidentTicket
from openenv.middleware import run_env_with_middleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Absolute path to the 3D sim directory (sibling of openenv/)
SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "3D_sim")
SIM_URL = "http://localhost:5173"


def _start_3d_sim() -> subprocess.Popen:
    """Launch `npm run dev` in 3D_sim/ and return the process handle."""
    print(f"[Demo] Starting 3D sim: npm run dev  ({SIM_DIR})")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=SIM_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait until Vite reports it's ready (up to 15 s)
    deadline = time.time() + 15
    for line in proc.stdout:  # type: ignore[union-attr]
        print(f"  [vite] {line.rstrip()}")
        if "localhost" in line and ("5173" in line or "ready" in line.lower()):
            break
        if time.time() > deadline:
            break

    return proc


EXAMPLE_TICKETS = [
    IncidentTicket(
        ticket_id="INC-001",
        raw_text=(
            "Waymo AV was traveling at 28 mph on Market St in rainy conditions. "
            "A pedestrian jaywalked suddenly from between parked cars at 15m ahead. "
            "AV detected the pedestrian and braked hard, narrowly avoiding contact. "
            "Speed limit: 30 mph. High traffic density."
        ),
        severity="critical",
    ),
    IncidentTicket(
        ticket_id="INC-002",
        raw_text=(
            "Vehicle approaching intersection at 35 mph, fog conditions, 35 mph zone. "
            "Cross traffic ran a red light at approximately 60 ft distance. "
            "Ego vehicle braked and swerved right, near-miss recorded. "
            "Two additional trailing vehicles within 20m behind ego."
        ),
        severity="severe",
    ),
    IncidentTicket(
        ticket_id="INC-003",
        raw_text=(
            "On I-280 highway, vehicle in adjacent lane initiated abrupt lane change "
            "into ego's lane without signaling. Relative closing speed ~5 mph. "
            "Clear weather, 65 mph limit, light traffic. "
            "Ego decelerated and moved to shoulder to avoid sideswipe."
        ),
        severity="moderate",
    ),
]


async def main(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  OpenENV — Waymo Incident RL Environment")
    print("=" * 60)

    env = WaymoEnv(
        render_mode="websocket" if not args.headless else "none",
        max_steps=200,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    if args.headless:
        print("[Demo] Running in headless mode (no WebSocket server)")
        _run_headless(env, args.episodes)
        return

    # 1. Start the 3D simulation frontend
    vite_proc = _start_3d_sim()

    # 2. Brief pause then open browser
    await asyncio.sleep(1.5)
    print(f"[Demo] Opening 3D sim at {SIM_URL}?openenv=1")
    webbrowser.open(f"{SIM_URL}?openenv=1")
    print("[Demo] Click 'OpenENV ●' (top-right) to connect to the RL backend.")
    print()

    print(f"[Demo] WebSocket server → ws://localhost:{args.port}")
    print("[Demo] Example tickets available:")
    for t in EXAMPLE_TICKETS:
        print(f"  [{t.ticket_id}] {t.raw_text[:70]}...")
    print()

    seed = EXAMPLE_TICKETS[0]  # same ticket every episode = deterministic replay
    print(f"[Demo] All episodes seeded from: [{seed.ticket_id}] {seed.raw_text[:60]}...")

    try:
        await run_env_with_middleware(
            env=env,
            policy=None,        # swap in a trained SB3 model: policy=model
            host="localhost",
            port=args.port,
            max_episodes=args.episodes,
            seed_ticket=seed,
        )
    finally:
        vite_proc.terminate()
        print("[Demo] 3D sim stopped.")


def _run_headless(env: WaymoEnv, n_episodes: int) -> None:
    """Headless training loop — no WebSocket, no 3D sim."""
    import numpy as np
    for ep in range(n_episodes):
        ticket = EXAMPLE_TICKETS[ep % len(EXAMPLE_TICKETS)]
        obs, info = env.reset(options={"ticket": ticket})
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()  # replace with trained policy
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc

        print(f"  Episode {ep+1:3d}/{n_episodes}  "
              f"ticket={ticket.ticket_id}  "
              f"type={info['incident_type']:<12}  "
              f"reward={total_reward:+8.2f}  "
              f"steps={info['step']}")

    print("\nHeadless run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenENV demo runner")
    parser.add_argument("--headless", action="store_true", help="No WebSocket server")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    args = parser.parse_args()

    asyncio.run(main(args))
