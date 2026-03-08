"""
CurriculumManager — ported from openenv/training/curriculum.py.

Same 4-stage progression and same reward thresholds. Adapted for
OverflowEnvironment: no ticket injection (the env has its own scripted
NPCs), stages instead control training logging and advancement criteria.

Stage 1  No extra pressure.  Goal: learn basic speed + lane keeping.
Stage 2  Standard traffic.   Goal: survive without crashing.
Stage 3  Evaluate more.      Goal: consistent goal-reaching.
Stage 4  Full evaluation.    Goal: high mean reward over long window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class StageConfig:
    stage:             int
    name:              str
    description:       str
    advance_threshold: float   # mean episode reward to advance
    advance_window:    int     # consecutive episodes required


STAGES: List[StageConfig] = [
    StageConfig(
        stage=1, name="Survival",
        description="Learn basic speed control and lane keeping.",
        advance_threshold=50.0, advance_window=8,
    ),
    StageConfig(
        stage=2, name="Crash Avoidance",
        description="Navigate traffic without colliding.",
        advance_threshold=120.0, advance_window=15,
    ),
    StageConfig(
        stage=3, name="Goal Reaching",
        description="Consistently reach the goal position.",
        advance_threshold=200.0, advance_window=15,
    ),
    StageConfig(
        stage=4, name="Mastery",
        description="High reward, smooth driving, minimal near-misses.",
        advance_threshold=280.0, advance_window=15,
    ),
]


class CurriculumManager:
    """
    Tracks stage progression based on episode rewards.
    Same API as openenv CurriculumManager — PPOTrainer calls it unchanged.
    """

    def __init__(self, seed: int = 0):
        self._stage_idx    = 0
        self._rewards: List[float] = []
        self._auto_advance = True

    @property
    def current_stage(self) -> int:
        return STAGES[self._stage_idx].stage

    @property
    def config(self) -> StageConfig:
        return STAGES[self._stage_idx]

    def step(self, sim_time: float) -> list:
        """No ticket injection in OverflowEnvironment — always returns []."""
        return []

    def record_episode_reward(self, reward: float) -> bool:
        """Record episode reward and advance stage if threshold met."""
        self._rewards.append(reward)
        cfg    = self.config
        window = self._rewards[-cfg.advance_window:]

        if (
            self._auto_advance
            and len(window) >= cfg.advance_window
            and sum(window) / len(window) >= cfg.advance_threshold
            and self._stage_idx < len(STAGES) - 1
        ):
            self._stage_idx += 1
            self._rewards = []
            print(f"[Curriculum] Advanced to Stage {self.current_stage}: {self.config.name}")
            return True
        return False

    def force_stage(self, stage: int) -> None:
        idx = stage - 1
        if 0 <= idx < len(STAGES):
            self._stage_idx = idx
            self._rewards   = []
            print(f"[Curriculum] Forced to Stage {stage}: {self.config.name}")
