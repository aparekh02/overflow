"""
Data models for the APEX-Agents Environment.

A professional task evaluation environment wrapping the Mercor APEX-Agents
benchmark (480 tasks across Law, Investment Banking, and Management Consulting).
Single-step episodes: reset returns a task prompt, step grades the response.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ── Sub-models ────────────────────────────────────────────────────────────


class RubricResult(BaseModel):
    """Result of evaluating a single rubric criterion."""

    criterion: str = Field(description="The rubric criterion text")
    score: int = Field(
        description="Judge score 1-10 for this criterion", ge=1, le=10
    )
    passed: bool = Field(description="Whether the criterion was met (score >= 7)")
    explanation: str = Field(default="", description="Judge explanation for the score")


# ── OpenEnv core models ──────────────────────────────────────────────────


class ApexAction(Action):
    """
    Action for the APEX-Agents environment.

    The agent provides a text response to a professional task prompt.
    """

    response: str = Field(
        default="", description="The agent's response to the task prompt"
    )


class ApexObservation(Observation):
    """
    Observation from the APEX-Agents environment.

    On reset: contains the task prompt and metadata.
    On step: contains grading results and reward breakdown.
    """

    # ── Task info ──
    task_prompt: str = Field(default="", description="The professional task prompt")
    domain: str = Field(
        default="", description="Task domain: law, investment_banking, or consulting"
    )
    task_id: str = Field(default="", description="Unique task identifier")

    # ── Grading results (populated after step) ──
    rubric_results: List[RubricResult] = Field(
        default_factory=list, description="Per-criterion grading results"
    )
    criteria_passed: int = Field(
        default=0, description="Number of rubric criteria passed"
    )
    criteria_total: int = Field(
        default=0, description="Total number of rubric criteria"
    )
    reward_mode: str = Field(
        default="capped", description="Reward mode used: capped or uncapped"
    )
    token_count: int = Field(
        default=0, description="Token count of the response (for uncapped mode)"
    )


class ApexState(State):
    """Internal state for the APEX-Agents environment."""

    task_id: str = Field(default="", description="Current task identifier")
    domain: str = Field(default="", description="Current task domain")
    reward_mode: str = Field(default="capped", description="Current reward mode")
    judge_model: str = Field(
        default="gpt-4o-mini", description="LLM judge model used for grading"
    )
    graded: bool = Field(default=False, description="Whether the response was graded")
