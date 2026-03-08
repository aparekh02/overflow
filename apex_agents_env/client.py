"""
APEX-Agents Environment Client.

Provides the client for connecting to an APEX-Agents Environment server
via WebSocket for persistent sessions.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    ApexAction,
    ApexObservation,
    ApexState,
    RubricResult,
)


class ApexAgentsEnv(EnvClient[ApexAction, ApexObservation, ApexState]):
    """
    WebSocket client for the APEX-Agents Environment.

    Example:
        >>> with ApexAgentsEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(domain="law")
        ...     print(result.observation.task_prompt)
        ...     action = ApexAction(response="Based on the applicable statutes...")
        ...     result = env.step(action)
        ...     print(f"Score: {result.observation.criteria_passed}/{result.observation.criteria_total}")
        ...     print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: ApexAction) -> Dict[str, Any]:
        """Convert ApexAction to JSON payload for step request."""
        return {"response": action.response}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ApexObservation]:
        """Parse server response into StepResult[ApexObservation]."""
        obs_data = payload.get("observation", {})

        rubric_results = [
            RubricResult(**r) for r in obs_data.get("rubric_results", [])
        ]

        observation = ApexObservation(
            task_prompt=obs_data.get("task_prompt", ""),
            domain=obs_data.get("domain", ""),
            task_id=obs_data.get("task_id", ""),
            rubric_results=rubric_results,
            criteria_passed=obs_data.get("criteria_passed", 0),
            criteria_total=obs_data.get("criteria_total", 0),
            reward_mode=obs_data.get("reward_mode", "capped"),
            token_count=obs_data.get("token_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ApexState:
        """Parse server response into ApexState."""
        return ApexState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            domain=payload.get("domain", ""),
            reward_mode=payload.get("reward_mode", "capped"),
            judge_model=payload.get("judge_model", "gpt-4o-mini"),
            graded=payload.get("graded", False),
        )
