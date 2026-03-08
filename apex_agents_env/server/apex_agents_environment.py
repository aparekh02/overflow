"""
APEX-Agents Environment Implementation.

Single-step episodes wrapping the Mercor APEX-Agents benchmark:
- reset() picks a task (by task_id, domain, or random), returns the prompt
- step() grades the response against rubric using LLM judge, returns reward + done=True
"""

import logging
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import ApexAction, ApexObservation, ApexState, RubricResult
from .dataset import get_task
from .grading import (
    compute_capped_reward,
    compute_uncapped_reward,
    count_tokens,
    grade_batch_async,
    grade_batch_sync,
)

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


class ApexAgentsEnvironment(Environment):
    """
    APEX-Agents benchmark environment.

    Single-step episodes: reset selects a professional task,
    step grades the agent's response via LLM-as-judge.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = ApexState(episode_id=str(uuid4()))
        self._task: Optional[Dict[str, Any]] = None
        self._rng = random.Random()
        self._done = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ApexObservation:
        """
        Reset the environment and select a task.

        Kwargs:
            task_id: Select a specific task by ID.
            domain: Filter to a domain (law, investment_banking, consulting).
            reward_mode: "capped" (default) or "uncapped".
            judge_model: LLM model for grading (default: gpt-4o-mini).
        """
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        task_id = kwargs.get("task_id")
        domain = kwargs.get("domain")
        reward_mode = kwargs.get("reward_mode", "capped")
        judge_model = kwargs.get("judge_model", DEFAULT_JUDGE_MODEL)

        # Select task
        self._task = get_task(task_id=task_id, domain=domain, rng=self._rng)
        self._done = False

        # Determine task metadata
        actual_task_id = str(
            self._task.get("task_id", self._task.get("id", "unknown"))
        )
        actual_domain = self._task.get("domain", "unknown")
        task_prompt = self._task.get(
            "prompt", self._task.get("task_prompt", self._task.get("question", ""))
        )

        self._state = ApexState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=actual_task_id,
            domain=actual_domain,
            reward_mode=reward_mode,
            judge_model=judge_model,
            graded=False,
        )

        return ApexObservation(
            task_prompt=task_prompt,
            domain=actual_domain,
            task_id=actual_task_id,
            reward_mode=reward_mode,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: ApexAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ApexObservation:
        """
        Grade the agent's response (synchronous).

        Returns observation with rubric results, reward, and done=True.
        """
        if self._done:
            return ApexObservation(
                task_prompt="",
                domain=self._state.domain,
                task_id=self._state.task_id,
                reward_mode=self._state.reward_mode,
                done=True,
                reward=0.0,
            )

        if self._task is None:
            return ApexObservation(
                task_prompt="No task loaded. Call reset() first.",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        self._done = True

        # Grade the response
        rubric_results = grade_batch_sync(
            task=self._task,
            response=action.response,
            judge_model=self._state.judge_model,
        )

        return self._build_graded_observation(action.response, rubric_results)

    async def step_async(
        self,
        action: ApexAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ApexObservation:
        """
        Grade the agent's response (async).

        Uses async OpenAI client for non-blocking judge calls.
        """
        if self._done:
            return ApexObservation(
                task_prompt="",
                domain=self._state.domain,
                task_id=self._state.task_id,
                reward_mode=self._state.reward_mode,
                done=True,
                reward=0.0,
            )

        if self._task is None:
            return ApexObservation(
                task_prompt="No task loaded. Call reset() first.",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        self._done = True

        # Grade the response (async)
        rubric_results = await grade_batch_async(
            task=self._task,
            response=action.response,
            judge_model=self._state.judge_model,
        )

        return self._build_graded_observation(action.response, rubric_results)

    def _build_graded_observation(
        self,
        response: str,
        rubric_results: List[RubricResult],
    ) -> ApexObservation:
        """Build the final observation after grading."""
        criteria_passed = sum(1 for r in rubric_results if r.passed)
        criteria_total = len(rubric_results)
        token_count = count_tokens(response)

        # Compute reward
        if self._state.reward_mode == "uncapped":
            reward = compute_uncapped_reward(
                criteria_passed, criteria_total, token_count
            )
        else:
            reward = compute_capped_reward(criteria_passed, criteria_total)

        self._state.graded = True

        return ApexObservation(
            task_prompt=self._task.get(
                "prompt",
                self._task.get("task_prompt", self._task.get("question", "")),
            )
            if self._task
            else "",
            domain=self._state.domain,
            task_id=self._state.task_id,
            rubric_results=rubric_results,
            criteria_passed=criteria_passed,
            criteria_total=criteria_total,
            reward_mode=self._state.reward_mode,
            token_count=token_count,
            done=True,
            reward=reward,
        )

    @property
    def state(self) -> ApexState:
        """Get the current environment state."""
        return self._state
