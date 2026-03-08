"""
LLM-as-judge grading for APEX-Agents.

Evaluates agent responses against rubric criteria using an OpenAI-compatible
LLM API. Supports batch evaluation (single call for all criteria) with
fallback to per-criterion calls if JSON parsing fails.
"""

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import RubricResult

logger = logging.getLogger(__name__)

# ── Judge configuration ──────────────────────────────────────────────────

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
PASSING_THRESHOLD = 7  # Score >= 7 means criterion passed


def _get_judge_client():
    """Create an OpenAI client for the judge model."""
    from openai import OpenAI

    api_key = os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    base_url = os.environ.get("JUDGE_BASE_URL", None)

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


async def _get_async_judge_client():
    """Create an async OpenAI client for the judge model."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    base_url = os.environ.get("JUDGE_BASE_URL", None)

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return AsyncOpenAI(**kwargs)


# ── Prompt templates ─────────────────────────────────────────────────────

BATCH_JUDGE_PROMPT = """You are an expert evaluator grading a professional's response to a task.

## Task Prompt
{task_prompt}

## Response to Evaluate
{response}

## Rubric Criteria
Evaluate the response against each criterion below. For each, assign a score from 1-10 where:
- 1-3: Poor, fails to address the criterion
- 4-6: Partial, addresses some aspects but has significant gaps
- 7-8: Good, adequately addresses the criterion
- 9-10: Excellent, thoroughly and expertly addresses the criterion

{criteria_list}

## Output Format
Return a JSON array with one object per criterion, in order:
```json
[
  {{"criterion": "...", "score": N, "explanation": "brief reason"}},
  ...
]
```

Return ONLY the JSON array, no other text."""

SINGLE_CRITERION_PROMPT = """You are an expert evaluator. Grade this response against ONE criterion.

## Task Prompt
{task_prompt}

## Response
{response}

## Criterion
{criterion}

Score from 1-10 (7+ = pass). Return JSON:
{{"score": N, "explanation": "brief reason"}}

Return ONLY the JSON object."""


# ── Grading functions ────────────────────────────────────────────────────


def _extract_criteria(task: Dict[str, Any]) -> List[str]:
    """Extract rubric criteria from a task dict."""
    # The dataset may store criteria in different fields
    criteria = task.get("rubric_criteria", task.get("rubric", task.get("criteria", [])))

    if isinstance(criteria, str):
        # Try JSON parse
        try:
            criteria = json.loads(criteria)
        except json.JSONDecodeError:
            # Split by newlines or numbered items
            lines = [
                line.strip()
                for line in re.split(r"\n\d+[\.\)]\s*|\n[-•]\s*", criteria)
                if line.strip()
            ]
            criteria = lines if lines else [criteria]

    if isinstance(criteria, list):
        result = []
        for c in criteria:
            if isinstance(c, dict):
                # Handle {"criteria": "...", ...} format from APEX dataset
                text = c.get("criteria", c.get("criterion", str(c)))
            else:
                text = str(c)
            text = text.strip()
            if text:
                result.append(text)
        return result

    return []


def _parse_judge_json(text: str) -> Any:
    """Extract and parse JSON from judge response text."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding array or object
    for pattern in [r"\[.*\]", r"\{.*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return None


def grade_batch_sync(
    task: Dict[str, Any],
    response: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> List[RubricResult]:
    """
    Grade a response against all rubric criteria in a single LLM call.

    Falls back to per-criterion calls if batch JSON parsing fails.
    """
    criteria = _extract_criteria(task)
    if not criteria:
        logger.warning("No rubric criteria found for task")
        return []

    task_prompt = task.get("prompt", task.get("task_prompt", task.get("question", "")))

    # Build criteria list
    criteria_text = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(criteria)
    )

    prompt = BATCH_JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        response=response,
        criteria_list=criteria_text,
    )

    client = _get_judge_client()

    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        result_text = resp.choices[0].message.content or ""
        parsed = _parse_judge_json(result_text)

        if isinstance(parsed, list) and len(parsed) == len(criteria):
            results = []
            for i, item in enumerate(parsed):
                score = int(item.get("score", 1))
                score = max(1, min(10, score))
                results.append(
                    RubricResult(
                        criterion=criteria[i],
                        score=score,
                        passed=score >= PASSING_THRESHOLD,
                        explanation=str(item.get("explanation", "")),
                    )
                )
            return results

        logger.warning("Batch parse failed, falling back to per-criterion grading")
    except Exception as e:
        logger.warning(f"Batch grading failed: {e}, falling back to per-criterion")

    # Fallback: grade each criterion individually
    return _grade_individual_sync(task_prompt, response, criteria, client, judge_model)


def _grade_individual_sync(
    task_prompt: str,
    response: str,
    criteria: List[str],
    client,
    judge_model: str,
) -> List[RubricResult]:
    """Grade each criterion with a separate LLM call."""
    results = []
    for criterion in criteria:
        prompt = SINGLE_CRITERION_PROMPT.format(
            task_prompt=task_prompt,
            response=response,
            criterion=criterion,
        )
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            result_text = resp.choices[0].message.content or ""
            parsed = _parse_judge_json(result_text)

            if isinstance(parsed, dict):
                score = int(parsed.get("score", 1))
                score = max(1, min(10, score))
                results.append(
                    RubricResult(
                        criterion=criterion,
                        score=score,
                        passed=score >= PASSING_THRESHOLD,
                        explanation=str(parsed.get("explanation", "")),
                    )
                )
                continue
        except Exception as e:
            logger.warning(f"Individual grading failed for criterion: {e}")

        # Default: fail
        results.append(
            RubricResult(
                criterion=criterion,
                score=1,
                passed=False,
                explanation="Grading failed",
            )
        )

    return results


async def grade_batch_async(
    task: Dict[str, Any],
    response: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> List[RubricResult]:
    """
    Async version of grade_batch_sync.

    Uses async OpenAI client for non-blocking judge calls.
    """
    criteria = _extract_criteria(task)
    if not criteria:
        logger.warning("No rubric criteria found for task")
        return []

    task_prompt = task.get("prompt", task.get("task_prompt", task.get("question", "")))

    criteria_text = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(criteria)
    )

    prompt = BATCH_JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        response=response,
        criteria_list=criteria_text,
    )

    client = await _get_async_judge_client()

    try:
        resp = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        result_text = resp.choices[0].message.content or ""
        parsed = _parse_judge_json(result_text)

        if isinstance(parsed, list) and len(parsed) == len(criteria):
            results = []
            for i, item in enumerate(parsed):
                score = int(item.get("score", 1))
                score = max(1, min(10, score))
                results.append(
                    RubricResult(
                        criterion=criteria[i],
                        score=score,
                        passed=score >= PASSING_THRESHOLD,
                        explanation=str(item.get("explanation", "")),
                    )
                )
            return results

        logger.warning("Batch parse failed, falling back to per-criterion grading")
    except Exception as e:
        logger.warning(f"Async batch grading failed: {e}, falling back to per-criterion")

    # Fallback: grade each criterion individually (async)
    return await _grade_individual_async(
        task_prompt, response, criteria, client, judge_model
    )


async def _grade_individual_async(
    task_prompt: str,
    response: str,
    criteria: List[str],
    client,
    judge_model: str,
) -> List[RubricResult]:
    """Async per-criterion grading fallback."""
    results = []
    for criterion in criteria:
        prompt = SINGLE_CRITERION_PROMPT.format(
            task_prompt=task_prompt,
            response=response,
            criterion=criterion,
        )
        try:
            resp = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            result_text = resp.choices[0].message.content or ""
            parsed = _parse_judge_json(result_text)

            if isinstance(parsed, dict):
                score = int(parsed.get("score", 1))
                score = max(1, min(10, score))
                results.append(
                    RubricResult(
                        criterion=criterion,
                        score=score,
                        passed=score >= PASSING_THRESHOLD,
                        explanation=str(parsed.get("explanation", "")),
                    )
                )
                continue
        except Exception as e:
            logger.warning(f"Async individual grading failed: {e}")

        results.append(
            RubricResult(
                criterion=criterion,
                score=1,
                passed=False,
                explanation="Grading failed",
            )
        )

    return results


# ── Reward computation ───────────────────────────────────────────────────


def compute_capped_reward(passed: int, total: int) -> float:
    """
    Capped reward: fraction of criteria passed.

    Returns:
        Float in [0, 1].
    """
    if total == 0:
        return 0.0
    return passed / total


def compute_uncapped_reward(passed: int, total: int, token_count: int) -> float:
    """
    Uncapped reward: correctness * logarithmic token bonus.

    Formula: (passed / total) * (1 + log2(1 + tokens/256))

    Correctness is multiplicative — wrong answers always get 0 regardless
    of length. Logarithmic token scaling rewards thoroughness with
    diminishing returns.

    Returns:
        Float in [0, ~6+].
    """
    if total == 0 or passed == 0:
        return 0.0
    correctness = passed / total
    token_bonus = 1.0 + math.log2(1.0 + token_count / 256.0)
    return correctness * token_bonus


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken, with chars/4 fallback.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)
