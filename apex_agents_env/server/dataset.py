"""
Dataset loader for the Mercor APEX-Agents benchmark.

Loads the HuggingFace dataset, caches it in memory, and provides
task selection by task_id, domain, or random sampling.
"""

import random
from typing import Any, Dict, List, Optional

_DATASET_CACHE: Optional[List[Dict[str, Any]]] = None
_DOMAIN_INDEX: Optional[Dict[str, List[int]]] = None

DATASET_REPO = "mercor/apex-agents"


def _load_dataset() -> List[Dict[str, Any]]:
    """Load and cache the APEX-Agents dataset from HuggingFace."""
    global _DATASET_CACHE, _DOMAIN_INDEX

    if _DATASET_CACHE is not None:
        return _DATASET_CACHE

    from datasets import load_dataset

    ds = load_dataset(DATASET_REPO, split="train")
    _DATASET_CACHE = [dict(row) for row in ds]

    # Build domain index for fast filtering
    _DOMAIN_INDEX = {}
    for idx, task in enumerate(_DATASET_CACHE):
        domain = _normalize_domain(task.get("domain", "unknown"))
        _DOMAIN_INDEX.setdefault(domain, []).append(idx)

    return _DATASET_CACHE


def _normalize_domain(domain: str) -> str:
    """Normalize domain string to a canonical form."""
    d = domain.strip().lower().replace(" ", "_").replace("-", "_")
    # Map common variations
    if "law" in d:
        return "law"
    if "investment" in d or "banking" in d or "ib" in d:
        return "investment_banking"
    if "consult" in d or "management" in d:
        return "consulting"
    return d


def get_domains() -> List[str]:
    """Return list of available domains."""
    _load_dataset()
    return sorted(_DOMAIN_INDEX.keys()) if _DOMAIN_INDEX else []


def get_task(
    task_id: Optional[str] = None,
    domain: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Select a task from the dataset.

    Priority:
    1. task_id — select by exact ID
    2. domain — random task from specified domain
    3. random — any random task

    Args:
        task_id: Specific task identifier to retrieve.
        domain: Filter to this domain before random selection.
        rng: Random number generator for reproducibility.

    Returns:
        Task dict with keys from the HuggingFace dataset.

    Raises:
        ValueError: If task_id not found or domain has no tasks.
    """
    data = _load_dataset()
    if rng is None:
        rng = random.Random()

    # 1. By task_id
    if task_id is not None:
        for task in data:
            # Try matching on various ID fields
            tid = str(task.get("task_id", task.get("id", "")))
            if tid == str(task_id):
                return task
        # Also try by index
        try:
            idx = int(task_id)
            if 0 <= idx < len(data):
                return data[idx]
        except (ValueError, TypeError):
            pass
        raise ValueError(f"Task ID '{task_id}' not found in dataset")

    # 2. By domain
    if domain is not None:
        norm = _normalize_domain(domain)
        if _DOMAIN_INDEX and norm in _DOMAIN_INDEX:
            idx = rng.choice(_DOMAIN_INDEX[norm])
            return data[idx]
        raise ValueError(
            f"Domain '{domain}' not found. Available: {get_domains()}"
        )

    # 3. Random
    return rng.choice(data)


def get_task_count() -> int:
    """Return total number of tasks in the dataset."""
    return len(_load_dataset())
